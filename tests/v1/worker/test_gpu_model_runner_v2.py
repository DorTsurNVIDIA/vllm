# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.config import CacheConfig, ParallelConfig
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.standalone_ar import speculator as ar_module
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states


def test_non_last_pp_rank_uses_global_batch_for_sample_feedback():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_last_pp_rank = False
    local_batch = object()
    global_batch = SimpleNamespace(idx_mapping=object())
    runner.pcp_manager = SimpleNamespace(
        global_batch=global_batch,
        restore_for_sampling=Mock(),
    )
    runner.pp_handler = SimpleNamespace(receive=Mock(return_value=False))
    runner.postprocess_num_computed_tokens = Mock()
    runner.model_state = SimpleNamespace(postprocess_state=Mock())
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.eplb = SimpleNamespace(step=Mock())
    runner.execute_model_state = ExecuteModelState(
        input_batch=local_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )

    runner.sample_tokens(None)

    runner.pp_handler.receive.assert_called_once_with(global_batch)
    runner.postprocess_num_computed_tokens.assert_called_once_with(global_batch)
    runner.model_state.postprocess_state.assert_called_once_with(
        global_batch.idx_mapping, 0
    )
    runner.pcp_manager.restore_for_sampling.assert_not_called()


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args, **kwargs: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""
    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


def _make_capture_runner(captured: bool) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False, capture_inner_cudagraphs=lambda *args: None
    )
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
    )
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda _cfg: contextlib.nullcontext()
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = None
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = None
    runner.attn_groups = None
    runner.kv_cache_config = None
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []


@pytest.mark.parametrize("target_buffer", ["absent", "none", "tensor"])
def test_get_drafter_hidden_states_tolerates_missing_target_buffer(target_buffer):
    """Targets allocate the MTP hidden buffer only for hidden-state drafters."""
    hidden_states = torch.zeros(4, 8)
    buffer = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
    if target_buffer == "absent":
        model = SimpleNamespace()
    else:
        returned = buffer if target_buffer == "tensor" else None
        model = SimpleNamespace(get_mtp_target_hidden_states=lambda: returned)

    out = get_drafter_hidden_states(model, hidden_states)

    if target_buffer == "tensor":
        assert torch.equal(out, buffer[:4])
    else:
        assert out is hidden_states


def test_async_copy_to_np_does_not_alias_reused_buffer():
    buffer = torch.zeros(4, dtype=torch.int64)

    snapshot = async_copy_to_np(buffer)
    buffer.fill_(1)

    assert snapshot.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("draft_dtype", "expected_dtype"),
    [("auto", "auto"), ("bfloat16", "bfloat16"), (None, "fp8_e4m3")],
)
def test_standalone_draft_kv_dtype_override(monkeypatch, draft_dtype, expected_dtype):
    @dataclass
    class Config:
        cache_config: CacheConfig
        parallel_config: ParallelConfig
        model_config: object
        quant_config: object | None = None

    target = Config(CacheConfig(cache_dtype="fp8_e4m3"), ParallelConfig(), object())
    speculator = object.__new__(ar_module.StandaloneARSpeculator)
    speculator.vllm_config = target
    speculator.draft_model_config = object()
    speculator.speculative_config = SimpleNamespace(
        kv_cache_dtype=draft_dtype, draft_parallel_config=ParallelConfig()
    )
    speculator.max_num_reqs = 4
    get_model = Mock(return_value=torch.nn.Module())
    monkeypatch.setattr(ar_module, "get_model", get_model)
    monkeypatch.setattr(ar_module, "set_model_tag", lambda _: contextlib.nullcontext())

    speculator.load_draft_model(object(), set())

    draft_config = get_model.call_args.kwargs["vllm_config"]
    assert draft_config.cache_config.cache_dtype == expected_dtype
    assert speculator.attn_vllm_config.cache_config.cache_dtype == expected_dtype
    assert target.cache_config.cache_dtype == "fp8_e4m3"


# These tests use a tiny independently evaluated recurrence so an incorrect
# restore/commit boundary cannot pass just by agreeing with another packer.
class MambaMixer2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = "draft.mamba"
        self.kv_cache = (torch.zeros(8, 3, 2), torch.zeros(8, 2))

    def get_state_shape(self):
        return ((3, 2), (2,))

    def get_state_dtype(self):
        return (torch.float32, torch.float32)


@pytest.fixture
def recurrent_replay():
    from vllm.v1.worker.gpu.spec_decode.standalone_ar.hybrid_replay import (
        HybridDraftReplay,
    )

    layer = MambaMixer2()
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=False, mamba_cache_mode="none"
        ),
        model_config=SimpleNamespace(enforce_eager=True),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    owner = SimpleNamespace(
        max_num_reqs=4,
        device=torch.device("cpu"),
        vllm_config=cfg,
        draft_attn_layer_names={layer.prefix},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=object.__new__(MambaSpec), layer_names=[layer.prefix]
                )
            ]
        ),
        block_tables=SimpleNamespace(
            input_block_tables=[torch.tensor([[5], [1], [3], [0]])]
        ),
        expanded_input_ids=torch.zeros(128, dtype=torch.int32),
        expanded_positions=torch.zeros(128, dtype=torch.int64),
        input_buffers=SimpleNamespace(
            query_start_loc=torch.zeros(5, dtype=torch.int32),
            seq_lens=torch.zeros(4, dtype=torch.int32),
        ),
        last_token_indices=torch.zeros(4, dtype=torch.int64),
        draft_is_prefilling=torch.zeros(4, dtype=torch.bool),
    )
    replay = HybridDraftReplay(owner, layer)
    replay.bind()
    return replay, owner, layer


def _replay_batch(slots, segments, starts):
    import numpy as np

    lengths = [len(s) for s in segments]
    return SimpleNamespace(
        num_reqs=len(slots),
        num_tokens=sum(lengths),
        idx_mapping=torch.tensor(slots),
        input_ids=torch.tensor(sum(segments, []), dtype=torch.int32),
        positions=torch.tensor(
            [p for start, n in zip(starts, lengths) for p in range(start, start + n)]
        ),
        query_start_loc_np=np.asarray([0, *np.cumsum(lengths)], dtype=np.int32),
    )


def _reference_recurrence(tokens, conv=None, ssm=None):
    conv = torch.zeros(3, 2) if conv is None else conv.clone()
    ssm = torch.zeros(2) if ssm is None else ssm.clone()
    for t in tokens:
        conv = torch.cat([conv[1:], torch.tensor([[float(t), float(t + 1)]])])
        ssm = 0.5 * ssm + conv.sum(dim=0)
    return conv, ssm


def _apply_packed(replay, owner, layer, qsl):
    for packed_row, original_row in enumerate(replay.order.tolist()):
        block = int(owner.block_tables.input_block_tables[0][original_row, 0])
        a, b = map(int, qsl[packed_row : packed_row + 2])
        old = (
            (None, None)
            if bool(replay.prefilling[packed_row])
            else tuple(c[block] for c in layer.kv_cache)
        )
        states = _reference_recurrence(owner.expanded_input_ids[a:b].tolist(), *old)
        for cache, value in zip(layer.kv_cache, states):
            cache[block].copy_(value)
    replay.commit()


@pytest.mark.parametrize("accepted", range(8))
def test_committed_state_excludes_rejected_proposals(recurrent_replay, accepted):
    replay, owner, layer = recurrent_replay
    prompt = [1, 2, 3, 4, 5, 6]
    first = _replay_batch([2], [prompt], [0])
    _, qsl, _ = replay.prepare(
        first, torch.tensor([0]), torch.tensor([1]), torch.tensor([0, 0, 70, 0])
    )
    _apply_packed(replay, owner, layer, qsl)
    # A speculative transaction advances only the working cache.
    for cache in layer.kv_cache:
        cache[5].fill_(999)
    proposed = list(range(20, 27))
    nxt = _replay_batch([2], [[70] + proposed], [6])
    total, qsl, _ = replay.prepare(
        nxt,
        torch.tensor([7 - accepted]),
        torch.tensor([accepted + 1]),
        torch.tensor([0, 0, 80, 0]),
    )
    assert owner.expanded_input_ids[:total].tolist() == proposed[:accepted] + [80]
    _apply_packed(replay, owner, layer, qsl)
    expected = _reference_recurrence(prompt + [70] + proposed[:accepted] + [80])
    for cache, want in zip(layer.kv_cache, expected):
        torch.testing.assert_close(cache[5], want)
    assert replay.committed_end[2] == 8 + accepted


def test_chunked_prefill_processes_each_token_once(recurrent_replay):
    replay, owner, layer = recurrent_replay
    for start, tokens, sampled in [(0, [1, 2, 3], 0), (3, [4, 5], 0), (5, [6, 7], 1)]:
        batch = _replay_batch([1], [tokens], [start])
        _, qsl, _ = replay.prepare(
            batch,
            torch.tensor([0]),
            torch.tensor([sampled]),
            torch.tensor([0, 99, 0, 0]),
        )
        _apply_packed(replay, owner, layer, qsl)
        for c in layer.kv_cache:
            c[5].fill_(-50)  # discarded speculative/intermediate working state
    expected = _reference_recurrence([1, 2, 3, 4, 5, 6, 7, 99])
    backups = replay.snapshots[layer.prefix][1]
    for saved, want in zip(backups, expected):
        torch.testing.assert_close(saved[1], want)


def test_request_reordering_reuse_and_independent_states(recurrent_replay):
    replay, owner, layer = recurrent_replay
    batch = _replay_batch([2, 0], [[1, 2, 3], [4, 5]], [0, 0])
    _, qsl, _ = replay.prepare(
        batch, torch.tensor([0, 0]), torch.tensor([1, 1]), torch.tensor([90, 0, 80, 0])
    )
    _apply_packed(replay, owner, layer, qsl)
    batch = _replay_batch(
        [2, 0], [[80] + list(range(10, 17)), [90] + list(range(30, 37))], [3, 2]
    )
    _, qsl, _ = replay.prepare(
        batch, torch.tensor([0, 7]), torch.tensor([8, 1]), torch.tensor([92, 0, 82, 0])
    )
    assert replay.order.tolist() == [1, 0]
    assert owner.last_token_indices[:2].tolist() == [8, 0]
    _apply_packed(replay, owner, layer, qsl)
    owner.block_tables.input_block_tables[0][:2] = torch.tensor([[1], [5]])
    batch = _replay_batch([0, 2], [[92, 12], [82, 13]], [3, 11])
    _, qsl, _ = replay.prepare(
        batch, torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([93, 0, 83, 0])
    )
    _apply_packed(replay, owner, layer, qsl)
    expected = [
        _reference_recurrence([4, 5, 90, 92, 93]),
        _reference_recurrence([1, 2, 3, 80] + list(range(10, 17)) + [82, 83]),
    ]
    for block, states in zip([1, 5], expected):
        for cache, want in zip(layer.kv_cache, states):
            torch.testing.assert_close(cache[block], want)
    batch = _replay_batch([0], [[11, 12]], [0])
    _, qsl, _ = replay.prepare(
        batch, torch.tensor([0]), torch.tensor([0]), torch.zeros(4)
    )
    _apply_packed(replay, owner, layer, qsl)
    for cache, want in zip(layer.kv_cache, _reference_recurrence([11, 12])):
        torch.testing.assert_close(cache[1], want)


def test_snapshot_storage_preexists_cache_binding(recurrent_replay, monkeypatch):
    replay, _, layer = recurrent_replay
    assert replay.snapshot_bytes == 4 * (3 * 2 + 2) * 4
    pointers = [s.data_ptr() for s in replay.snapshots[layer.prefix][1]]

    def unexpected(*args, **kwargs):
        raise AssertionError("binding must not allocate snapshot storage")

    monkeypatch.setattr(torch, "empty", unexpected)
    replay.layers.clear()
    replay.bind()
    assert pointers == [s.data_ptr() for s in replay.snapshots[layer.prefix][1]]


def test_snapshot_allocation_is_visible_to_model_memory_profiler(recurrent_replay):
    from vllm.utils.mem_utils import DeviceMemoryProfiler
    from vllm.v1.worker.gpu.spec_decode.standalone_ar.hybrid_replay import (
        HybridDraftReplay,
    )

    _, owner, layer = recurrent_replay
    owner.device = torch.device("cuda")
    with DeviceMemoryProfiler() as profile:
        replay = HybridDraftReplay(owner, layer)
    assert profile.consumed_memory >= replay.snapshot_bytes > 0


@pytest.mark.parametrize(
    "field,value",
    [("prefix", True), ("eager", False), ("async", True), ("mode", "align")],
)
def test_recurrent_replay_rejects_unsupported_modes(recurrent_replay, field, value):
    from vllm.v1.worker.gpu.spec_decode.standalone_ar.hybrid_replay import (
        HybridDraftReplay,
    )

    _, owner, layer = recurrent_replay
    settings = {
        "prefix": (owner.vllm_config.cache_config, "enable_prefix_caching"),
        "eager": (owner.vllm_config.model_config, "enforce_eager"),
        "async": (owner.vllm_config.scheduler_config, "async_scheduling"),
        "mode": (owner.vllm_config.cache_config, "mamba_cache_mode"),
    }
    obj, name = settings[field]
    setattr(obj, name, value)
    with pytest.raises(NotImplementedError, match="eager synchronous"):
        HybridDraftReplay(owner, layer)
