# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
    StandaloneARSpeculator,
)


@pytest.fixture(scope="module")
def config():
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    return EngineArgs(
        model="Qwen/Qwen3-0.6B",
        revision=revision,
        tokenizer_revision=revision,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=512,
        max_num_seqs=32,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        speculative_config={
            "method": "draft_model",
            "model": "Qwen/Qwen3-0.6B",
            "revision": revision,
            "num_speculative_tokens": 16,
            "draft_sample_method": "probabilistic",
            "enable_adaptive_verification": False,
            "rejection_sample_method": "standard",
        },
    ).create_engine_config()


@pytest.mark.parametrize("real,padded", [(3, 4), (15, 16), (19, 24), (31, 32), (8, 8)])
@pytest.mark.parametrize("marker", [-1, 151936])
@pytest.mark.parametrize("skip_attn", [False, True])
def test_inactive_decode_inputs_are_legal(config, real, padded, marker, skip_attn):
    device = torch.device("cuda")
    with set_current_vllm_config(config):
        speculator = StandaloneARSpeculator(config, device)
    batch = InputBatch.make_dummy(
        real, real, InputBuffers(32, 2048, device), is_padding=False
    )
    # Unavailable proposals may retain the -1 marker. Values of absent rows
    # cannot be used as model embedding indices, even when attention is empty.
    speculator.draft_tokens.fill_(marker)
    expected = torch.arange(7, 7 + real, dtype=torch.int64, device=device)
    speculator.draft_tokens[:real, 0].copy_(expected)
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, padded, padded)
    speculator._prepare_decode_inputs(
        batch,
        torch.zeros(real, dtype=torch.int64, device=device),
        real,
        skip_attn,
        descriptor,
    )
    actual = speculator.input_buffers.input_ids
    torch.testing.assert_close(actual[:real].long(), expected, rtol=0, atol=0)
    assert bool((speculator.input_buffers.seq_lens[real:padded] == 0).all())
    assert bool(
        (speculator.input_buffers.query_start_loc[real + 1 : padded + 1] == real).all()
    )
    ghost = actual[real:padded]
    assert bool(((ghost >= 0) & (ghost < 151936)).all()), (
        "inactive rows contain invalid embedding IDs"
    )
    assert bool((ghost == 0).all())
