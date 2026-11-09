# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager committed-state replay for an external Mamba2 draft.

Scope: synchronous V2, no prefix caching, Mamba cache mode 'none'.
State backups are allocated during model loading, before KV-capacity profiling.
"""

import numpy as np
import torch

from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata


def replay_plan(start, query_length, rejected, sampled, committed_end):
    if not (0 <= rejected < query_length and sampled >= 0):
        raise ValueError("invalid target transaction")
    end = start + query_length - rejected
    saved_end = start if committed_end is None else committed_end
    if not start <= saved_end <= end:
        raise ValueError("committed recurrent state does not cover target input prefix")
    correction = sampled > 0
    indices = list(range(saved_end - start, query_length - rejected))
    if not indices and not correction:
        raise ValueError("empty recurrent update")
    return indices, correction, end + int(correction)


class HybridDraftReplay:
    """Committed draft snapshots allocated before KV-capacity profiling."""

    def __init__(self, owner, draft_model):
        self.owner = owner
        state_layers = {
            layer.prefix: layer
            for layer in draft_model.modules()
            if hasattr(layer, "get_state_shape") and hasattr(layer, "kv_cache")
        }
        self.enabled = bool(state_layers)
        self.layers = []
        self.snapshots = {}
        self.initialized = np.zeros(owner.max_num_reqs, dtype=bool)
        self.committed_end = np.zeros(owner.max_num_reqs, dtype=np.int64)
        if not self.enabled:
            return
        cfg = owner.vllm_config
        if (
            cfg.cache_config.enable_prefix_caching
            or cfg.cache_config.mamba_cache_mode != "none"
            or not cfg.model_config.enforce_eager
            or cfg.scheduler_config.async_scheduling
        ):
            raise NotImplementedError(
                "Hybrid draft replay requires eager synchronous execution "
                "without prefix caching and mamba_cache_mode=none"
            )
        for name, layer in state_layers.items():
            if type(layer).__name__ != "MambaMixer2" or len(layer.kv_cache) != 2:
                raise NotImplementedError(f"Unsupported hybrid draft layer: {name}")
            shapes, dtypes = layer.get_state_shape(), layer.get_state_dtype()
            if len(shapes) != 2 or len(dtypes) != 2:
                raise NotImplementedError(f"Expected two native Mamba2 states: {name}")
            self.snapshots[name] = (
                layer,
                tuple(
                    torch.empty(
                        (owner.max_num_reqs, *shape),
                        dtype=dtype,
                        device=owner.device,
                    )
                    for shape, dtype in zip(shapes, dtypes)
                ),
            )
        self.snapshot_bytes = sum(
            state.numel() * state.element_size()
            for _, states in self.snapshots.values()
            for state in states
        )

    def bind(self):
        """Bind profiled snapshots to allocated draft caches without allocating."""
        o = self.owner
        bound = set()
        for gid, group in enumerate(o.kv_cache_config.kv_cache_groups):
            if not isinstance(group.kv_cache_spec, MambaSpec):
                continue
            for name in set(group.layer_names) & o.draft_attn_layer_names:
                if name not in self.snapshots:
                    raise ValueError(f"Missing profiled draft snapshot: {name}")
                layer, backups = self.snapshots[name]
                for cache, backup in zip(layer.kv_cache, backups):
                    if (
                        cache.shape[1:] != backup.shape[1:]
                        or cache.dtype != backup.dtype
                        or cache.device != backup.device
                    ):
                        raise ValueError(f"Draft state changed after profiling: {name}")
                    self.layers.append((gid, cache, backup))
                bound.add(name)
        if bound != self.snapshots.keys():
            raise ValueError("Profiled draft state does not match allocated caches")

    def prepare(self, batch, num_rejected, num_sampled, last_sampled):
        """Restore committed state; pack only new valid tokens and correction."""
        if not self.layers:
            # set_attn runs before native cache tensors replace layer placeholders.
            # Storage was already allocated and profiled during model loading.
            self.bind()
        o = self.owner
        n = batch.num_reqs
        ids_cpu = batch.idx_mapping[:n].cpu().tolist()
        if any(i < 0 for i in ids_cpu):
            raise NotImplementedError(
                "Filtered/sentinel rows require explicit handling"
            )
        if len(set(ids_cpu)) != n:
            raise ValueError("Duplicate request state slots")
        positions = batch.positions[: batch.num_tokens].cpu().tolist()
        rejected = num_rejected[:n].cpu().tolist()
        sampled = num_sampled[:n].cpu().tolist()
        qsl = batch.query_start_loc_np
        gather, correction_rows, correction_offsets = [], [], []
        packed_qsl, packed_positions, ends, continuing = [0], [], [], []
        for row, slot in enumerate(ids_cpu):
            a, b = int(qsl[row]), int(qsl[row + 1])
            start = positions[a]
            if start == 0:
                self.initialized[slot] = False
            elif not self.initialized[slot]:
                raise ValueError("Missing committed draft state for continuation")
            old_end = int(self.committed_end[slot]) if self.initialized[slot] else None
            indices, correction, end = replay_plan(
                start, b - a, int(rejected[row]), int(sampled[row]), old_end
            )
            if self.initialized[slot]:
                continuing.append(row)
            for i in indices:
                gather.append(a + i)
                packed_positions.append(positions[a + i])
            if correction:
                correction_offsets.append(len(gather))
                correction_rows.append(slot)
                gather.append(a)  # overwritten with the actual correction token
                packed_positions.append(end - 1)
            ends.append(end)
            packed_qsl.append(len(gather))
        self.rows = torch.as_tensor(ids_cpu, dtype=torch.long, device=o.device)
        self.ends = ends
        if continuing:
            cr = torch.as_tensor(continuing, dtype=torch.long, device=o.device)
            slots = self.rows[cr]
            for gid, cache, backup in self.layers:
                blocks = o.block_tables.input_block_tables[gid][:n, 0].long()[cr]
                cache.index_copy_(0, blocks, backup.index_select(0, slots))
        # Reorder only this packed draft forward, not target/scheduler buffers.
        # Native Mamba metadata expects one-token continuations before prefills.
        order = sorted(
            range(n),
            key=lambda i: (
                packed_qsl[i + 1] - packed_qsl[i] > 1 or positions[int(qsl[i])] == 0
            ),
        )
        old_corrections = dict(zip(correction_offsets, correction_rows))
        new_gather, new_positions, new_qsl = [], [], [0]
        correction_rows, correction_offsets = [], []
        last_indices = [0] * n
        for row in order:
            for old in range(packed_qsl[row], packed_qsl[row + 1]):
                if old in old_corrections:
                    correction_offsets.append(len(new_gather))
                    correction_rows.append(old_corrections[old])
                new_gather.append(gather[old])
                new_positions.append(packed_positions[old])
            last_indices[row] = len(new_gather) - 1
            new_qsl.append(len(new_gather))
        gather, packed_positions, packed_qsl = new_gather, new_positions, new_qsl
        self.order = torch.as_tensor(order, dtype=torch.long, device=o.device)
        self.ordered_seq_lens = torch.as_tensor(
            [ends[i] for i in order], dtype=torch.int32
        )
        self.prefilling = torch.as_tensor([positions[int(qsl[i])] == 0 for i in order])
        total = len(gather)
        indices = torch.as_tensor(gather, dtype=torch.long, device=o.device)
        o.expanded_input_ids[:total].copy_(batch.input_ids[indices])
        if correction_rows:
            dst = torch.as_tensor(correction_offsets, dtype=torch.long, device=o.device)
            src = torch.as_tensor(correction_rows, dtype=torch.long, device=o.device)
            o.expanded_input_ids[dst] = last_sampled.reshape(-1)[src].to(
                o.expanded_input_ids.dtype
            )
        o.expanded_positions[:total].copy_(
            torch.as_tensor(packed_positions, device=o.device)
        )
        o.input_buffers.query_start_loc[: n + 1].copy_(
            torch.as_tensor(packed_qsl, device=o.device)
        )
        o.input_buffers.query_start_loc[n + 1 :].fill_(total)
        o.input_buffers.seq_lens[:n].copy_(self.ordered_seq_lens.to(o.device))
        o.input_buffers.seq_lens[n:].zero_()
        o.last_token_indices[:n].copy_(torch.as_tensor(last_indices, device=o.device))
        o.draft_is_prefilling[:n].zero_()
        return total, np.asarray(packed_qsl, dtype=np.int32), self.ordered_seq_lens

    def metadata(self, qsl, slot_mappings):
        o = self.owner
        n = self.rows.numel()
        return build_attn_metadata(
            attn_groups=o.attn_groups,
            num_reqs=n,
            num_tokens=int(qsl[-1]),
            query_start_loc_gpu=o.input_buffers.query_start_loc[: n + 1],
            query_start_loc_cpu=torch.from_numpy(qsl),
            max_query_len=int(np.diff(qsl).max()),
            seq_lens=o.input_buffers.seq_lens[:n],
            max_seq_len=int(self.ordered_seq_lens.max()),
            block_tables=[
                x[:n].index_select(0, self.order)
                for x in o.block_tables.input_block_tables
            ],
            slot_mappings=slot_mappings,
            kv_cache_config=o.kv_cache_config,
            causal=True,
            seq_lens_cpu_upper_bound=self.ordered_seq_lens,
            is_prefilling=self.prefilling,
        )

    def commit(self):
        """Save state before speculative draft tokens advance the working cache."""
        o = self.owner
        n = self.rows.numel()
        for gid, cache, backup in self.layers:
            blocks = o.block_tables.input_block_tables[gid][:n, 0].long()
            backup.index_copy_(0, self.rows, cache.index_select(0, blocks))
        ids = self.rows.cpu().tolist()
        self.committed_end[ids] = self.ends
        self.initialized[ids] = True
