# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only decision-KL sampler. Ideal guarantee is documented separately."""

import gzip
import hashlib
import json
from pathlib import Path
from typing import TypedDict

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import spec_step_seed, tl_rand32
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logprobs_and_logsumexp,
    _compute_global_residual_mass,
)


class TraceMetadata(TypedDict):
    logical_id: str
    skip: bool
    lost_restore: bool


class DecisionKLState:
    """GPU ledger, preserved by request ID through worker slot preemption."""

    def __init__(self, max_reqs, device, trace_dir=None, audit_dir=None, k=1):
        # epsilon, remaining, horizon, spent, rounds, fallbacks, last L, last KL
        self.values = torch.zeros((max_reqs, 8), dtype=torch.float64, device=device)
        self.saved = {}
        self.owners = {}
        self.metadata: dict[str, TraceMetadata] = {}
        self.exported: set[str] = set()
        self.trace_dir = trace_dir
        self.audit_dir = audit_dir
        self.audit_width = k + 1
        self.audit = (
            torch.zeros((max_reqs, 2 * (k + 1) + 8), dtype=torch.float64, device=device)
            if audit_dir
            else None
        )
        self.score_snapshots = {}
        self.audit_history = {}
        self.capture_counts = {}

    def capture(self, slots, **tensors):
        if self.audit_dir:
            assert self.audit is not None
            snapshot = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in tensors.items()
            }
            for slot in slots:
                slot = int(slot)
                capture_index = self.capture_counts.get(slot, 0)
                entry = dict(
                    snapshot,
                    audit_version=2,
                    slot=slot,
                    width=self.audit_width,
                    decision=self.audit[slot].clone(),
                    ledger=self.values[slot].clone(),
                    capture_index=capture_index,
                )
                if capture_index < 8:
                    self.audit_history.setdefault(slot, []).append(entry)
                self.score_snapshots[slot] = entry
                self.capture_counts[slot] = capture_index + 1

    def add(self, req_id, slot, params, resumed=False):
        params._validate_request_kl()
        epsilon = (params.extra_args or {}).get("spec_kl_epsilon", 0.0)
        self.owners[req_id] = slot
        previous_metadata = self.metadata.get(req_id)
        lost_restore = (resumed and req_id not in self.saved) or (
            previous_metadata is not None and previous_metadata["lost_restore"]
        )
        self.metadata[req_id] = {
            "logical_id": str((params.extra_args or {}).get("abv_logical_id", req_id)),
            "skip": bool((params.extra_args or {}).get("abv_skip_trace", False)),
            "lost_restore": bool(lost_restore),
        }
        if self.audit is not None:
            self.audit[slot].zero_()
            self.audit_history[slot] = []
            self.capture_counts[slot] = 0
        if req_id in self.saved:
            self.values[slot].copy_(self.saved.pop(req_id))
        else:
            # Initialize the persistent FP64 row without a temporary H2D tensor.
            row = self.values[slot]
            row.zero_()
            row[0].fill_(epsilon)
            row[1].fill_(0.0 if resumed else epsilon)
            row[2].fill_(params.max_tokens)
            row[3].fill_(epsilon if resumed else 0.0)
            row[5].fill_(int(resumed))
            row[6].fill_(-1.0)

    def export(self, req_id):
        """Export once after completion; allowed at a request-boundary control RPC."""
        slot = self.owners.get(req_id)
        if slot is None or self.metadata[req_id]["skip"]:
            return None
        logical_id = self.metadata[req_id]["logical_id"]
        name = hashlib.sha256(logical_id.encode()).hexdigest()
        path = Path(self.trace_dir) / (name + ".json") if self.trace_dir else None
        if req_id in self.exported:
            return str(path) if path else None
        if self.audit_dir:
            assert self.audit is not None
            snapshots = list(self.audit_history.get(slot, []))
            last = self.score_snapshots.get(slot)
            if last is not None and not any(last is item for item in snapshots):
                snapshots.append(last)
            for snapshot in snapshots:
                snapshot = {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in snapshot.items()
                }
                snapshot.update(
                    request_id=req_id,
                    logical_request_id=logical_id,
                )
                destination = Path(self.audit_dir)
                destination.mkdir(parents=True, exist_ok=True)
                with gzip.open(
                    destination / (name + f"-r{snapshot['capture_index']:05d}.pt.gz"),
                    "wb",
                    compresslevel=3,
                ) as stream:
                    torch.save(snapshot, stream)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            values = self.values[slot].cpu().tolist()
            path.write_text(
                json.dumps(
                    {
                        "request_id": req_id,
                        "logical_request_id": logical_id,
                        "prior_cost_unknown": self.metadata[req_id]["lost_restore"],
                        **dict(
                            zip(
                                [
                                    "epsilon",
                                    "remaining",
                                    "horizon",
                                    "charged_kl",
                                    "rounds",
                                    "fallbacks",
                                    "last_length",
                                    "last_conditional_kl",
                                ],
                                values,
                            )
                        ),
                    }
                )
                + "\n"
            )
        self.exported.add(req_id)
        return str(path) if path else None

    def export_logical(self, logical_id):
        for req_id, metadata in self.metadata.items():
            if metadata["logical_id"] == logical_id:
                return self.export(req_id)
        if self.trace_dir:
            path = Path(self.trace_dir) / (
                hashlib.sha256(logical_id.encode()).hexdigest() + ".json"
            )
            if path.exists():
                return str(path)
        return None

    def remove(self, req_id, finished):
        slot = self.owners.get(req_id)
        if slot is not None:
            if finished:
                self.export(req_id)
            else:
                self.saved[req_id] = self.values[slot].clone()
            self.score_snapshots.pop(slot, None)
            self.audit_history.pop(slot, None)
            self.capture_counts.pop(slot, None)
            self.owners.pop(req_id, None)
        if finished:
            self.saved.pop(req_id, None)
            self.metadata.pop(req_id, None)
            self.exported.discard(req_id)


@triton.jit
def _tilted(log_pi, lengths, beta):
    weights = log_pi + beta * lengths
    peak = tl.max(weights, 0)
    unnormalized = tldevice.exp(weights - peak)
    pi = unnormalized / tl.sum(unnormalized, 0)
    log_z = peak + tldevice.log(tl.sum(unnormalized, 0))
    delta = beta * lengths - log_z
    cost = tl.sum(tl.where(pi > 0, pi * delta, 0.0), 0)
    return pi, tl.maximum(cost, 0.0)


@triton.jit
def _select_length(
    forced,
    ledger,
    audit,
    prompt_lens,
    target,
    target_stride,
    draft,
    ds0,
    ds1,
    row_mapping,
    target_max,
    tms,
    target_sum,
    tss,
    draft_max,
    dms,
    draft_sum,
    dss,
    sampled,
    cu,
    mapping,
    temps,
    seeds,
    positions,
    cumulative,
    residual,
    rs,
    vocab_blocks,
    PAD_V: tl.constexpr,
    PAD_K: tl.constexpr,
    BLOCK: tl.constexpr,
    PROGRESS: tl.constexpr,
    AUDIT_WIDTH: tl.constexpr,
):
    r = tl.program_id(0)
    slot = tl.load(mapping + r).to(tl.int64)
    start = tl.load(cu + r).to(tl.int64)
    end = tl.load(cu + r + 1).to(tl.int64)
    k = end - start - 1
    candidate_positions = tl.arange(0, PAD_K)
    candidate_tokens = tl.load(
        sampled + start + candidate_positions + 1,
        mask=candidate_positions < k,
        other=0,
    )
    k = tl.min(
        tl.where(
            (candidate_positions < k) & (candidate_tokens < 0),
            candidate_positions,
            k,
        ),
        0,
    )
    budget = tl.load(ledger + slot * 8 + 1)
    tl.store(forced + r, -1)
    if audit is not None:
        tl.store(audit + slot * (2 * AUDIT_WIDTH + 8) + 2 * AUDIT_WIDTH + 4, 0.0)
    if k > 0 and budget > 0:
        temp = tl.load(temps + slot)
        indices = tl.arange(0, PAD_K)
        log_pi = tl.full((PAD_K,), float("-inf"), tl.float64)
        survival = tl.full((), 0.0, tl.float64)
        for offset in range(k + 1):
            i = k - offset if BLOCK else offset
            if BLOCK:
                log_a = tl.full((), 0.0, tl.float64)
                if i == k:
                    log_a = tl.load(cumulative + start + i - 1).to(tl.float64)
                elif i > 0:
                    s = tldevice.exp(tl.load(cumulative + start + i - 1).to(tl.float64))
                    token = tl.load(sampled + start + i + 1)
                    z = _compute_global_residual_mass(
                        residual,
                        rs,
                        s,
                        target,
                        target_stride,
                        target_max,
                        tms,
                        target_sum,
                        tss,
                        token,
                        start + i,
                        vocab_blocks,
                        PAD_V,
                        True,
                    ).to(tl.float64)
                    a = tl.where(z + 1 - s > 0, z / (z + 1 - s), 1.0)
                    log_a = tldevice.log(a)
                mass = survival + log_a
                survival += tldevice.log1p(-tldevice.exp(log_a))
            else:
                log_a = tl.full((), float("-inf"), tl.float64)
                if i < k:
                    token = tl.load(sampled + start + i + 1)
                    lp, lq, _target_lse, _draft_lse = (
                        _compute_global_logprobs_and_logsumexp(
                            token,
                            True,
                            start + i,
                            slot,
                            i,
                            temp,
                            target,
                            target_stride,
                            target_max,
                            tms,
                            target_sum,
                            tss,
                            draft,
                            ds0,
                            ds1,
                            row_mapping,
                            draft_max,
                            dms,
                            draft_sum,
                            dss,
                            vocab_blocks,
                            PAD_V,
                            True,
                        )
                    )
                    log_a = tl.minimum((lp - lq).to(tl.float64), 0.0)
                mass = survival + tldevice.log1p(-tldevice.exp(log_a))
                survival += log_a
            log_pi = tl.where(indices == i, mass, log_pi)
        peak = tl.max(log_pi, 0)
        log_pi -= peak + tldevice.log(tl.sum(tldevice.exp(log_pi - peak), 0))
        base = tldevice.exp(log_pi)
        mean = tl.sum(base * indices, 0)
        horizon = tl.load(ledger + slot * 8 + 2)
        epsilon = tl.load(ledger + slot * 8)
        t = tl.load(positions + start) + 1 - tl.load(prompt_lens + slot)
        allowance = tl.minimum(budget, epsilon / horizon)
        if PROGRESS:
            allowance = tl.minimum(
                budget, budget * (1 + mean) / tl.maximum(horizon - t, 1)
            )
        # Engineering margin for upstream FP32 score reductions, calibrated
        # against extended-precision real-row audits. This is not a certified
        # floating-point error bound. At unresolved budgets use the reference.
        guard = 1e-10 + 1e-3 * tldevice.sqrt(allowance) + 1e-6 * allowance
        usable = tl.maximum(0.0, allowance - guard)
        lengths = indices.to(tl.float64) - k
        maximum = tl.max(tl.where(log_pi > float("-inf"), indices, -1), 0)
        limit = -tl.sum(tl.where(indices == maximum, log_pi, 0.0), 0)
        lo = tl.full((), 0.0, tl.float64)
        pi = base
        cost = tl.full((), 0.0, tl.float64)
        bracketed = tl.full((), True, tl.int1)
        if usable >= limit and limit > 0:
            pi = (indices == maximum).to(tl.float64)
            cost = limit
            lo = tl.full((), float("inf"), tl.float64)
        elif usable > 0 and limit > 0:
            hi = tl.full((), 1.0, tl.float64)
            _hi_pi, hi_cost = _tilted(log_pi, lengths, hi)
            for _bracket_step in range(32):
                if hi_cost < usable:
                    hi *= 2
                    _hi_pi, hi_cost = _tilted(log_pi, lengths, hi)
            bracketed = (hi_cost >= usable) | (usable >= limit)
            for _bisect_step in range(64):
                mid = (lo + hi) / 2
                _mid_pi, cost = _tilted(log_pi, lengths, mid)
                feasible = (cost <= usable) & (cost + guard <= allowance)
                lo = tl.where(feasible, mid, lo)
                hi = tl.where(feasible, hi, mid)
            pi, cost = _tilted(log_pi, lengths, lo)
        charged = tl.where(cost > 0, cost + guard, 0.0)
        valid = (
            bracketed
            & (cost == cost)
            & (cost <= usable)
            & (charged <= allowance)
            & (charged <= budget)
            & (allowance > guard)
            & (cost > 0)
            & (tl.abs(tl.sum(pi, 0) - 1.0) < 1e-10)
        )
        if audit is not None:
            offset = audit + slot * (2 * AUDIT_WIDTH + 8)
            tl.store(offset + indices, log_pi, mask=indices < AUDIT_WIDTH)
            tl.store(offset + AUDIT_WIDTH + indices, pi, mask=indices < AUDIT_WIDTH)
            tl.store(offset + 2 * AUDIT_WIDTH, lo)
            tl.store(offset + 2 * AUDIT_WIDTH + 1, cost)
            tl.store(offset + 2 * AUDIT_WIDTH + 2, usable)
            tl.store(offset + 2 * AUDIT_WIDTH + 3, k)
            tl.store(offset + 2 * AUDIT_WIDTH + 4, valid.to(tl.float64))
            tl.store(offset + 2 * AUDIT_WIDTH + 5, guard)
            tl.store(offset + 2 * AUDIT_WIDTH + 6, charged)
            tl.store(offset + 2 * AUDIT_WIDTH + 7, allowance)
        if valid:
            seed = spec_step_seed(tl.load(seeds + slot), 20261118)
            u = tl_rand32(seed, tl.load(positions + start), includes_zero=False)
            length = tl.sum((tl.cumsum(pi, 0) < u).to(tl.int32), 0)
            length = tl.minimum(length, k)
            tl.store(forced + r, length)
            tl.store(ledger + slot * 8 + 1, budget - charged)
            tl.store(ledger + slot * 8 + 3, epsilon - (budget - charged))
            tl.store(ledger + slot * 8 + 6, length)
            tl.store(ledger + slot * 8 + 7, cost)
        elif limit != 0:
            tl.store(ledger + slot * 8 + 5, tl.load(ledger + slot * 8 + 5) + 1)
        tl.store(ledger + slot * 8 + 4, tl.load(ledger + slot * 8 + 4) + 1)


def select_lengths(
    ledger,
    prompt_lens,
    target,
    cache,
    sampled,
    cu,
    mapping,
    temps,
    seeds,
    positions,
    cumulative,
    residual,
    target_max,
    target_sum,
    draft_max,
    draft_sum,
    vocab_blocks,
    num_steps,
    block,
    progress,
    audit=None,
    audit_width=0,
):
    forced = torch.empty(cu.shape[0] - 1, dtype=torch.int32, device=target.device)
    _select_length[(forced.numel(),)](
        forced,
        ledger,
        audit,
        prompt_lens,
        target,
        target.stride(0),
        cache.values,
        cache.values.stride(0),
        cache.values.stride(1),
        cache.row_mapping,
        target_max,
        target_max.stride(0),
        target_sum,
        target_sum.stride(0),
        draft_max,
        draft_max.stride(0),
        draft_sum,
        draft_sum.stride(0),
        sampled,
        cu,
        mapping,
        temps,
        seeds,
        positions,
        cumulative,
        residual,
        residual.stride(0) if residual is not None else 0,
        vocab_blocks,
        PAD_V=triton.next_power_of_2(vocab_blocks),
        PAD_K=triton.next_power_of_2(num_steps + 1),
        BLOCK=block,
        PROGRESS=progress,
        AUDIT_WIDTH=audit_width,
        num_warps=1,
    )
    return forced
