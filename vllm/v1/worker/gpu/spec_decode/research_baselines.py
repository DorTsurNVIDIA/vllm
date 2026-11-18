# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research baseline equations from Cactus and FSD; see research/abv/NOTICE."""

import torch

from vllm.triton_utils import tl, triton


def process_baseline(
    target,
    cache,
    sampled,
    mapping,
    local_pos,
    temperature,
    delta,
    fsd_threshold,
    statistic,
):
    if cache is None:
        raise ValueError("Research baselines require cached probabilistic draft logits")
    bonus = torch.cat(
        (local_pos[1:] == 0, torch.ones(1, device=target.device, dtype=torch.bool))
    )
    raw_token = torch.roll(sampled, -1)
    token = raw_token.clamp_min(0).long()
    rows = mapping if cache.row_mapping is None else cache.row_mapping[mapping]
    step = local_pos.clamp_max(cache.values.shape[1] - 1).long()
    q_logits = cache.values[rows.long(), step].float()
    q_logits = q_logits / temperature[mapping.long(), None]
    lp = torch.log_softmax(target.float(), -1)
    lq = torch.log_softmax(q_logits, -1)
    p, q = lp.exp(), lq.exp()
    if delta > 0:
        pn = p.gather(1, token[:, None]).squeeze(1)
        hn = (pn + (2 * delta * pn * (1 - pn)).sqrt()).clamp_max(1)
        h = p * ((1 - hn) / (1 - pn)).nan_to_num(nan=1)[:, None]
        h.scatter_(1, token[:, None], hn[:, None])
        result = torch.where(bonus[:, None], target.float(), h.log())
        return result, None
    lm = torch.logaddexp(lp, lq) - 0.6931471805599453
    if statistic == "js":
        d = 0.5 * (
            torch.where(p > 0, p * (lp - lm), 0) + torch.where(q > 0, q * (lq - lm), 0)
        ).sum(-1)
    else:
        m = lm.exp()
        d = 0.5 * torch.where(m > 0, m * (2 * lm - lp - lq), 0).sum(-1)
    return target, (d < fsd_threshold) & ~bonus & (raw_token >= 0)


@triton.jit
def _gate_lengths(output, gates, cu):
    r = tl.program_id(0)
    start, end = tl.load(cu + r), tl.load(cu + r + 1)
    length = 0
    accepting = True
    for i in range(end - start - 1):
        accepting &= tl.load(gates + start + i)
        length += accepting.to(tl.int32)
    tl.store(output + r, length)


def gate_lengths(gates, cu):
    out = torch.empty(cu.shape[0] - 1, dtype=torch.int32, device=cu.device)
    _gate_lengths[(out.numel(),)](out, gates, cu)
    return out
