# Request-wide decision KL (research fork)

This fork adds `block_kl`, `token_kl`, `cactus`, and `fsd` as
`rejection_sample_method` values. The upstream `standard`, `block`, and `fly`
methods retain their original paths. These additions require numerical and
model evaluation before performance or output-divergence claims.

## Block-KL and Token-KL

The new methods tilt the exact verifier's final accepted-length distribution
by `exp(beta * length)` while retaining its correction distribution. Block-KL
uses the last successful independent block coin; Token-KL uses the first
rejected token. They charge actual conditional decision KL to a persistent
request ledger, in approximate-to-reference nats.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    revision="b968826d9c46dd6066d109eabc6255188de91218",
    generation_config="vllm",
    dtype="bfloat16",
    tensor_parallel_size=1,
    speculative_config={
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "num_speculative_tokens": 8,
        "draft_sample_method": "probabilistic",
        "enable_adaptive_verification": False,
        "rejection_sample_method": "block_kl",
        "decision_kl_policy": "progress",
    },
)
params = SamplingParams(
    n=1, temperature=0.8, top_p=1, top_k=0, min_p=0,
    max_tokens=2048, extra_args={"spec_kl_epsilon": 0.1},
)
```

The allocation policy is `uniform` (epsilon divided by the output limit per
round) or `progress` (remaining budget times expected chunk progress divided
by remaining output horizon). Both debit the conditional cost plus a numerical
guard. Zero budget selects the original verifier. A failed scalar solve selects
the original verifier before sampling. Lost request state disables approximation.

The initial implementation requires finite nonnegative epsilon, a finite output
limit, `n=1`, positive temperature, probabilistic drafting, no probability
truncation or penalties, and no structured-output constraint. Greedy targets
cannot change their output under a finite forward KL bound. The theoretical
bound concerns ideal arithmetic; floating-point reductions and sampling need
the numerical audit described in the research note.

## Existing approximate controls

For `cactus`, set `cactus_delta` to the published Taylor-rule parameter. The
candidate-conditioned surrogate and recovery follow
[Cactus](https://arxiv.org/abs/2604.04987); delta is not a request KL budget.

For `fsd`, set `fsd_threshold` and `fsd_statistic="js"` for the paper-defined
Jensen–Shannon statistic. The pilot-only `released_js` option evaluates the
released code's reverse-to-mixture expression. Accepted drafts are followed
by target recovery at the first failed gate, following
[FSD](https://arxiv.org/abs/2502.20704).

`abv_trace_dir` optionally writes a final request-ledger record. Tracing performs
a CPU transfer when a request is removed; exclude trace-export cost from any
isolated kernel microbenchmark and report its effect in end-to-end latency.
`abv_audit_dir` additionally snapshots early and final processed score rows and the
conditional length calculation. Use this only for separate numerical-audit
jobs, since copying vocabulary rows changes memory traffic and timing.
The study harness supplies `extra_args["abv_logical_id"]` to join traces across
engine restarts and suppresses warmup traces with `abv_skip_trace`. Finished
requests may export their ledger at a request-boundary control RPC; no round
requires a CPU transfer.
An unrestored request records epsilon as a conservative charge and sets
`prior_cost_unknown`, then continues with the exact reference verifier.
Fresh FP64 ledger rows are initialized directly on the GPU. This avoids the
temporary host-to-device row allocation during request admission; saved rows
still restore through the same GPU copy.

The broker study harness uses named worker-extension RPCs for request-boundary
diagnostics. Optional component GPU events run in separate diagnostic jobs;
their wall time is excluded from the main latency frontier. Internal engine
startup warmups use the reference verifier without a research request ledger.

See `research/abv` for the pre-run theory note, provenance, broker-only launch
controller, reproducible synthetic checks, and experiment gates.
