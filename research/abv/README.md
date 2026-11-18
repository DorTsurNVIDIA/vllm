# Request-wide decision KL research fork

This experimental implementation is based on vLLM v0.34.0 commit
`62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`.

It adds Block-KL and Token-KL accepted-length tilting, request-identity budget
state, Cactus's published Taylor rule, and FSD's paper-defined JS control.
The upstream standard, block, and FLy paths are retained.

GPU kernel checks, numerical audits on real score rows, lifecycle validation,
and model experiments must pass before scientific use. Static checks and a
proof in ideal arithmetic do not certify floating-point output divergence or
establish a speedup. No publication novelty is claimed.

See [the feature documentation](../../docs/features/speculative_decoding/decision_kl.md)
and `NOTICE` for API details and attribution. The broker-only study controller,
pre-run LaTeX research note, synthetic checks, and experiment artifacts are
provided as separate study deliverables.
