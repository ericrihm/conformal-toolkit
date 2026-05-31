# Math-Tooling Gaps & Verification Roadmap

This audit was as much a stress-test of *our verification tooling* as of the
library. This document records where the tooling fell short, and the concrete
plan to close each gap so that the checks in [`../ERRATA.md`](../ERRATA.md) run
automatically — not by hand — next time.

The goal is a **verification harness that fails loudly**: a dead backend or an
unverifiable claim should break CI, never silently degrade a proof to "trust
me."

---

## Root cause: silent backend degradation

Every symbolic backend was unavailable during the audit:

| Tool | Failure | Consequence |
|------|---------|-------------|
| `math_verify_identity`, `math_simplify` | `Neither Lean 4 nor SymPy available` | No symbolic identity checks |
| `math_compute` | `sympy unavailable` | Numeric checks fell back to ad-hoc `python3` |
| `wolfram_query` | `no_api_key` | No exact closed-form / factorial verification |

All scalar checks were re-run in plain Python; all tensor identities were done
by hand. **It still worked** — but only because the reviewers happened to know
the closed forms. A less-expert pass would have produced false "verified"
stamps. That is the real risk.

**P0 fix — make the stack self-healthchecking.** Bundle a pinned `sympy` (and
optional Lean 4 + mathlib) into the Dynamo MCP server image; have
`math_stack_status` return non-zero when a backend is down, and gate any
verification workflow on it so a dead backend *fails the job* instead of
silently downgrading it. Provision `WOLFRAM_ALPHA_APPID` from Keychain
(`cobalt-vault/wolfram-appid`, per house keychain policy) and surface
`wolfram_quota_status`.

---

## Gap 1 — No symbolic tensor / abstract-index calculus verifier *(P0 for this domain)*

The single biggest gap. There is no SageManifolds/`xAct`/`cadabra2`-class engine
behind the math tools, so **every** tensor claim had to be hand-reduced to a
scalar on `Sⁿ`: the Bach contraction order, `|P|²`, `Δ` on a `(0,2)` tensor, the
Thomas-D slots, tractor-connection parallelism, "middle curvature block = Weyl."

**Proposal — `math_tensor_verify`.** Wrap SageManifolds (or `cadabra2`).
Inputs: a metric/connection plus an abstract-index expression and an identity to
check. Output: the simplified residual tensor and a per-component pass/fail.
Would have caught `M2` (missing `R`), the `M3`/`M8`/`M12` signs, and the
`C1`/`M11` trace coefficients **mechanically**.

## Gap 2 — No conformal-weight / covariance checker *(P1, high leverage)*

Several errors are pure **density-weight bookkeeping**: `M4` (linear-`H` term
and bidegree-sign), `M9` (`£_v h` homogeneity), `m4` (μ-slot weight). These need
no heavy CAS.

**Proposal — `conformal_weight_check`.** Given an operator and its slot weights,
verify homogeneity, bidegree *direction* (`target − source` sign), and conformal
covariance under `g → e^{2ω}g`. Cheap, fast, and it mechanizes exactly the
reasoning that caught `M4` by eye.

## Gap 3 — No degenerate-metric / Carrollian Lie-derivative engine *(P2)*

`£_ξ` of a *degenerate* `(0,2)` tensor and the conformal-Carroll pair condition
`(£_ξ h = 2λh, £_ξ v = −λv)` had no symbolic support.

**Proposal — extend Gap 1's verifier with degenerate-metric support** plus a
`carroll_symmetry_check(ξ, h, v)` predicate. Directly validates `M9`/`M10`/`m5`.

## Gap 4 — No conformal-invariant basis enumerator *(P2)*

No Weyl-invariant-theory / Hironaka-decomposition tool, so the weight-4
hypersurface count (`M5`) could only be bounded below (`> 4`), not pinned.

**Proposal — `conformal_invariant_basis(weight, ambient_dim, hypersurface=bool)`**
returning an independent generating set (algebraic + derivative invariants).
Hard, but it is what makes `count_invariants` checkable exactly.

## Gap 5 — No library-execution harness in CI *(P1, highest ROI after backends)*

We could not run the repo's own Sage code, so prose-vs-return mismatches went
unverified (does `q_curvature(order=4)` literally return 6? does
`count_invariants` return the stated ints?).

**Proposal — a Sage-enabled CI job** that evaluates each operator on canonical
anchors — round `Sⁿ`, `S²×S²`, **and a non-conformally-flat metric** — and
asserts known values (Branson `Q_n=(n−1)!`, conformal invariance / vanishing).
This catches `M2`, `M8`, `C1`/`C2`, `M16` **empirically, with no symbolic engine
at all.** The anchor library is already specified in `ERRATA.md → How we caught
them`; turning it into `tests/test_anchors/` is the concrete first deliverable.

## Gap 6 — Literature retrieval isn't citation-grade *(P3)*

`world_arxiv` returned irrelevant collaboration papers; `WebFetch` couldn't pull
numbered equations from arXiv PDFs (binary / ar5iv paraphrase only). This left
`M5`/`M7` (Fialkow, GW counts) and the Graham–Hirachi constant (`m3`) at
moderate confidence.

**Proposal — an arXiv *source* fetcher** that pulls the LaTeX source (not the
rendered PDF) and indexes numbered equations for verbatim citation lookup.

---

## Priority summary

| Gap | Priority | Effort | Catches |
|-----|----------|--------|---------|
| Backend healthcheck + Wolfram key | **P0** | Low | the silent-degradation risk itself |
| Tensor / abstract-index verifier | **P0** | High | `M2`, `M3`, `M8`, `M12`, `C1`, `M11` |
| Library-execution CI harness | **P1** | Low–Med | `M2`, `M8`, `C1`, `C2`, `M16` (empirically) |
| Conformal-weight checker | **P1** | Low | `M4`, `M9`, `m4` |
| Degenerate-metric / Carroll engine | **P2** | Med | `M9`, `M10`, `m5` |
| Invariant-basis enumerator | **P2** | High | `M5`, `M6` |
| arXiv source fetcher | **P3** | Low | `M5`, `M7`, `m3` |

**Do first:** the backend healthcheck (P0) and the library-execution CI harness
(P1). Together they are low-effort and would have caught the two *critical*
errors and most majors automatically — the highest return per hour of the whole
list.
