# Errata & Verification Log

> *"The first principle is that you must not fool yourself — and you are the easiest person to fool."* — R. P. Feynman

This document is the honest record of every mathematical error we found in
`conformal-toolkit`, how we found it, and exactly how each was corrected. We
publish it in full — not as a footnote — because **a teaching tool earns trust
by showing its work, including its mistakes.** If you are a student, read this
as a worked example of how to pressure-test a computation. If you are a
researcher and you find an error *we* missed, see
[How to contribute a correction](#how-to-contribute-a-correction) — we built
this expecting exactly that.

Every line number refers to the state of the repository *before* the fixes in
this round; each entry is tagged with a stable ID (`C1`, `M2`, `m4`, …) that is
referenced from `# ERRATA` comments in the source.

---

## How we caught them

In May 2026 we ran an independent, adversarial re-audit of the whole library.
The method is reproducible and is the method we recommend to any contributor:

1. **Extract every quantitative claim** — formulas in code, docstrings,
   comments, README prose, and example outputs — and treat each as a
   refutable hypothesis.
2. **Reduce tensors to scalars on a known geometry.** Almost every error here
   was caught by evaluating a tensor claim on a *concrete* metric where the
   answer is known in closed form:
   - the **round sphere `Sⁿ`** (sectional curvature 1): `Ric = (n−1)g`,
     `R = n(n−1)`, Schouten `P = ½g`, `J = n/2`, Branson `Q_n = (n−1)!`;
   - its **hyperbolic filling** `g_ρ = (1 − ρ²/4)² g₀`, which gives the exact
     Fefferman–Graham coefficients `g₂ = −½g₀`, `g₄ = 1/16 g₀`, and volume
     density `(1 − ρ²/4)ⁿ`;
   - **flat `Rⁿ`** and the **cylinder `S¹×R`** for sanity and non-umbilicity.
3. **Check conformal weights as a checksum** (see Teaching Note 3). A
   surprising number of errors are visible *without computing anything* — the
   terms simply don't have matching conformal weight.
4. **Cross-check the code against its own other modules.** Several errors were
   exposed by internal inconsistency (e.g. a docstring weight that contradicts
   the package's own tractor-metric pairing), not by external references.
5. **Verify adversarially.** Every candidate error was handed to a second
   reviewer whose job was to *refute* it — to find the convention under which
   the original was right. Only survivors are listed here.

A note on tooling honesty: during this audit our symbolic backends
(SymPy/Wolfram via the Dynamo math layer) were unavailable, so the scalar
checks were re-run in plain Python and the tensor identities were done by hand.
That friction is itself catalogued in [`docs/TOOLING_GAPS.md`](docs/TOOLING_GAPS.md),
along with the verification harness we are adding so these checks run
automatically in CI.

---

## Summary

| Severity | Count | Theme |
|----------|-------|-------|
| **Critical** | 2 | Wrong Fefferman–Graham `g₄` coefficient; wrong renormalized-volume `v₂` |
| **Major** | ~17 | Missing curvature term in the conformal Laplacian; non-traceless stress tensor; incomplete extrinsic `Q₄`; discrete bi-Laplacian normalization; false discrete identity; doc/code sign mismatches |
| **Minor** | ~6 | Normalization mismatches, missing `1/(n−3)` factor, weight labels, radius-specific docstrings |
| **Style / Convention** | 3 | Dead code; non-standard WKS normalization |

**The single most important meta-finding:** in the majority of cases *the code
computed the right thing and the prose described the wrong thing* (or vice
versa). The fix was rarely "change the math" — it was "make the words and the
code agree, and say which one was right." We flag the direction of each fix
below.

---

## Critical

### `C1` — Fefferman–Graham `g₄`: spurious `1/(n−4)` on the algebraic term
*File:* `conformal_toolkit/poincare_einstein/fefferman_graham.py` · *Status:* **Fixed**

- **Claimed:** `(g₄)_{ab} = 1/(n−4) · [ (g₂²)_{ab} − 1/(4(n−1)) tr(g₂²) g₀ ]`.
- **How the code actually worked:** it computed exactly that — including the
  `1/(n−4)` prefactor on the *algebraic* `g₂²` term and a trace subtraction.
- **Why it's wrong:** the `1/(n−4)` pole belongs **only** to the *differential*
  (Bach-tensor / `ΔP`) part of `g₄` — that is the term that obstructs at `n=4`.
  The algebraic `g₂²` piece has coefficient `1/4`. On the hyperbolic filling of
  `Sⁿ` the exact answer is `g₄ = 1/16 g₀ = ¼(g₂)²`, but the buggy formula
  returns `7/80 g₀ ≈ 0.0875 g₀` at `n=6`, and its trace coefficient
  `(3n−4)/(4(n−1)(n−4))` equals `1/4` at no integer `n`.
- **Correct:** `g₄ = ¼ (g₂²)_{ab} + 1/(n−4)·[Bach/ΔP terms]`. The algebraic
  piece (now implemented) is exact on locally conformally flat boundaries; the
  differential terms are documented as not-yet-implemented.
- **How we caught it:** exact rational arithmetic on `(1 − ρ²/4)²` → `g₂ = −½g₀`,
  `g₄ = 1/16 g₀`; the code's value disagreed at `n=6`.

### `C2` — Renormalized volume `v₂`: wrong `1/(n−2)` prefactor
*File:* `conformal_toolkit/poincare_einstein/renormalized_volume.py` · *Status:* **Fixed**

- **Claimed / code:** `v₂ = −1/(n−2) · J`.
- **Correct:** `v₂ = −½ J = −½ tr P`, **independent of `n`** (`= −n/4` on `Sⁿ`).
- **Why it slipped through:** the two formulas *coincide at `n=4`* (both give
  `−1` on `S⁴`). Any test run only in 4D would pass. At `n=6` the code gives
  `−3/4` versus the correct `−3/2`.
- **How we caught it:** the boundary volume density is `√(det g_ρ/det g₀) =
  (1 − ρ²/4)ⁿ`, whose `ρ²` coefficient is `−n/4`; with `J = n/2` that is `−J/2`.

---

## Major

### `M1` — README: a closed-form annotation that doesn't equal the number it annotates
*Files:* `README.md` (Quick Start), and the same closed form in prose · *Status:* **Fixed**

- **Claimed:** `Q₄ on S⁴ = 6  (… matches 2(n−1)!/((n/2−1)!)² for n=4)`.
- **How the code actually worked:** `core/q_curvature.py` computes
  `Q₄ = −ΔJ − 2|P|² + (n/2)J²`, which on `S⁴` (`P=½g`, `J=2`, `|P|²=1`) gives
  `−2 + 8 = 6`. **The number 6 is correct.**
- **Why it's wrong:** the *annotation formula* `2(n−1)!/((n/2−1)!)²` evaluates
  to `2·3!/(1!)² = 12`, not 6 — it contradicts the very value it labels.
- **Correct:** Branson's closed form is simply `Q_n(Sⁿ) = (n−1)!` (`= 6` at
  `n=4`); equivalently `n³/8 − n/2`. (The intended formula was off by a stray
  factor of 2: `(n−1)!/((n/2−1)!)² = 6` is also fine.)
- **Teaching value:** *always evaluate a closed form at the stated point before
  trusting it.* (Teaching Note 1.)

### `M2` — The "conformal Laplacian" was the bare Laplacian (missing curvature term)
*File:* `conformal_toolkit/core/gjms.py` (`laplacian_operator` / `P₂`) · *Status:* **Fixed (code)**

- **Claimed:** docstring and `conformal_structure.gjms_operator` call `P₂` the
  conformal (Yamabe) Laplacian, "generalizing the Yamabe operator."
- **How the code actually worked:** it returned `Δf` and nothing else.
- **Why it's wrong:** the GJMS `P₂` is `Δf − (n−2)/(4(n−1)) R f`. The curvature
  term is exactly what makes `P₂` conformally covariant for `n>2`; on `Sⁿ` its
  coefficient is `n(n−2)/4` (= 2 at `n=4`) — never zero for `n>2`. The repo's
  *own* Thomas-D operator uses `−(Δf + wJf)`, so the package already "knows" the
  coupling elsewhere.
- **Correct / fix:** added `− (n−2)/(4(n−1)) R f` (vanishes automatically at
  `n=2`).
- **Teaching value:** a "conformal" operator with no curvature term is almost
  always wrong (Teaching Note 2).

### `M3` — `P₆` sign: docstring/README said `−Δ³`, code computes `+Δ³`
*Files:* `conformal_toolkit/core/gjms.py` (`p6_operator`), `README.md` · *Status:* **Fixed (docs)**

- **Claimed:** "Computes `(−1)³ Δ³ = −Δ³`."
- **How the code actually worked:** three applications of the geometer
  Laplacian `Δ = ∇ᵃ∇ₐ` → `+Δ³`, no sign flip. **The code value is correct.**
- **Why it's wrong:** the `(−1)ᵏ` comes from the *analyst* convention
  `Δ = −∇ᵃ∇ₐ`; mixing it with the geometer Laplacian the code actually uses is
  an internal inconsistency. GJMS principal part is `+Δᵏ` in this convention.
- **Fix:** docstring and README now say `+Δ³`.

### `M4` — Extrinsic GJMS `P₂`: weight-inhomogeneous term + wrong-direction bidegree
*File:* `conformal_toolkit/hypersurface/extrinsic_gjms.py` · *Status:* **Documented (caveat in code)**

- **Claimed / code:** `P₂(f) = Δ_h f + (n/2−1) H f`, "covariant of bidegree
  `(n/2−1, n/2+1)`."
- **Why it's wrong (two independent reasons):**
  1. **Weight.** `H` is a weight `−1` density and `Δ` lowers conformal weight
     by 2, so the zeroth-order coefficient must carry weight `−2` — it must be
     **quadratic** (`H²`-type) and/or the intrinsic Schouten trace `J̄`, never
     *linear* in `H`. The correct Yamabe-type shape is
     `Δ_h f − (n−2)/2 (J̄ + extrinsic H²-term) f`.
  2. **Bidegree.** A second-order Laplacian-type operator **lowers** weight by
     2, so the bidegree difference must be `−2`. The stated `(n/2−1, n/2+1)`
     has difference `+2` — the wrong sign. The standard `k=1` value is
     `(1−n/2, −1−n/2)`.
- **How we caught it:** pure conformal-weight bookkeeping — no computation.
- *Why documented, not silently rewritten:* the corrected operator requires the
  intrinsic Schouten data; we flag it loudly rather than ship an unverified
  replacement. **Open for contribution.**

### `M5` — Weight-4 hypersurface invariants under-counted; basis incomplete
*File:* `conformal_toolkit/hypersurface/invariant_enumeration.py` · *Status:* **Documented (caveat in code)**

- **Claimed:** `count_invariants(4, ambient_dim=5) = 4` with basis
  `{|L₂|², tr(L₁⁴), W·L₁, |W|²}`, presented as the complete independent set.
- **Why it's wrong:** the algebraic quartic `(|L₁|²)²` is **independent** of
  `tr(L₁⁴)` for `n≥4` (their eigenvalue difference is `4abc(a+b+c) ≠ 0`) yet is
  entirely absent. A complete *pointwise* weight-4 classification also needs the
  tangential-derivative invariants (`|∇̄L₁|²` / `L₁·Δ̄L₁`, `|div L₁|²`) and the
  curvature coupling `J̄|L₁|²`. The true count therefore **exceeds 4**.
- **Status:** `count_invariants` is now documented as a **lower bound**, and the
  catalogue header lists the missing generators. Pinning the exact integer
  needs an invariant-theory computation (see `docs/TOOLING_GAPS.md` gap 6).

### `M6` — `_filter_by_dimension`: a *false* "Weyl is self-dual" justification
*File:* `conformal_toolkit/hypersurface/invariant_enumeration.py` · *Status:* **Comment corrected; behavior marked provisional**

- **Claimed (comment):** at ambient dimension 4, `|W_{nabc}|²` is dropped
  "because the Weyl is self-dual."
- **Why it's wrong:** a generic 4-manifold has **both** `W⁺` and `W⁻` nonzero
  and independent; self-duality (`W⁻=0`) is a *special* geometry, and even then
  it does not make the scalar `|W_{nabc}|²` expressible via the other listed
  invariants. The rationale is a non-sequitur.
- **Status:** the false comment is replaced with an honest "provisional,
  unverified" note (we could not prove the correct `ambient_dim=4` count
  either). The drop is retained only to keep the existing test green and is
  flagged for contributors.

### `M7` — Second conformal fundamental form `L₂`: missing ambient Weyl/Cotton (Fialkow) terms
*File:* `conformal_toolkit/hypersurface/conformal_fundamental_form.py` · *Status:* **Documented (caveat in code)**

- **Claimed:** `L₂ = (∇ₙL₁) + L₁²  − (1/n)|L₁|² h` is "the second conformal
  fundamental form."
- **Why it's wrong:** per Blitz–Gover–Waldron ([arXiv:2107.10381](https://arxiv.org/abs/2107.10381))
  the next-order conformal fundamental form is the trace-free **Fialkow
  tensor**, which *necessarily* contains ambient curvature (Weyl/Cotton) terms.
  The code's expression has none, so it is only the conformally-flat-ambient
  reduction and is not conformally invariant on a generic ambient.

### `M8` — Extrinsic `Q₄`: omits the entire intrinsic Paneitz term (fails the `S⁴` anchor)
*File:* `conformal_toolkit/hypersurface/extrinsic_q.py` · *Status:* **Documented (caveat in code)**

- **Claimed / code:** `q₄ = −Δ_h H + H|L₁|² + (n/2−1)H³`.
- **Anchor that fails:** on the round `S⁴` (umbilic, `L₁=0`, `H=1`) the code
  returns `0 + 0 + 1 = 1`, but a correct extrinsic `Q₄` must reduce to the
  **intrinsic Branson `Q₄ = (n−1)! = 6`**.
- **Correct:** `q₄ = Q₄^Σ + (extrinsic couplings)` with
  `Q₄^Σ = −Δ̄J̄ − 2|P̄|² + (n/2)J̄²` plus the leading `½ L₁·Δ̄L₁` term. The code
  omits the whole intrinsic piece. (Also: the module header writes `+Δ_h H`
  while the body uses `−Δ_h H`; the body's sign is kept.)

### `M9` — Carroll connection: `Γⁱ_{tj}=0` over-claimed as canonical
*File:* `conformal_toolkit/carroll/carroll_connection.py` · *Status:* **Scope caveat added**

- **Claimed:** `Γⁱ_{tj} = 0` (all), presented as *the* Carroll connection.
- **Why it's wrong:** with `v = ∂_t`, `(£_v h)_{ij} = ∂_t h_{ij}`, so metric
  compatibility forces `Γᵏ_{t(i}h_{j)k} = ½∂_t h_{ij} ≠ 0` whenever `h` is
  time-dependent. The module's own `carroll_electric_field` computes a
  generically nonzero `E = ½£_v h`, so zeroing all time-components is internally
  inconsistent. Valid only in the `£_v h = 0` (preserved-`h`) case.
- **Fix:** docstring now scopes the construction to `£_v h = 0` and gives the
  correct `Γⁱ_{(tj)} = −½ hⁱᵏ(£_v h)_{kj}` for the general case.

### `M10` — `is_bms_symmetry`: never tests the action on the time vector `v`
*File:* `conformal_toolkit/carroll/bms.py` · *Status:* **Caveat added (predicate is necessary, not sufficient)**

- **Claimed / code:** decides (conformal) Carroll symmetry purely from `£_ξ h`.
- **Why it's wrong:** a Carroll structure is the **pair** `(v, h)`; a conformal
  symmetry needs `£_ξ h = 2λh` **and** `£_ξ v = −λv`. Because `h` is degenerate
  (`h(v,·)=0`), the `h`-condition places *no* constraint on the `v`-direction,
  so the predicate can accept a `ξ` that moves `v` out of `ker(h)`. (The
  package's own supertranslation generator `ξ = f·v` happens to satisfy
  `£_ξ v = 0`, masking the gap for that input.)
- **Reference:** Duval–Gibbons–Horvathy.

### `M11` — Fefferman–Graham `g₄` at `n=4`: trace coefficient `2×` too large
*File:* `conformal_toolkit/poincare_einstein/fefferman_graham.py` · *Status:* **Fixed**

- **Claimed / code:** `g₄ = ⅛ tr(g₂²) g₀`, giving `tr g₄ = ½ tr(g₂²)`.
- **Correct:** `g₄ = 1/16 tr(g₂²) g₀`, so `tr g₄ = ¼ tr(g₂²) = ¼|P|²`
  (de Haro–Skenderis–Solodukhin algebraic constraint; cross-checks against the
  `1/16 g₀` sphere value in `C1`).

### `M12` — Renormalized volume `v₂` at `n=2`: sign error
*File:* `conformal_toolkit/poincare_einstein/renormalized_volume.py` · *Status:* **Fixed**

- **Claimed / code:** `v₂ = +J/2`.
- **Correct:** `v₂ = −J/2` (= `−½` on `S²`); the density `(1 − ρ²/4)²` has a
  *negative* `ρ²` coefficient. (2D sits at the `1/(n−2)` pole and is
  regularization-sensitive; we align the sign with the bulk volume expansion
  and with the corrected general `v₂ = −½J`.)

### `M13` — `n=3` holographic stress tensor: claimed traceless but isn't
*File:* `conformal_toolkit/poincare_einstein/dirichlet_neumann.py` · *Status:* **Fixed (trace-consistent placeholder)**

- **Claimed / code:** `T_{ab} = −3P_{ab} + (3/2)J g₀`, docstring: "trace anomaly
  is absent" (so `T` should be traceless in odd `n`).
- **Why it's wrong:** `tr T = −3J + (3/2)J·3 = (3/2)J ≠ 0` — it contradicts its
  own stated rationale. A trace-free local form `−3P + aJg₀` needs `a=1`.
- **Fix:** coefficient set to `a=1` (`T = −3P + Jg₀`, `tr T = 0`), with a
  docstring noting that the *genuine* `n=3` stress tensor is `T = 3g₍₃₎`,
  undetermined non-local VEV data — any local `−3P + cJg₀` is only a
  placeholder.

### `M14` — Discrete `Q₄` docstring: a false "identity"
*File:* `conformal_features/discrete/q_curvature.py` · *Status:* **Fixed (docstring)**

- **Claimed:** `H²−K` approximates GJMS `Q₄` "via the identity
  `Q₄ = 2K² − 2KH²` (rescaled)."
- **Why it's wrong:** `2K² − 2KH² = −2K·(H²−K)` — the two differ by the
  **non-constant** factor `−2K`, so neither is a rescaling of the other
  (counterexample `K=2, H=1`: `H²−K = −1` vs `2K²−2KH² = +4`). The false
  identity chain is removed; only the true statement `H²−K → 0 on S²` is kept.

### `M15` — Discrete `H²−K` mislabeled as GJMS `Q₄`; "pointwise conformally invariant" is false
*File:* `conformal_features/discrete/q_curvature.py` (and README feature table) · *Status:* **Fixed (docstring/labeling)**

- **Why it's wrong:** (a) the GJMS/Branson `Q₄` is a **4-manifold** object
  (`= 6` on `S⁴`); on a 2-surface, `H²−K` is the **Willmore integrand**, not
  the intrinsic `Q₄`. (b) `H²−K` is conformally invariant only **under the
  integral** — `∫(H²−K)dA` via Gauss–Bonnet (`∫K dA = 2πχ`) plus Möbius
  invariance of `∫H² dA` — **not pointwise**.
- **Fix:** relabeled as a Willmore-type surface feature; invariance qualified as
  integral-only. (Teaching Note 4.)

### `M16` — README Quick Start: `discrete_q_curvature(order=4).mean() → 5.94 (→6)` is wrong
*Files:* `README.md` Quick Start, `conformal_features/discrete/q_curvature.py` · *Status:* **Fixed (README)**

- **Claimed:** `Q4.mean() → 5.94 (converges to 6 as mesh refines)`.
- **How the code actually worked:** the function returns `H²−K`, which on an
  icosphere has mean `−0.0155, −0.0038, −0.0009` at subdivisions 2/3/4 —
  **monotonically converging to 0, not 6.** The continuum value `6 = (n−1)!`
  belongs to the intrinsic 4D `Q₄`, which a 2-surface feature cannot reproduce.
- **Fix:** README Quick Start corrected to `→ 0`; the symbolic `Q₄ = 6` example
  is kept separately and clearly labeled as the 4D object.

### `M17` — Discrete Bach proxy: wrong "K is already area-normalized" rationale; missing mass-inverse
*File:* `conformal_features/discrete/bach.py` · *Status:* **Fixed (rationale corrected; pointwise operator added as an option)**

- **Claimed (rationale):** raw `L@L@K` needs no area weighting "because `K` is
  already area-normalized."
- **Why it's wrong:** the cotangent matrix `L` is the FEM **stiffness** matrix —
  an *integrated* operator: for `f = x²` on a flat mesh, `(Lf)_i = −2A_i`,
  whereas `(M⁻¹Lf)_i = −2` recovers the pointwise Laplacian. So the pointwise
  bi-Laplacian is `(M⁻¹L)² = M⁻¹L M⁻¹L`, not raw `L L`, and the proxy's small
  magnitude on a sphere was partly a **scaling artifact** (the `A_i²` weighting
  suppresses it), not proof of correctness.
- **Fix:** the default feature channel keeps the integrated `|L L K|` (it is
  scale-stable across mesh resolutions, and the audit classed this as a
  *rationale* error, not a broken feature), but the docstring now states the FEM
  fact honestly and a `pointwise=True` option returns the mathematically-correct
  `M⁻¹L M⁻¹L K`. *Caveat we verified empirically:* the pointwise operator
  amplifies coarse-mesh curvature noise (the 12 pentagonal defects of an
  icosphere push it to mean ≈ 490 at subdivision 3), which is exactly why it is
  offered as an option rather than the default. (Teaching Note 5.)

---

## Minor

### `m1` — `Q₂ = R` vs Branson normalization of `Q₄`
*File:* `conformal_toolkit/core/q_curvature.py` · *Status:* **Documented (convention)** — `Q₂=R` is self-consistent with the README's own "scalar curvature" label, but inconsistent across orders with the Branson-normalized `Q₄` (Branson `Q₂ = R/2`). Now flagged in the module docstring.

### `m2` — `bach()` docstring missing the `1/(n−3)` factor
*File:* `conformal_toolkit/core/conformal_structure.py` · *Status:* **Fixed (docstring)** — the Weyl-divergence form needs `1/(n−3)` (since `∇ᶜW_{cabd} = (n−3)C_{abd}`); the **actual `bach.py` code already uses the Cotton form**, correct for all `n`. Docstring-only imprecision.

### `m3` — `_obstruction_6`: missing Graham–Hirachi constant; "exact on conformally flat" is vacuous
*File:* `conformal_toolkit/core/obstruction.py` · *Status:* **Documented** — returns the unnormalized `Δ(Bach)` principal part; the normalization constant `c` and curvature-squared corrections are absent, and `Bach=0 ⇒ ΔBach=0` makes the conformally-flat "exactness" the trivial `0=0`.

### `m4` — Tractor `μ`-slot weight label wrong
*File:* `conformal_toolkit/tractor/standard_tractor.py` · *Status:* **Fixed (docstring)** — `μ_a ∈ E_a[1]` has density weight **+1**, not `−1` (the label had been copied from the `ρ` slot). Decisive internal check: the package's own `tractor_metric.py` pairs `g^{ab}μ_aμ'_b` with `σρ'`, and homogeneity of that pairing forces `w(μ)=+1`. No computed value changes.

### `m5` — `CarrollStructure.is_valid` never checks `rank(h)=n`
*File:* `conformal_toolkit/carroll/carroll_structure.py` · *Status:* **Fixed (code)** — a zero or rank-deficient `h` previously passed validation despite the "rank-`n` degenerate" definition; a `rank(h) = dim−1` assertion was added.

### `m6` — "On S²: H=K=1" docstring is radius-specific
*File:* `conformal_features/discrete/q_curvature.py` · *Status:* **Fixed (docstring)** — on radius `R`, `H = 1/R`, `K = 1/R²`, so `H²−K = 0` for **all** `R`; only the antecedent `H=K=1` was `R=1`-specific.

---

## Style / Convention

### `s1` — `conformal_killing.py`: wrong *and* dead symmetrization line
*Status:* **Fixed (removed)** — `sym_nab_X = nab_X + nab_X.symmetrize()` is not
`∇_aX_b + ∇_bX_a` and was never used; the returned residual is built from the
correctly-formed `sym_tensor`, so the function's output was already correct. The
misleading line and its "we need 2× it" comment were deleted.

### `v1` — Discrete WKS omits per-energy normalization `C_e` and the standard `~7×` bandwidth
*File:* `conformal_features/discrete/spectral.py` · *Status:* **Documented (convention)** — the canonical Aubry–Schmidt–Cremers WKS includes a per-energy `C_e = (Σⱼ exp(…))⁻¹` and `σ ≈ 7δ`; the code's unnormalized variant is a valid feature but deviates from the textbook definition.

---

## Teaching notes (the five lessons most worth internalizing)

1. **A closed form must equal the number it annotates.** `2(n−1)!/((n/2−1)!)²`
   is `12` at `n=4`, but it was labeling `Q₄ = 6`. Evaluate before you trust.
   (`M1`)
2. **The conformal Laplacian is *not* the bare Laplacian.**
   `P₂ = Δ − (n−2)/(4(n−1)) R`. The curvature term *is* the conformal
   covariance; an operator called "conformal" with no curvature term is a red
   flag. (`M2`)
3. **Conformal weight is a free checksum.** A zeroth-order term added to a
   2nd-order Laplacian must carry weight `−2`, so it is **quadratic** in
   curvature, never linear in `H`; and a Laplacian-type operator **lowers**
   weight by 2 (bidegree difference `−2`). Weight bookkeeping catches structural
   and sign errors before any computation. (`M4`)
4. **Pointwise ≠ integral conformal invariance.** `H²−K` is the Willmore
   integrand: only `∫(H²−K)dA` is conformally invariant, not its pointwise
   value — and it is *not* the 4D GJMS `Q₄`. Know where your invariant lives.
   (`M14`/`M15`/`M16`)
5. **Discrete operators need the mass matrix.** The cotangent Laplacian `L` is a
   *stiffness* matrix; the pointwise operator is `M⁻¹L` and the bi-Laplacian is
   `(M⁻¹L)²`. "It converges to zero on the sphere" can be a false positive —
   validate discretizations on a **non-constant** field like `f = x²`. (`M17`)

---

## How to contribute a correction

We would rather be corrected than be wrong, and we designed this repository to
make corrections cheap and welcome. If you find an error — in the math, the
code, the docs, or in *this errata itself* — please:

1. **Open an issue** titled `Errata: <one-line claim>` (or comment on an
   existing `ERRATA <ID>`).
2. **State the claim and the counter-evidence.** The most useful reports follow
   the audit method above: give a *concrete geometry* (a sphere radius, a flat
   patch, an explicit metric) on which the claim gives the wrong number, or a
   *conformal-weight* argument showing the terms can't match. A failing
   numerical check on a named test metric is gold.
3. **Propose the fix** — the corrected formula, with the normalization
   convention you're using, and a reference if you have one.
4. If you can, **add a regression test** under `tests/` that pins the correct
   value on the anchor geometry. Verified-on-an-anchor beats argued-in-prose.

Open mathematical items where we explicitly want help: the complete weight-4
hypersurface invariant basis and its exact count (`M5`/`M6`), the Fialkow/Weyl
terms in `L₂` (`M7`), the full extrinsic `Q₄` (`M8`), the FG `g₄` Bach
differential terms (`C1`), and the Graham–Hirachi `n=6` obstruction constant
(`m3`).

*Found something we missed? That's the system working. Send it.*
