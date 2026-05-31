# Conformal Toolkit

[![Tests](https://github.com/ericrihm/conformal-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/ericrihm/conformal-toolkit/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![SageMath 10.x](https://img.shields.io/badge/SageMath-10.x-orange.svg)](https://sagemath.org)

**Symbolic and discrete conformal geometry for SageMath and PyTorch.**

Two Python packages for computing conformal invariants — geometric quantities unchanged by local stretching — both symbolically (exact formulas via SageMath) and numerically on triangle meshes (GPU-ready via PyTorch). Implements tractor calculus, GJMS operators, Q-curvature, Blitz's conformal fundamental forms, Carroll geometry, and Fefferman-Graham holographic data.

> **This library was independently re-audited for mathematical correctness in May 2026.** Every confirmed error — and exactly how we caught it — is documented in **[ERRATA.md](ERRATA.md)**. We publish the full record on purpose: a teaching tool earns trust by showing its work, mistakes included. See **["Verify it yourself"](#verify-it-yourself)** and **["Contributing corrections"](#contributing-corrections)**.

---

## Quick Start

Compute the Branson Q-curvature exactly on the round 4-sphere, then extract a discrete Willmore feature on a mesh:

```python
# Symbolic: exact Q₄ on S⁴ via SageMath
from conformal_toolkit import ConformalStructure
cs = ConformalStructure(g_sphere4)
cs.q_curvature(order=4)  # → 6   (exact; Branson's Q_n(Sⁿ) = (n-1)! = 3! = 6)

# Discrete: a 4th-order surface feature on an icosphere mesh via PyTorch.
# NOTE: this returns the Willmore integrand H² − K, NOT the 4D GJMS Q₄
# (a 2-surface quantity cannot reproduce the 4-manifold value 6 — see ERRATA M15/M16).
from conformal_features.discrete.q_curvature import discrete_q_curvature
Q4 = discrete_q_curvature(vertices, faces, order=4)
Q4.mean()  # → 0   (H² − K vanishes on a round sphere of any radius; → 0 under refinement)
```

The symbolic formulas ground-truth the geometry; the discrete features are
mesh-domain analogues — *not* always the same number, and the docs now say which
is which.

---

## Installation

```bash
# PyTorch discrete features only (no SageMath needed)
pip install -e ".[ml]"

# Full development environment
pip install -e ".[dev,ml]"

# SageMath symbolic package — requires SageMath 10.x
# Option A: Native (fastest, especially on Apple Silicon)
micromamba create -n sage -c conda-forge sage python=3.11 -y
micromamba run -n sage sage -python -m pytest tests/test_core/ -v

# Option B: Docker
docker compose run sage
```

---

## What is conformal geometry?

Imagine stretching a rubber sheet. Distances change, areas change, but **angles are preserved**. Conformal geometry studies exactly this: properties of shapes that survive arbitrary local stretching.

This matters because:

- **Physics**: The AdS/CFT correspondence, which connects gravity to quantum field theory, is built on conformal symmetry. The Weyl tensor, Q-curvature, and tractor bundles are the mathematical language.
- **Computer vision**: Two 3D scans of the same face under different expressions have different metrics but the same conformal structure — conformal features can recognize the face regardless of expression.
- **Geometry**: The Willmore energy measures how far a surface is from being a round sphere, conformally speaking. Its minimizers (Willmore surfaces) appear in biology (cell membranes), materials science, and geometric analysis.

This toolkit makes these abstract invariants **computable**.

---

## Examples

### Compute curvature invariants on any Riemannian manifold

```python
from sage.all import Manifold, sin
from conformal_toolkit import ConformalStructure

# The round 2-sphere: ds² = dθ² + sin²θ dφ²
S = Manifold(2, 'S2', structure='Riemannian')
chart = S.chart(r'theta:(0,pi) phi:(0,2*pi)')
theta, phi = chart[:]
g = S.metric('g')
g[0,0] = 1
g[1,1] = sin(theta)**2

cs = ConformalStructure(g)

cs.ricci_scalar()       # → 2
cs.schouten()           # → P = (1/2)g  (Schouten tensor)
cs.q_curvature(order=2) # → 2  (Q₂ = scalar curvature)
cs.bach()               # → 0  (Bach vanishes — S² is conformally flat)
```

### Detect if a surface is conformally round (Blitz's theory)

A surface is **umbilical** (conformally equivalent to a sphere) iff L₁ = 0. The conformal fundamental forms ([Blitz–Gover–Waldron, *Indiana Univ. Math. J.* 2023](https://arxiv.org/abs/2107.10381)) are a hierarchy of extrinsic conformal invariants measuring successive orders of non-roundness:

```python
from conformal_toolkit.hypersurface import (
    conformal_fundamental_form_L1, willmore_density_W2, willmore_density_W4
)

# Unit sphere S² ⊂ R³: all principal curvatures equal
L1 = conformal_fundamental_form_L1(h_sphere, L_sphere)
L1.display()  # → 0  (sphere IS umbilical — conformally round)

# Cylinder S¹×R ⊂ R³: principal curvatures (1, 0)
L1 = conformal_fundamental_form_L1(h_cyl, L_cyl)
L1.display()  # → (1/2)dθ⊗dθ + (-1/2)dz⊗dz  (NOT umbilical)

willmore_density_W2(h_cyl, L_cyl)  # → |L₁|² = 1/2  (Willmore integrand)
```

### Classify conformal hypersurface invariants

How many independent conformal invariants does a hypersurface have? Blitz's classification theorem ([arXiv:2212.11711](https://arxiv.org/abs/2212.11711)) gives the answer:

```python
from conformal_toolkit.hypersurface import list_invariants, count_invariants

# Weight-2: always exactly 1 (the Willmore integrand |L₁|²)
count_invariants(order=2, ambient_dim=4)  # → 1

# Weight-4: grows with dimension
count_invariants(order=4, ambient_dim=3)  # → 1  (just |L₂|²)
count_invariants(order=4, ambient_dim=5)  # → 4

for inv in list_invariants(order=4, ambient_dim=5):
    print(f"  {inv['name']}: {inv['formula']}")
# → |L₂|², tr(L₁⁴), W·L₁, |W|²
```

### Extract conformal features for machine learning

No SageMath required — just PyTorch:

```python
import torch
from conformal_features import mesh_conformal_features

vertices = torch.randn(1000, 3)  # your mesh vertices
faces = torch.randint(0, 1000, (2000, 3))  # triangle connectivity

# 10D per-vertex features: 7 conformal + 3 isometric invariants
features = mesh_conformal_features(vertices, faces)  # → (1000, 10)

# Conformal-only (for Möbius-invariant tasks)
features = mesh_conformal_features(vertices, faces, include_isometry=False)  # → (1000, 7)
```

The 10 features per vertex:

| Index | Feature | Invariance | Description |
|-------|---------|-----------|-------------|
| 0 | Conformal factor | Conformal | From discrete Yamabe flow |
| 1 | Willmore density | Conformal | H² (distance from minimality) |
| 2 | Q₂ | Conformal | Discrete scalar curvature 2K |
| 3 | Willmore density H²−K | Conformal (integral) | 4th-order surface feature — the Willmore integrand, *not* the 4D GJMS Q₄ ([ERRATA M15](ERRATA.md)) |
| 4 | Bach norm | Conformal | Bi-Laplacian proxy for non-conformal-flatness |
| 5–6 | Cross-ratio stats | Möbius | Edge cross-ratio mean and variance |
| 7 | Gaussian curvature | Isometric | Intrinsic curvature K |
| 8 | Mean curvature | Isometric | Extrinsic curvature H |
| 9 | H² − K | Isometric | Alternative Willmore density |

**Conformal features are Möbius-invariant** — they survive inversions and other conformal deformations that destroy standard geometric features like HKS and Gaussian curvature. A shape classifier using conformal features recognizes a Möbius-deformed face; one using HKS does not.

### Work with tractor calculus

Tractors are conformal geometry's fundamental algebraic objects — like spinors for conformal symmetry. The standard tractor bundle carries a rank-(n+2) vector bundle with a canonical connection:

```python
from conformal_toolkit.tractor import StandardTractor, thomas_d, tractor_inner

# Thomas D-operator: maps conformal densities to tractors
T = thomas_d(cs, f, weight=1)
# → T.sigma = (n + 2w - 2) · w · f      (top slot)
# → T.mu    = (n + 2w - 2) · ∇f         (middle slot, a 1-form)
# → T.rho   = −Δf − w · J · f           (bottom slot)

# Tractor inner product: h(I,J) = σρ' + ρσ' + g^{ab}μ_a μ'_b
h = tractor_inner(cs, T1, T2)
```

### Compute holographic data from boundary geometry

Given a boundary metric, compute the bulk Poincaré-Einstein expansion:

```python
from conformal_toolkit.poincare_einstein import fg_expansion, holographic_weyl_anomaly

# Fefferman-Graham: g_bulk = ρ⁻²(dρ² + g₀ + ρ²g₂ + ρ⁴g₄ + ...)
coeffs = fg_expansion(g_boundary, order=4)
# → coeffs[2] = -P(g₀)  (Schouten tensor determines the first correction)

# Holographic Weyl anomaly (the "a-anomaly")
anomaly = holographic_weyl_anomaly(g_boundary)
# → R/2 for 2D boundary, Q₄ for 4D boundary
```

### Verify conformal covariance symbolically

The toolkit can serve as a proof machine — verify that conformal identities hold exactly:

```python
cs = ConformalStructure(g)
cs_hat = cs.under_rescaling(omega)

# The Weyl tensor is conformally invariant
(cs_hat.weyl() - cs.weyl()).display()  # → 0  (for any omega)

# The Bach tensor is conformally invariant in dimension 4
(cs_hat.bach() - cs.bach()).display()  # → 0
```

---

## Example Notebooks

| # | Notebook | What it demonstrates |
|---|---------|---------------------|
| 01 | [Curvature Invariants](examples/01_willmore_and_curvature.ipynb) | Schouten, Bach, Q-curvature, GJMS on S², S⁴, and flat space |
| 02 | [Conformal Hypersurface Invariants](examples/02_conformal_fundamental_forms.ipynb) | L₁, L₂, Willmore densities on sphere vs cylinder (Blitz classification) |
| 03 | [Tractor Calculus](examples/03_tractor_calculus.ipynb) | Standard tractors, tractor connection, Thomas D-operator |
| 04 | [Poincaré-Einstein](examples/04_poincare_einstein.ipynb) | Fefferman-Graham expansion, holographic anomaly |
| 05 | [Symbolic → ML Bridge](examples/05_bridge_symbolic_to_ml.ipynb) | Export symbolic invariants, compare to discrete approximations |
| 06 | [ML Shape Classification](examples/06_ml_shape_classification.ipynb) | Conformal features on meshes, rotation invariance verification |

---

## Architecture

```
conformal-toolkit/
├── conformal_toolkit/          # SageMath symbolic package
│   ├── core/                   # ConformalStructure, Schouten, Bach, Q, GJMS, CKV
│   ├── tractor/                # Standard tractor bundle, connection, Thomas-D
│   ├── hypersurface/           # L₁, L₂, Willmore, extrinsic Q, invariant enumeration
│   ├── carroll/                # Carroll geometry, BMS symmetries
│   ├── poincare_einstein/      # Fefferman-Graham, holographic data
│   └── export/                 # tensor_to_numpy, tensor_to_torch
├── conformal_features/         # PyTorch discrete package (no SageMath needed)
│   ├── discrete/               # Curvature, Q, Bach, Willmore, cross-ratios, Yamabe, spectral
│   ├── features/               # mesh_conformal_features() pipeline
│   └── benchmarks/             # ShapeNet, SHREC, FAUST evaluation (WIP)
├── tests/                      # 160 tests across both packages
├── examples/                   # 6 Jupyter notebooks
└── paper.md                    # JOSS paper draft
```

---

## Module Reference

### conformal_toolkit (SageMath)

| Module | Key Functions | What it computes |
|--------|-------------|-----------------|
| `core` | `ConformalStructure(g)` | Central class wrapping a metric |
| | `.schouten()` | Schouten tensor P_ab |
| | `.bach()` | Bach tensor B_ab |
| | `.q_curvature(order)` | Q-curvature (Q₂ or Q₄) |
| | `.gjms_operator(f, order)` | GJMS operator (P₂, P₄, or P₆†) |
| | `.obstruction_tensor()` | Fefferman-Graham obstruction (n=4, 6†) |
| | `.under_rescaling(omega)` | New structure for e^{2ω}g |
| `tractor` | `StandardTractor(cs, σ, μ, ρ)` | Section of the rank-(n+2) tractor bundle |
| | `thomas_d(cs, f, weight)` | Thomas D-operator: density → tractor |
| | `tractor_connection(cs, I)` | Normal tractor connection ∇^T |
| `hypersurface` | `conformal_fundamental_form_L1(h, L)` | Trace-free 2nd fundamental form |
| | `willmore_density_W2(h, L)` | Willmore integrand \|L₁\|² |
| | `list_invariants(order, dim)` | Catalogue of independent invariants |
| `carroll` | `CarrollStructure(M, v, h)` | Degenerate geometry for c → 0 limit |
| | `is_bms_symmetry(cs, ξ)` | Check BMS symmetry of a vector field |
| `poincare_einstein` | `fg_expansion(g₀, order)` | Fefferman-Graham coefficients |
| | `holographic_weyl_anomaly(g₀)` | Conformal anomaly density |
| `export` | `conformal_feature_vector(cs)` | Dict of all invariants at a point |
| | `tensor_to_numpy(T)` | SageMath tensor → NumPy array |

†P₆ computes the leading term (+Δ³) only; exact on conformally flat metrics. Obstruction at n=6 is a leading-order approximation (the Graham–Hirachi normalization constant is not applied — see [ERRATA m3](ERRATA.md)).

### conformal_features (PyTorch)

| Function | Input | Output |
|----------|-------|--------|
| `mesh_conformal_features(V, F)` | Vertices (V,3), faces (F,3) | Per-vertex features (V, 7 or 10) |
| `discrete_gaussian_curvature(V, F)` | Mesh | K per vertex via angle defect |
| `discrete_mean_curvature(V, F)` | Mesh | H per vertex via cotangent Laplacian |
| `discrete_q_curvature(V, F, order)` | Mesh | Q₂ or Q₄ per vertex |
| `discrete_conformal_factor(V, F)` | Mesh | Conformal factor via Yamabe flow |
| `discrete_cross_ratios(V, F)` | Mesh | Möbius-invariant edge cross-ratios |
| `cotangent_laplacian(V, F)` | Mesh | Sparse (V,V) cotangent weight matrix |
| `lbo_eigenvectors(V, F, k)` | Mesh | First k LBO eigenvalues + eigenvectors |
| `heat_kernel_signature(V, F, k)` | Mesh | HKS descriptors (V, T) |
| `wave_kernel_signature(V, F, k)` | Mesh | WKS descriptors (V, E) |

---

## Why this toolkit?

Existing tools cover parts of this space, but none bridges symbolic conformal differential geometry with discrete ML features:

| Existing tool | What it does well | What it doesn't cover |
|--------------|-------------------|----------------------|
| [SageManifolds](https://sagemanifolds.obspm.fr/) | General differential geometry: Riemann, Ricci, Weyl, Schouten, Cotton | No Q-curvature, GJMS operators, tractor calculus, conformal hypersurface invariants, or ML export |
| [xAct](https://xact.es/) (Mathematica) | Abstract-index tensor algebra for GR; Weyl tensor | No Schouten, Q-curvature, tractor calculus, or GJMS. Requires Mathematica |
| [DiffusionNet](https://github.com/nmwsharp/diffusion-net) | Deep learning on meshes via learned diffusion; HKS features | HKS is isometry-invariant, not conformally invariant. No symbolic geometry |
| [geometry-central](https://geometry-central.net/) / [libigl](https://libigl.github.io/) | Discrete conformal parameterization (BFF, LSCM) | C++/mesh-processing "conformal" (angle-preserving maps), not smooth conformal DG |
| [geomstats](https://geomstats.github.io/) | Riemannian geometry for ML on specific manifolds | Riemannian, not conformal. No Weyl, Q-curvature, tractors |

To the best of our knowledge, `conformal-toolkit` is the first reusable, Python-accessible implementation of:

1. **Tractor calculus** — standard tractors, tractor connection, Thomas D-operator, tractor curvature
2. **GJMS operators and Q-curvature** — Paneitz operator P₄, Q₂, Q₄ as general-purpose functions
3. **Blitz's conformal fundamental forms** — L₁, L₂, Willmore densities, invariant enumeration ([arXiv:2107.10381](https://arxiv.org/abs/2107.10381), [arXiv:2212.11711](https://arxiv.org/abs/2212.11711))
4. **Symbolic-to-discrete bridge** — the same conformal invariants computed exactly via SageMath and approximately on meshes via PyTorch

Prior computational work exists in FORM scripts (ancillary files of [arXiv:2107.10381](https://arxiv.org/abs/2107.10381)) and narrow Mathematica packages ([Peterson, UND Commons](https://commons.und.edu/data/13/)), but not as a standalone, pip-installable toolkit.

---

## Benchmarks (Work in Progress)

Three CLI tools for evaluating conformal features on standard shape analysis tasks are scaffolded:

```bash
conformal-shapenet --data-dir ./data/shapenet --feature-set conformal
conformal-shrec --data-dir ./data/shrec
conformal-faust --data-dir ./data/faust
```

Feature extraction and model architectures are implemented; dataset integration and training loops are in progress.

---

## Development

```bash
# Run all SageMath tests (auto-detects micromamba or Docker)
./sage-run.sh test

# Run specific symbolic tests
./sage-run.sh pytest tests/test_core/ -v

# Run discrete tests (PyTorch only, no SageMath needed)
pytest tests/test_discrete/ tests/test_features/ -v
```

---

## Verify it yourself

Don't take our word for any formula — the whole point of a symbolic toolkit is
that you can check it. Every correction in [ERRATA.md](ERRATA.md) was caught by
evaluating a claim on a geometry where the answer is known in closed form. Here
are the anchors we use; copy them into a Sage session and confirm:

```python
from sage.all import Manifold, sin
from conformal_toolkit import ConformalStructure

# --- Anchor 1: the round 4-sphere, where Branson's Q_n(Sⁿ) = (n-1)! ---
# On Sⁿ (sectional curvature 1):  Ric = (n-1)g,  R = n(n-1),
#   Schouten P = ½g,  J = tr P = n/2,  so  Q₄ = -ΔJ - 2|P|² + (n/2)J² = 6.
cs = ConformalStructure(g_sphere4)
assert cs.q_curvature(order=4) == 6          # = (4-1)! = 3!  (ERRATA M1)

# --- Anchor 2: the conformal Laplacian carries a curvature term ---
# P₂ f = Δf - (n-2)/(4(n-1)) R f.  The R-term is NOT optional for n > 2.
# On S⁴ its coefficient is n(n-2)/4 = 2, never zero.  (ERRATA M2)

# --- Anchor 3: Fefferman-Graham on the hyperbolic filling of Sⁿ ---
# g_ρ = (1 - ρ²/4)² g₀  ⟹  g₂ = -½ g₀  and  g₄ = 1/16 g₀ exactly.  (ERRATA C1/M11)

# --- Anchor 4 (discrete, PyTorch only): validate on a NON-constant field ---
# The cotangent Laplacian L is a *stiffness* matrix; for f = x² on a flat mesh,
# (L f)_i = -2·A_i  while  (M⁻¹ L f)_i = -2  recovers the pointwise Laplacian.
# "It vanishes on a sphere" is a false positive — constants are annihilated by
# any linear operator.  (ERRATA M17)
```

The method generalizes: **reduce a tensor claim to a scalar on a known geometry,
and check conformal weights as a free checksum.** That single discipline caught
most of the errata.

---

## Contributing corrections

We would rather be corrected than be wrong, and this repository is built to make
that easy. **Finding an error we missed is the system working — please send it.**

1. Open an issue titled `Errata: <one-line claim>`.
2. Give the counter-evidence the way we give ours: a *concrete geometry* (a
   sphere radius, a flat patch, an explicit metric) on which the claim returns
   the wrong number, or a *conformal-weight* argument that the terms can't match.
   A failing check on a named anchor metric is the gold standard.
3. Propose the corrected formula, stating your normalization convention (Branson
   vs. analyst signs differ — half of conformal geometry's "errors" are
   convention clashes), with a reference if you have one.
4. If you can, add a regression test under `tests/` pinning the right value on
   the anchor. *Verified-on-an-anchor beats argued-in-prose.*

Open problems where we explicitly want help are listed at the end of
[ERRATA.md](ERRATA.md) — the complete weight-4 hypersurface invariant basis
(`M5`), the Fialkow/Weyl terms in `L₂` (`M7`), the full extrinsic `Q₄` (`M8`),
and the FG `g₄` Bach differential terms (`C1`).

---

## Citation

```bibtex
@software{conformal_toolkit,
  author = {Rihm, Eric and Blitz, Samuel},
  title  = {conformal-toolkit: Symbolic and Discrete Conformal Geometry
            for SageMath and PyTorch},
  year   = {2026},
  url    = {https://github.com/ericrihm/conformal-toolkit}
}
```

The conformal hypersurface module implements theory from:

```bibtex
@article{blitz2023conformal,
  author  = {Blitz, Samuel and Gover, A. Rod and Waldron, Andrew},
  title   = {Conformal Fundamental Forms and the Asymptotically
             {Poincar\'{e}--Einstein} Condition},
  journal = {Indiana University Mathematics Journal},
  volume  = {72},
  number  = {6},
  pages   = {2215--2284},
  year    = {2023},
  doi     = {10.1512/iumj.2023.72.9578}
}

@article{blitz2022classification,
  author  = {Blitz, Samuel},
  title   = {Toward a Classification of Conformal Hypersurface Invariants},
  journal = {Journal of Mathematical Physics},
  volume  = {64},
  year    = {2023},
  doi     = {10.1063/5.0147870}
}

@article{blitz2024willmore,
  author  = {Blitz, Samuel and Gover, A. Rod and Waldron, Andrew},
  title   = {Generalized {W}illmore Energies, {Q}-Curvatures, Extrinsic
             {P}aneitz Operators, and Extrinsic {L}aplacian Powers},
  journal = {Communications in Contemporary Mathematics},
  year    = {2024},
  doi     = {10.1142/S0219199723500530}
}
```

## License

MIT
