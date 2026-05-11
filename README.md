# Conformal Toolkit

[![Tests](https://github.com/ericrihm/conformal-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/ericrihm/conformal-toolkit/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![SageMath 10.x](https://img.shields.io/badge/SageMath-10.x-orange.svg)](https://sagemath.org)

**The first open-source toolkit for computational conformal geometry.**

Two Python packages that let you compute conformal invariants — the geometric quantities unchanged by local stretching of a surface — both symbolically (exact formulas via SageMath) and numerically on triangle meshes (GPU-ready via PyTorch).

---

## What is conformal geometry?

Imagine stretching a rubber sheet. Distances change, areas change, but **angles are preserved**. Conformal geometry studies exactly this: properties of shapes that survive arbitrary local stretching.

This matters because:

- **Physics**: The AdS/CFT correspondence, which connects gravity to quantum field theory, is built on conformal symmetry. The Weyl tensor, Q-curvature, and tractor bundles are the mathematical language.
- **Computer vision**: Two 3D scans of the same face under different expressions have different metrics but the same conformal structure — conformal features can recognize the face regardless of expression.
- **Geometry**: The Willmore energy measures how far a surface is from being a round sphere, conformally speaking. Its minimizers (Willmore surfaces) appear in biology (cell membranes), materials science, and geometric analysis.

This toolkit makes these abstract invariants **computable**.

---

## What can you do with it?

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

### Classify conformal hypersurface invariants (Blitz's theory)

How many independent conformal invariants does a surface-in-space have? The answer depends on dimension:

```python
from conformal_toolkit.hypersurface import list_invariants, count_invariants

# Weight-2 invariants: always exactly 1 (the Willmore integrand |L₁|²)
count_invariants(order=2, ambient_dim=4)  # → 1

# Weight-4 invariants: grows with dimension
count_invariants(order=4, ambient_dim=3)  # → 1 (just |L₂|²)
count_invariants(order=4, ambient_dim=5)  # → 4 (|L₂|², tr(L₁⁴), W·L₁, |W|²)

for inv in list_invariants(order=4, ambient_dim=5):
    print(f"  {inv['name']}: {inv['formula']}")
```

### Detect if a surface is "conformally round"

A surface is **umbilical** (conformally equivalent to a sphere) iff L₁ = 0:

```python
from conformal_toolkit.hypersurface import conformal_fundamental_form_L1, mean_curvature

# Unit sphere S² ⊂ R³: second fundamental form L = h (identity)
L1 = conformal_fundamental_form_L1(h_sphere, L_sphere)
L1.display()  # → 0  (sphere IS umbilical — it's conformally round)

# Cylinder S¹×R ⊂ R³: L has eigenvalues (1, 0)
L1 = conformal_fundamental_form_L1(h_cyl, L_cyl)
L1.display()  # → (1/2)dθ⊗dθ + (-1/2)dz⊗dz  (NOT umbilical)
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
| 1 | Willmore density | Conformal | H² (how far from minimal) |
| 2 | Q₂ | Conformal | Discrete scalar curvature 2K |
| 3 | Q₄ | Conformal | Higher-order curvature |
| 4 | Bach norm | Conformal | Measures non-conformal-flatness |
| 5-6 | Cross-ratio stats | Möbius | Edge cross-ratio mean and variance |
| 7 | Gaussian curvature | Isometric | Intrinsic curvature K |
| 8 | Mean curvature | Isometric | Extrinsic curvature H |
| 9 | H² − K | Isometric | Alternative Willmore density |

### See how a surface changes under conformal rescaling

```python
cs_flat = ConformalStructure(flat_metric)
cs_flat.schouten()  # → 0 (flat space has no curvature)

# Rescale: g_hat = e^{2x} g — stretching space exponentially in x
cs_hat = cs_flat.under_rescaling(x)
cs_hat.schouten()   # → non-zero! Rescaling creates curvature.
```

### Work with tractor calculus

Tractors are conformal geometry's fundamental algebraic tool — like spinors for conformal symmetry:

```python
from conformal_toolkit.tractor import StandardTractor, thomas_d, tractor_inner

# Thomas D-operator: maps functions to tractors
tractor = thomas_d(cs, f, weight=1)  # D_A f = (coefficient·f, ∇f, -Δf - wJf)
print(tractor.sigma, tractor.mu, tractor.rho)

# Tractor inner product
h = tractor_inner(cs, tractor1, tractor2)  # σρ' + ρσ' + g^{ab}μ_a μ'_b
```

### Compute holographic data from boundary geometry

Given a boundary metric, compute the bulk Poincaré-Einstein expansion:

```python
from conformal_toolkit.poincare_einstein import fg_expansion, holographic_weyl_anomaly

# Fefferman-Graham: g_bulk = ρ⁻²(dρ² + g₀ + ρ²g₂ + ρ⁴g₄ + ...)
coeffs = fg_expansion(g_boundary, order=4)
# coeffs[2] = -P(g₀)  (Schouten tensor determines the first correction)

# Holographic Weyl anomaly (the "a-anomaly")
anomaly = holographic_weyl_anomaly(g_boundary)  # → R/2 for 2D, Q₄ for 4D
```

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
├── conformal_features/         # PyTorch discrete package
│   ├── discrete/               # Curvature, Q, Bach, Willmore, cross-ratios, Yamabe, spectral
│   ├── features/               # mesh_conformal_features() pipeline
│   └── benchmarks/             # ShapeNet, SHREC, FAUST evaluation
├── tests/                      # 157 tests across both packages
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
| | `.gjms_operator(f, order)` | GJMS operator (P₂, P₄, or P₆) |
| | `.obstruction_tensor()` | Fefferman-Graham obstruction (n=4,6) |
| | `.under_rescaling(omega)` | New structure for e^{2ω}g |
| `tractor` | `StandardTractor(cs, σ, μ, ρ)` | Section of the rank-(n+2) tractor bundle |
| | `thomas_d(cs, f, weight)` | Thomas D-operator: density → tractor |
| | `tractor_connection(cs, I)` | Normal tractor connection ∇^T |
| `hypersurface` | `conformal_fundamental_form_L1(h, L)` | Trace-free 2nd fundamental form |
| | `willmore_density_W2(h, L)` | Willmore integrand \|L₁\|² |
| | `list_invariants(order, dim)` | Catalogue of independent invariants |
| `carroll` | `CarrollStructure(M, v, h)` | Degenerate geometry for c→0 limit |
| | `is_bms_symmetry(cs, ξ)` | Check BMS symmetry of a vector field |
| `poincare_einstein` | `fg_expansion(g₀, order)` | Fefferman-Graham coefficients |
| | `holographic_weyl_anomaly(g₀)` | Conformal anomaly density |
| `export` | `conformal_feature_vector(cs)` | Dict of all invariants at a point |
| | `tensor_to_numpy(T)` | SageMath tensor → NumPy array |

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

## Benchmarks

Three CLI tools for evaluating conformal features on standard shape analysis tasks:

```bash
# ShapeNet classification (conformal vs xyz vs HKS features)
conformal-shapenet --data-dir ./data/shapenet --feature-set conformal

# SHREC'17 shape retrieval (mAP via cosine similarity)
conformal-shrec --data-dir ./data/shrec

# FAUST human body correspondence (geodesic error via functional maps)
conformal-faust --data-dir ./data/faust
```

See [benchmarks/README.md](conformal_features/benchmarks/README.md) for dataset download instructions.

---

## Development

```bash
# Run all SageMath tests (auto-detects micromamba or Docker)
./sage-run.sh test

# Run specific Track A tests
./sage-run.sh pytest tests/test_core/ -v

# Run Track B tests (PyTorch only, no SageMath needed)
pytest tests/test_discrete/ tests/test_features/ -v
```

---

## Why this toolkit?

| Existing tool | What it does | What it doesn't do |
|--------------|-------------|-------------------|
| SageManifolds | General Riemannian geometry | No conformal specialization, no tractors, no GJMS |
| DiffusionNet | Surface learning with Laplacian features | No conformal invariants |
| Keenan Crane's tools | Conformal Willmore flow (C++) | Not pip-installable, not symbolic, no ML bridge |

**conformal-toolkit** is the first package that:
1. Implements tractor calculus in any computer algebra system
2. Computes Blitz's conformal fundamental forms as software
3. Bridges symbolic conformal geometry to discrete ML features

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

## License

MIT
