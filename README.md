# Conformal Toolkit

Symbolic conformal geometry for SageMath + discrete conformal features for geometric deep learning.

## Mathematical Context

Conformal geometry studies the properties of manifolds that are preserved under local
rescalings of the metric: g_ab -> e^{2*omega} g_ab.  The resulting invariant objects —
the Weyl tensor, Schouten tensor, Q-curvature, tractor bundles, and GJMS operators —
appear throughout mathematical physics: in the AdS/CFT correspondence, in the analysis
of conformally-flat initial data for the Einstein equations, in the study of CR manifolds
and Poincare-Einstein spaces, and in the ultra-relativistic Carroll limit relevant to
flat-space holography.

This toolkit provides two complementary interfaces to these structures.  The SageMath
package `conformal_toolkit` is aimed at symbolic computation: deriving exact formulae,
verifying curvature identities, and exploring new geometric constructions.  The PyTorch
package `conformal_features` translates the key invariants to the discrete (mesh)
setting and exposes them as per-vertex feature vectors suitable for graph neural networks
and shape-analysis benchmarks.

## Features

### conformal_toolkit (SageMath)

**Core curvature (`conformal_toolkit.core`)**
- `ConformalStructure` — central class; lazy-cached Schouten, Weyl, Bach, Q-curvature
- Schouten tensor P_ab and trace J
- Bach tensor B_ab (conformally-Einstein obstruction in dimension 4)
- Q-curvature Q_4 and GJMS operators P_{2k}
- Fefferman-Graham obstruction tensor (even dimensions)
- Conformal Killing equation solver

**Tractor calculus (`conformal_toolkit.tractor`)**
- `StandardTractor` — sections of the rank-(n+2) tractor bundle I^A = (sigma, mu_a, rho)
- Normal tractor connection nabla^T
- Thomas D-operator D_A acting on conformal densities

**Hypersurface geometry (`conformal_toolkit.hypersurface`)**
- Conformal fundamental forms L_1 (trace-free second fundamental form) and L_2
- Mean curvature and umbilicity detection

**Carroll geometry (`conformal_toolkit.carroll`)**
- `CarrollStructure` — degenerate geometry (M, v, h) for the c -> 0 limit
- Spatial Christoffel symbols and Carroll electric field E_{ij} = (1/2) L_v h

**Poincare-Einstein metrics (`conformal_toolkit.poincare_einstein`)**
- Fefferman-Graham expansion g = rho^{-2}(drho^2 + g_0 + rho^2 g_2 + rho^4 g_4 + ...)
- Coefficients g_2 = -P(g_0) and g_4 (with the n=4 log-term caveat)

### conformal_features (PyTorch)

Discrete conformal invariants as per-vertex features for triangle meshes.
No SageMath required; pip-installable.

**7D feature vector (conformal) + 3D (isometry):**
- Conformal factor (discrete Yamabe flow)
- Willmore density H^2
- Q-curvature (orders 2 and 4)
- Bach tensor norm (iterated Laplacian proxy)
- Mobius-invariant cross-ratios
- Gaussian curvature K and mean curvature H

## Installation

```bash
# PyTorch features only (no SageMath required)
pip install -e ".[ml]"

# With benchmark dependencies
pip install -e ".[ml,benchmarks]"

# Development (linting, testing, PyTorch)
pip install -e ".[dev,ml]"

# SageMath symbolic package — requires SageMath 10.x
# Option A: native micromamba environment (fastest on Apple Silicon)
micromamba create -n sage -c conda-forge sage python=3.11 -y
micromamba activate sage
sage -python -m pytest tests/

# Option B: Docker (x86 emulation via Rosetta)
docker compose run sage pytest tests/
```

## Quick Start — Symbolic (SageMath)

```python
from sage.manifolds.manifold import Manifold
from conformal_toolkit.core.conformal_structure import ConformalStructure

# Build the round 2-sphere
S2 = Manifold(2, 'S^2', structure='Riemannian')
U = S2.open_subset('U')
phi = U.chart('th ph')
th, ph = phi[:]

g = S2.metric('g')
g[0, 0] = 1
g[1, 1] = sin(th)**2

cs = ConformalStructure(g)

# Schouten tensor: P_ab = (1/(n-2))(Ric_ab - R/(2(n-1)) g_ab)
P = cs.schouten()
print(P.display())   # P = (1/2) dth*dth + (1/2) sin(th)^2 dph*dph

# Ricci scalar
R = cs.ricci_scalar()
print(R)             # 2  (round S^2 has R = 2)
```

## Quick Start — Discrete Features (PyTorch)

```python
import torch
from conformal_features import mesh_conformal_features

# vertices: (V, 3) float tensor, faces: (F, 3) long tensor
vertices = torch.load('mesh_vertices.pt')
faces    = torch.load('mesh_faces.pt')

features = mesh_conformal_features(vertices, faces)  # -> (V, 7) tensor

# Pass directly to a GNN
from torch_geometric.nn import GCNConv
conv = GCNConv(in_channels=7, out_channels=64)
```

## Module Overview

| Module | Contents |
|---|---|
| `conformal_toolkit.core.conformal_structure` | `ConformalStructure` class — main entry point |
| `conformal_toolkit.core.schouten` | Schouten tensor P_ab and trace J |
| `conformal_toolkit.core.bach` | Bach tensor B_ab |
| `conformal_toolkit.core.q_curvature` | Q-curvature Q_4 |
| `conformal_toolkit.core.gjms` | GJMS operators P_{2k} |
| `conformal_toolkit.core.obstruction` | Fefferman-Graham obstruction tensor |
| `conformal_toolkit.tractor.standard_tractor` | `StandardTractor` — tractor bundle sections |
| `conformal_toolkit.tractor.tractor_connection` | Normal tractor connection nabla^T |
| `conformal_toolkit.tractor.thomas_d` | Thomas D-operator |
| `conformal_toolkit.hypersurface.conformal_fundamental_form` | L_1, L_2 conformal fundamental forms |
| `conformal_toolkit.carroll.carroll_structure` | `CarrollStructure` (M, v, h) |
| `conformal_toolkit.carroll.carroll_connection` | Spatial Christoffel symbols |
| `conformal_toolkit.carroll.carroll_curvature` | Carroll electric field E_{ij} |
| `conformal_toolkit.poincare_einstein.fefferman_graham` | FG expansion coefficients g_2, g_4 |
| `conformal_features` | Discrete conformal feature extraction |

## Development

```bash
# Run SageMath tests (auto-detects micromamba or Docker)
./sage-run.sh test

# Run a specific test file
./sage-run.sh pytest tests/test_schouten.py -v

# Run PyTorch tests only (no SageMath needed)
pytest tests/test_features.py -v
```

## Citation

If you use this toolkit in academic work, please cite:

```bibtex
@software{conformal_toolkit,
  author  = {Blitz, Samuel and Rihm, Eric},
  title   = {Conformal Toolkit: Symbolic Conformal Geometry and Discrete
             Conformal Features for Geometric Deep Learning},
  year    = {2024},
  url     = {https://github.com/ericrihm/conformal-toolkit},
}
```

## License

MIT
