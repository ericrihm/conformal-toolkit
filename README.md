# Conformal Toolkit

Two Python packages for conformal geometry — symbolic computation and geometric deep learning.

## Packages

### conformal_features (PyTorch)

Discrete conformal invariants as per-vertex features for triangle meshes. Pip-installable, no SageMath required.

```python
from conformal_features import mesh_conformal_features

features = mesh_conformal_features(vertices, faces)  # (V, 7) tensor
```

**Features extracted (7D conformal, 3D isometry):**
- Conformal factor (discrete Yamabe flow)
- Willmore density (H²)
- Q-curvature (orders 2 and 4)
- Bach tensor norm (iterated Laplacian proxy)
- Mobius-invariant cross-ratios
- Gaussian and mean curvature (isometry features)

### conformal_toolkit (SageMath)

Symbolic conformal geometry extending SageManifolds. Requires SageMath 10.x.

Implements: Schouten tensor, Bach tensor, Q-curvature, GJMS operators, tractor calculus, conformal fundamental forms, Carroll geometry, Poincare-Einstein metrics.

*Status: In development. Run via Docker: `docker compose run sage`*

## Installation

```bash
# Track B only (PyTorch features)
pip install -e ".[ml]"

# With benchmarks
pip install -e ".[ml,benchmarks]"

# Track A (requires SageMath via Docker)
docker compose run sage pytest tests/
```

## Benchmarks

```bash
python -m conformal_features.benchmarks.shapenet_classify --data_dir /path/to/shapenet --features conformal
python -m conformal_features.benchmarks.shrec_retrieval --data_dir /path/to/shrec --features conformal
python -m conformal_features.benchmarks.faust_correspondence --data_dir /path/to/faust --features conformal
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"
pytest tests/
```

## License

MIT
