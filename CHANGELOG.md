# Changelog

## [0.1.0] - 2026-05-11

### Added
- Core conformal geometry: Schouten tensor, Bach tensor, Q-curvature (Q_2, Q_4), GJMS operators (P_2 Laplacian, P_4 Paneitz), obstruction tensor, conformal rescaling, conformal Killing vectors
- Tractor calculus: standard tractor bundle, tractor metric, tractor connection, tractor curvature, Thomas-D operator
- Conformal hypersurface invariants: conformal fundamental forms (L_1, L_2), Willmore energy densities (W_2, W_4), extrinsic Q-curvature, extrinsic GJMS operators, invariant enumeration
- Carroll geometry: Carroll structure, Carroll connection, spatial curvature, electric field, BMS symmetries
- Poincaré-Einstein: Fefferman-Graham expansion, Dirichlet-to-Neumann operator, holographic stress tensor, holographic Weyl anomaly, renormalized volume
- Export module: tensor_to_numpy, tensor_to_torch, conformal_feature_vector
- Discrete conformal features (PyTorch): mesh utilities, Gaussian/mean curvature, discrete Q-curvature, discrete Bach norm, Willmore density, cross-ratios, conformal factor via Yamabe flow
- Feature extraction pipeline: mesh_conformal_features with rotation-invariant features
- Benchmark scripts: ShapeNet classification, SHREC retrieval, FAUST correspondence
- 125 tests across both tracks
