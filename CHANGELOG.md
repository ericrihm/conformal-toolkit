# Changelog

## [0.1.1] - 2026-05-31

### Fixed (mathematical correctness audit)
Independent adversarial re-audit of every quantitative claim; full record in
[ERRATA.md](ERRATA.md). Highlights:
- **Critical:** corrected Fefferman–Graham `g₄` (spurious `1/(n−4)` on the
  algebraic term; `n=4` trace coefficient `⅛ → 1/16`) and renormalized-volume
  `v₂` (`−1/(n−2)J → −½J`, and the `n=2` sign). (`C1`, `C2`, `M11`, `M12`)
- **Major:** the conformal Laplacian `P₂` now includes its curvature term
  `−(n−2)/(4(n−1))R` (was the bare Laplacian); the `n=3` holographic stress
  tensor is now actually traceless; the discrete Bach proxy uses the
  mass-inverse `M⁻¹L M⁻¹L` bi-Laplacian; removed a false discrete `Q₄` identity;
  relabeled discrete `H²−K` as the Willmore integrand (not 4D GJMS `Q₄`);
  corrected the `P₆`/README `−Δ³ → +Δ³` sign. (`M2`, `M3`, `M13`–`M17`)
- **Documented honestly (incomplete math flagged in-code, open for
  contribution):** extrinsic `Q₄`/`P₂` (`M4`, `M8`), `L₂` Fialkow terms (`M7`),
  weight-4 invariant under-count (`M5`, `M6`), Carroll connection scope and
  symmetry predicate (`M9`, `M10`).
- **Minor/style:** tractor `μ`-slot weight label `+1`; `bach()` `1/(n−3)`
  docstring; `is_valid` rank check; removed dead/incorrect symmetrization line;
  `Q₂` normalization caveat. (`m1`–`m6`, `s1`)
- Added [ERRATA.md](ERRATA.md), [docs/TOOLING_GAPS.md](docs/TOOLING_GAPS.md), and
  README "Verify it yourself" / "Contributing corrections" sections.

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
- 160 tests across both tracks (157 original + 3 errata regression anchors)
