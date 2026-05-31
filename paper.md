---
title: 'conformal-toolkit: Symbolic and Discrete Conformal Geometry for SageMath and PyTorch'
tags:
  - Python
  - conformal geometry
  - differential geometry
  - tractor calculus
  - geometric deep learning
  - SageMath
  - PyTorch
authors:
  - name: Eric Rihm
    orcid: # TODO
    affiliation: 1
  - name: Sam Blitz
    orcid: # TODO
    affiliation: 2
affiliations:
  - name: Independent Researcher
    index: 1
  - name: # TODO — confirm affiliation
    index: 2
date: 11 May 2026
bibliography: paper.bib
---

# Summary

`conformal-toolkit` is an open-source Python library that provides both
symbolic and discrete computational tools for conformal geometry. It consists of
two packages: `conformal_toolkit`, a SageMath extension implementing tractor
calculus, GJMS operators, conformal hypersurface invariants, Carroll geometry,
and Poincare-Einstein structures; and `conformal_features`, a pip-installable
PyTorch package that discretizes conformal invariants on triangle meshes and
exposes them as per-vertex feature vectors for graph neural networks. Together,
the packages bridge a gap between the theoretical conformal geometry literature
and applied geometric deep learning, providing researchers with the first
unified software environment for computing, exporting, and applying conformal
invariants from the symbolic to the discrete setting.

# Statement of Need

Conformal geometry -- the study of properties invariant under local rescalings
$g_{ab} \mapsto e^{2\omega} g_{ab}$ of the metric -- underpins major areas of
mathematical physics and geometric analysis. The Weyl tensor, Schouten tensor,
Q-curvature, GJMS operators, and tractor bundles appear throughout the AdS/CFT
correspondence, the analysis of Poincare-Einstein manifolds, and in conformal
field theory. Despite this centrality, no existing software package offers
dedicated conformal geometry functionality.

SageManifolds [@gourgoulhon_sagemanifolds], bundled with SageMath, provides
excellent general-purpose differential geometry -- metric tensors, connections,
curvature -- but does not implement conformal-specific constructions: there is
no tractor calculus, no GJMS operator beyond the Laplacian, no conformal
fundamental forms, and no support for conformal rescaling identities.
Researchers who wish to compute, for instance, the Paneitz operator $P_4$ on a
given metric, or the conformal fundamental forms $L_1, L_2$ of a hypersurface,
must derive everything by hand, an error-prone process for expressions involving
dozens of terms.

Recent work by Blitz [@blitz2023; @blitz2020; @blitz2025] has produced a
systematic classification of conformal hypersurface invariants, organized by
conformal weight and dimension. These results extend the classical Willmore
functional to higher-order analogues $W_4 = \int |L_2|^2 \, dA$ and establish
enumeration theorems for independent invariants. However, the classification
remains purely theoretical: no computational tools exist for researchers to
apply it to concrete geometries.

On the machine learning side, geometric deep learning methods such as
DiffusionNet [@sharp2022] and RIMeshGNN [@ho2024] achieve rotation and
translation invariance, but completely ignore conformal invariance. The Mobius
group -- conformal transformations of Euclidean space -- strictly contains the
isometry group, so conformal invariants capture geometric information that
isometric features miss. Providing these invariants as input features could
improve shape classification, retrieval, and correspondence tasks, but no
existing package makes conformal invariants available in a form suitable for
neural network training.

`conformal-toolkit` addresses all three gaps: it extends SageManifolds with a
comprehensive conformal geometry layer, provides the first computational
implementation of Blitz's hypersurface invariant classification, and bridges
symbolic conformal geometry to discrete mesh representations usable in PyTorch.

# Functionality

The software consists of two Python packages, 40 source modules in total, with
160 tests and 6 example notebooks.

## Symbolic Package: `conformal_toolkit`

The `conformal_toolkit` package extends SageMath's differential geometry
capabilities with five modules.

**Core curvature** (`conformal_toolkit.core`). The `ConformalStructure` class
wraps a SageManifolds metric and lazily computes derived conformal tensors on
demand: the Schouten tensor $P_{ab} = \frac{1}{n-2}\bigl(\mathrm{Ric}_{ab} -
\frac{R}{2(n-1)} g_{ab}\bigr)$, the Bach tensor $B_{ab}$, Q-curvature $Q_2$
and $Q_4$, GJMS operators $P_2$ (conformal Laplacian) and $P_4$ (Paneitz
operator) [@branson1995; @gjms1992], and the Fefferman-Graham obstruction tensor
[@fefferman1985]. A `conformal_killing` module solves the conformal Killing
equation on a given background.

**Tractor calculus** (`conformal_toolkit.tractor`). Implements the standard
tractor bundle of rank $n+2$, where sections are represented as triples $I^A =
(\sigma, \mu_a, \rho)$. The module provides the tractor metric $h_{AB}$, the
normal tractor connection $\nabla^T$, tractor curvature, and the Thomas
D-operator $D_A$ acting on conformal densities.

**Conformal hypersurface invariants** (`conformal_toolkit.hypersurface`).
Computes the conformal fundamental forms $L_1$ (trace-free second fundamental
form) and $L_2$ following Blitz's classification [@blitz2023; @blitz2020],
Willmore energy densities $W_2 = |L_1|^2$ and $W_4 = |L_2|^2$, extrinsic
Q-curvatures, and the extrinsic GJMS operator $P_2^{\Sigma}$. An
`invariant_enumeration` module catalogues the independent scalar conformal
hypersurface invariants at each weight and dimension.

**Carroll geometry** (`conformal_toolkit.carroll`). Implements the degenerate
Carroll structure $(M, v^a, h_{ab})$ arising in the $c \to 0$ ultra-relativistic
limit, including the Carroll connection, spatial curvature, and BMS symmetries
relevant to flat-space holography.

**Poincare-Einstein metrics** (`conformal_toolkit.poincare_einstein`). Computes
Fefferman-Graham expansion coefficients $g_{(2)} = -P(g_0)$ and $g_{(4)}$, the
Dirichlet-to-Neumann map, holographic stress tensor, Weyl anomaly, and
renormalized volume [@fefferman1985].

**Export layer** (`conformal_toolkit.export`). Functions `tensor_to_numpy`,
`tensor_to_torch`, and `conformal_feature_vector` extract numerical arrays from
symbolic tensor fields evaluated on coordinate grids, enabling direct transfer
of symbolic results into numerical and machine learning workflows.

## Discrete Package: `conformal_features`

The `conformal_features` package requires only PyTorch and NumPy (no SageMath)
and is pip-installable. Its core function `mesh_conformal_features(vertices,
faces)` takes a triangle mesh and returns a per-vertex feature tensor:

```python
import torch
from conformal_features import mesh_conformal_features

vertices = torch.rand(1000, 3)  # (V, 3) vertex positions
faces = torch.randint(0, 1000, (2000, 3))  # (F, 3) triangle indices

features = mesh_conformal_features(vertices, faces)  # (V, 10) tensor
```

The 10-dimensional output concatenates 7 conformal-invariant features --
discrete conformal factor (via Yamabe flow), Willmore density $H^2$,
Q-curvature at orders 2 and 4, discrete Bach norm, and mean and variance of
Mobius-invariant edge cross-ratios -- with 3 isometry-invariant features
(Gaussian curvature $K$, mean curvature $H$, and the alternative Willmore
density $H^2 - K$). The isometry features can be excluded via
`include_isometry=False` for ablation studies.

Three benchmark scripts (`conformal-shapenet`, `conformal-shrec`,
`conformal-faust`) evaluate the features on standard shape analysis tasks:
ShapeNet classification, SHREC'17 retrieval, and FAUST human body
correspondence.

# Related Work

**SageManifolds** [@gourgoulhon_sagemanifolds] provides the differential
geometry substrate on which `conformal_toolkit` builds. It handles manifolds,
metrics, connections, and generic curvature tensors, but does not specialize to
conformal geometry: no tractor calculus, GJMS operators, or conformal
hypersurface invariants are available.

**DiffusionNet** [@sharp2022] learns on surfaces using the Laplace-Beltrami
operator and achieves discretization-agnostic representations, but its features
are isometry-invariant, not conformally invariant. **RIMeshGNN** [@ho2024]
constructs rotation-invariant mesh features for GNNs but similarly does not
consider conformal symmetry. `conformal_features` provides a complementary
feature set that captures strictly more geometric information.

Keenan Crane's group has produced conformal Willmore flow implementations, and
Gruber and Aulisa have developed p-Willmore flow code. Both are specialized C++
implementations targeting energy minimization rather than feature extraction,
are not pip-installable, and do not integrate with SageMath or PyTorch
workflows.

No existing package combines symbolic conformal geometry computation with
discrete conformal feature extraction for machine learning.

# Acknowledgements

The conformal hypersurface invariant classification implemented in
`conformal_toolkit.hypersurface` is based on the mathematical work of Sam
Blitz. We thank the SageManifolds development team for the differential
geometry infrastructure on which the symbolic package builds.

# References
