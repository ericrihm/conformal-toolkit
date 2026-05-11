"""Carroll geometry module for the conformal toolkit.

Provides tools for working with Carroll manifolds (M, v, h) — the
ultra-relativistic (c→0) limit of Lorentzian geometry — relevant to null
boundaries, asymptotic symmetries, and holography.

Public API
----------
CarrollStructure
    Central class representing a Carroll manifold (M, v, h).

carroll_connection.spatial_christoffel
    Spatial Christoffel symbols Γ^i_{jk} from h.

carroll_connection.carroll_christoffel_full
    Full (n+1)^3 Christoffel array (time components zero).

carroll_curvature.carroll_spatial_riemann
    Spatial Riemann tensor R^i_{jkl}.

carroll_curvature.carroll_spatial_ricci
    Spatial Ricci tensor R_{jl}.

carroll_curvature.carroll_electric_field
    Carroll electric field E_{ij} = (1/2) £_v h.

bms.bms_supertranslation_generator
    Vector field for a BMS supertranslation f * v.

bms.is_bms_symmetry
    Test whether a vector field is a (conformal) Carroll symmetry.
"""

from conformal_toolkit.carroll.carroll_structure import CarrollStructure
from conformal_toolkit.carroll.carroll_connection import (
    spatial_christoffel,
    carroll_christoffel_full,
)
from conformal_toolkit.carroll.carroll_curvature import (
    carroll_spatial_riemann,
    carroll_spatial_ricci,
    carroll_electric_field,
)
from conformal_toolkit.carroll.bms import (
    bms_supertranslation_generator,
    is_bms_symmetry,
)

__all__ = [
    "CarrollStructure",
    "spatial_christoffel",
    "carroll_christoffel_full",
    "carroll_spatial_riemann",
    "carroll_spatial_ricci",
    "carroll_electric_field",
    "bms_supertranslation_generator",
    "is_bms_symmetry",
]
