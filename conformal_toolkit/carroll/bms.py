"""BMS (Bondi-Metzner-Sachs) symmetries of Carroll structures.

The BMS group is the asymptotic symmetry group of asymptotically flat
spacetimes.  At null infinity it acts on a Carroll manifold, with:

- **Supertranslations**: ξ = f ∂_t + (subleading spatial terms)
  parameterised by a function f on the celestial sphere.
- **Superrotations**: conformal Killing vectors of the sphere (extended BMS).

Carroll symmetries satisfy £_ξ h = 0 (strict) or £_ξ h = λ h (conformal).
"""

from sage.all import SR


def bms_supertranslation_generator(carroll_struct, f):
    """Construct the vector field for a BMS supertranslation.

    At leading order a supertranslation is:

        ξ = f · v

    where f is a function on the spatial slice (the celestial sphere in the
    physical context) and v is the Carroll time vector.

    Parameters
    ----------
    carroll_struct : CarrollStructure
    f : symbolic expression
        The supertranslation parameter (a function on M, typically depending
        only on spatial coordinates).

    Returns
    -------
    SageManifolds vector field
        The vector field ξ = f * v on the manifold.
    """
    v = carroll_struct.temporal_vector
    manifold = carroll_struct.manifold

    # Build ξ = f * v as a new vector field.
    xi = manifold.vector_field(name='xi')
    chart = manifold.default_chart()
    frame = chart.frame()
    dim = carroll_struct.dimension

    for mu in range(dim):
        comp = v[frame, mu]
        v_val = comp.expr() if hasattr(comp, 'expr') else SR(comp)
        xi[frame, mu] = f * v_val

    return xi


def is_bms_symmetry(carroll_struct, vector_field):
    """Check whether a vector field is a symmetry of the Carroll structure.

    A vector field ξ is a **strict Carroll symmetry** if £_ξ h = 0, and a
    **conformal Carroll symmetry** if £_ξ h = λ h for some scalar λ.

    INCOMPLETE PREDICATE -- necessary but NOT sufficient (see ERRATA M10).
    A Carroll structure is the PAIR (v, h), so a symmetry must preserve BOTH:
        strict:    £_ξ h = 0      and  £_ξ v = 0
        conformal: £_ξ h = 2λ h   and  £_ξ v = -λ v   (Duval-Gibbons-Horvathy)
    Because h is degenerate (h(v, ·) = 0), the £_ξ h condition places NO
    constraint on the v-direction, so this routine -- which only tests
    £_ξ h -- can wrongly accept a ξ that moves v out of ker(h). A correct
    predicate must ALSO verify £_ξ v ∝ v (strict: = 0; conformal: = -λ v).
    (The supertranslation generator ξ = f·v built above does satisfy
    £_ξ v = 0, so it passes; the gap bites for general user input.)

    The Lie derivative is computed component-wise:

        (£_ξ h)_{ij} = ξ^k ∂_k h_{ij} + h_{kj} ∂_i ξ^k + h_{ik} ∂_j ξ^k

    Parameters
    ----------
    carroll_struct : CarrollStructure
    vector_field : SageManifolds vector field
        The candidate symmetry generator ξ.

    Returns
    -------
    tuple (is_symmetry, is_conformal, lambda_factor)
        - ``is_symmetry`` (bool): True if ξ preserves the Carroll structure
          (strict or conformal).
        - ``is_conformal`` (bool): True if the symmetry is only conformal
          (£_ξ h = λ h with λ ≠ 0).
        - ``lambda_factor``: The conformal factor λ (or 0 for strict).
    """
    manifold = carroll_struct.manifold
    h = carroll_struct.spatial_metric
    xi = vector_field
    dim = carroll_struct.dimension
    n = dim - 1

    chart = manifold.default_chart()
    frame = chart.frame()
    coords = chart[:]

    # Extract h components (full manifold indices).
    h_comps = {}
    for mu in range(dim):
        for nu in range(dim):
            comp = h[frame, mu, nu]
            h_comps[(mu, nu)] = comp.expr() if hasattr(comp, 'expr') else SR(comp)

    # Extract ξ components.
    xi_comps = {}
    for mu in range(dim):
        comp = xi[frame, mu]
        xi_comps[mu] = comp.expr() if hasattr(comp, 'expr') else SR(comp)

    # Compute (£_ξ h)_{ij} for spatial i, j (manifold indices i+1, j+1).
    lie_h = {}
    for i in range(n):
        for j in range(n):
            mi = i + 1
            mj = j + 1

            term1 = SR(0)
            for k in range(dim):
                term1 += xi_comps[k] * h_comps[(mi, mj)].diff(coords[k])

            term2 = SR(0)
            for k in range(dim):
                term2 += h_comps[(k, mj)] * xi_comps[k].diff(coords[mi])

            term3 = SR(0)
            for k in range(dim):
                term3 += h_comps[(mi, k)] * xi_comps[k].diff(coords[mj])

            lie_h[(i, j)] = (term1 + term2 + term3).simplify_full()

    # Check if all components vanish (strict symmetry).
    all_zero = all(bool(v == 0) for v in lie_h.values())
    if all_zero:
        return (True, False, SR(0))

    # Check conformal symmetry: £_ξ h = λ h.
    # Try to read off λ from the first non-zero h_{ij}.
    lambda_factor = None
    is_conformal = True

    for (i, j), lie_val in lie_h.items():
        mi = i + 1
        mj = j + 1
        h_ij = h_comps[(mi, mj)]
        if bool(h_ij == 0):
            # If h_{ij}=0, the corresponding Lie component must also be 0.
            if not bool(lie_val == 0):
                is_conformal = False
                break
            continue
        # λ = (£_ξ h)_{ij} / h_{ij}
        candidate = (lie_val / h_ij).simplify_full()
        if lambda_factor is None:
            lambda_factor = candidate
        else:
            diff = (candidate - lambda_factor).simplify_full()
            if not bool(diff == 0):
                is_conformal = False
                break

    if is_conformal and lambda_factor is not None:
        return (True, True, lambda_factor)

    return (False, False, SR(0))
