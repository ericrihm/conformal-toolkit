"""Conformal invariant feature vector extraction from a ConformalStructure."""


def _norm_sq_02(g, T):
    """Compute |T|^2 = T_{ab} T^{ab} for a symmetric (0,2) tensor T.

    SageManifolds T.up(g) on a (0,2) tensor raises *both* indices at once
    producing a (2,0) tensor.  We then contract T_{ab} with T^{ab} by
    raising one index at a time via T.contract(0, T.up(g, 0), 0).trace(0, 1).
    """
    # Raise only the first index: (0,2) -> (1,1)
    T_mixed = T.up(g, 0)                   # T^a_b
    # Contract T_{ab} (0,2) with T^a_b (1,1): first index of T_mixed (contra)
    # against first index of T (covariant) -> (0,2) result... no.
    # Easier: T_{ab} T^{ab} = T_{ab} g^{ac} g^{bd} T_{cd}
    # = T_mixed^a_b * T^b_a = trace of (T_mixed * T_mixed).
    # (T_mixed)^a_b contracted with (T_mixed)^b_a:
    prod = T_mixed.contract(1, T_mixed, 0)  # (1,1) * (1,1) on indices 1 and 0 -> (1,1)
    return prod.trace(0, 1)


def _norm_sq_04(g, T):
    """Compute |T|^2 = T_{abcd} T^{abcd} for a (0,4) tensor T.

    Uses a component loop: T_up = g^{ae} g^{bf} g^{cg} g^{dh} T_{efgh},
    then contracts with T_{abcd}.
    """
    M = g.domain()
    n = M.dim()
    frame = list(M.frames())[0]

    # Raise all four indices to get T^{abcd}
    T_up = T.up(g)  # (4,0)

    total = 0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    t_down = T[frame, a, b, c, d]
                    t_up = T_up[frame, a, b, c, d]
                    if hasattr(t_down, 'expr'):
                        t_down = t_down.expr()
                    if hasattr(t_up, 'expr'):
                        t_up = t_up.expr()
                    total += t_down * t_up
    return total


def conformal_feature_vector(cs, point_dict=None, chart=None):
    """Extract a vector of conformal invariants from a ConformalStructure.

    Parameters
    ----------
    cs : ConformalStructure
        The conformal structure to extract features from.
    point_dict : dict, optional
        Mapping {coord_symbol: value} for numeric evaluation.
        If None, returns symbolic expressions.
    chart : SageManifolds chart, optional
        Chart to read tensor components from.

    Returns
    -------
    dict with keys:
        'scalar_curvature' : R
        'schouten_trace'   : J = R/(2*(n-1))
        'q2'               : Q_2 = R
        'q4'               : Q_4 (if dim >= 4)
        'bach_norm'        : |B|^2 (if dim >= 4)
        'weyl_norm'        : |W|^2 (if dim >= 4)

    Values are float if point_dict is provided, otherwise symbolic.
    """
    from conformal_toolkit.export.to_numpy import scalar_at_point

    g = cs.metric
    n = cs.dimension

    if chart is None:
        chart = list(cs.manifold.top_charts())[0]

    def _scalar_value(sf):
        """Return float or symbolic value of a scalar field."""
        if point_dict is not None:
            return scalar_at_point(sf, point_dict, chart=chart)
        expr = sf.expr(chart)
        if hasattr(expr, 'is_numeric') and expr.is_numeric():
            return float(expr)
        return expr

    def _expr_value(expr):
        """Return float or symbolic from a raw expression (not a scalar field)."""
        if point_dict is not None:
            if hasattr(expr, 'subs'):
                val = expr.subs(point_dict)
            else:
                val = expr
            return float(val)
        if hasattr(expr, 'is_numeric') and expr.is_numeric():
            return float(expr)
        return expr

    R = cs.ricci_scalar()
    J = cs.schouten_trace()

    features = {
        'scalar_curvature': _scalar_value(R),
        'schouten_trace': _scalar_value(J),
        'q2': _scalar_value(cs.q_curvature(order=2)),
    }

    if n >= 4:
        features['q4'] = _scalar_value(cs.q_curvature(order=4))

        # |B|^2 — Bach is (0,2)
        B = cs.bach()
        B_norm_sq = _norm_sq_02(g, B)
        # B_norm_sq is a scalar field or expression
        if hasattr(B_norm_sq, 'expr'):
            B_val = _expr_value(B_norm_sq.expr(chart) if hasattr(B_norm_sq, 'expr') else B_norm_sq)
        else:
            B_val = _expr_value(B_norm_sq)
        features['bach_norm'] = B_val

        # |W|^2 — SageManifolds returns Weyl as (1,3); lower to (0,4) first.
        W = cs.weyl()
        wtype = W.tensor_type()
        if wtype == (1, 3):
            # Lower the single contravariant index to get (0,4)
            W_04 = W.down(g, 0)
        else:
            # Already (0,4)
            W_04 = W
        W_norm_sq = _norm_sq_04(g, W_04)
        if hasattr(W_norm_sq, 'expr'):
            W_val = _expr_value(W_norm_sq.expr(chart) if hasattr(W_norm_sq, 'expr') else W_norm_sq)
        else:
            W_val = _expr_value(W_norm_sq)
        features['weyl_norm'] = W_val

    return features
