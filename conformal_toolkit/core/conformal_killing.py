"""Conformal Killing vector computation.

A vector field X is a conformal Killing vector (CKV) of (M, g) if:
    £_X g = lambda * g
for some scalar function lambda (the conformal factor).

Equivalently, in index notation:
    ∇_a X_b + ∇_b X_a = (2/n)(∇_c X^c) g_ab

or, defining the conformal Killing residual:
    T_{ab} = ∇_a X_b + ∇_b X_a - (2/n)(div X) g_ab = 0

X is a Killing vector iff lambda = 0, i.e. div(X) = 0 and T = 0.
"""


def killing_conformal_factor(g, X):
    """Compute the conformal factor lambda = (2/n) div(X) for a vector field X.

    If X is a CKV then £_X g = lambda * g.

    Parameters
    ----------
    g : SageManifolds metric
        Riemannian or pseudo-Riemannian metric.
    X : SageManifolds vector field
        The candidate conformal Killing vector.

    Returns
    -------
    Scalar field: lambda = (2/n) * (∇_c X^c).
    """
    n = g.domain().dim()
    nabla = g.connection()

    # Lower X to a 1-form and compute its covariant derivative
    X_flat = X.down(g)           # (0,1) tensor
    nab_X = nabla(X_flat)        # (0,2): nab_X[a, b] = ∇_a X_b

    # div(X) = ∇_a X^a = trace of ∇X with one index raised
    nab_X_mixed = nab_X.up(g, 0)  # raise first index -> (1,1)
    div_X = nab_X_mixed.trace(0, 1)

    lam = (2 / n) * div_X
    return lam


def conformal_killing_equation(g, X):
    """Compute the conformal Killing equation residual.

    T_{ab} = ∇_a X_b + ∇_b X_a - (2/n)(div X) g_{ab}

    X is a CKV iff T = 0.

    Parameters
    ----------
    g : SageManifolds metric
        Riemannian or pseudo-Riemannian metric.
    X : SageManifolds vector field
        The candidate conformal Killing vector.

    Returns
    -------
    (0,2) tensor field T (the CKV residual).
    """
    n = g.domain().dim()
    nabla = g.connection()

    X_flat = X.down(g)           # (0,1): X_b
    nab_X = nabla(X_flat)        # (0,2): ∇_a X_b

    # Symmetrize: ∇_a X_b + ∇_b X_a
    sym_nab_X = nab_X + nab_X.symmetrize()  # symmetrize() gives (∇_a X_b + ∇_b X_a)/2
    # Correct: sym_nab_X = nab_X + nab_X.swap_adjacent_indices(0, 1) would swap,
    # but SageManifolds provides .symmetrize() which averages, so we need 2* it.
    # Instead: manually build the symmetrized version.
    # nab_X[a,b] = ∇_a X_b;  we want nab_X[a,b] + nab_X[b,a].

    M = g.domain()
    frame = list(M.frames())[0]

    sym_tensor = M.tensor_field(0, 2, name='symNabX', sym=(0, 1))
    for a in range(n):
        for b in range(a, n):
            comp_ab = nab_X[frame, a, b]
            comp_ba = nab_X[frame, b, a]
            if hasattr(comp_ab, 'expr'):
                comp_ab = comp_ab.expr()
            if hasattr(comp_ba, 'expr'):
                comp_ba = comp_ba.expr()
            val = (comp_ab + comp_ba).simplify_full()
            sym_tensor[frame, a, b] = val
            if a != b:
                sym_tensor[frame, b, a] = val

    # Conformal factor
    lam = killing_conformal_factor(g, X)

    # Residual T = (∇_a X_b + ∇_b X_a) - lambda * g_{ab}
    T = sym_tensor - lam * g
    return T


def is_conformal_killing(g, X, simplify=True):
    """Check whether X is a conformal Killing vector field.

    Parameters
    ----------
    g : SageManifolds metric
    X : SageManifolds vector field
    simplify : bool
        Whether to call simplify_full() on each component before testing.

    Returns
    -------
    (bool, scalar_field)
        (True if X is a CKV, lambda = (2/n)*div(X))
    """
    lam = killing_conformal_factor(g, X)
    T = conformal_killing_equation(g, X)

    n = g.domain().dim()
    M = g.domain()
    frame = list(M.frames())[0]

    for a in range(n):
        for b in range(n):
            comp = T[frame, a, b]
            if hasattr(comp, 'expr'):
                comp = comp.expr()
            if simplify:
                comp = comp.simplify_full()
            if not bool(comp == 0):
                return False, lam

    return True, lam
