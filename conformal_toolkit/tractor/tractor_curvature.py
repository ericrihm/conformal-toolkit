"""Tractor curvature of the standard tractor bundle.

The curvature Omega^A_{B ab} = [nabla^T_a, nabla^T_b] of the normal tractor
connection encodes the Weyl tensor (in dim >= 4) and the Cotton tensor
(in dim 3).  Specifically, the middle-slot component is:

    Omega^c_{d ab} = W^c_{d ab}  (Weyl tensor)

On conformally flat manifolds (e.g. round spheres, flat R^n),
the tractor curvature vanishes identically.
"""


def tractor_curvature(cs):
    """Return the Weyl tensor, which is the tractor curvature's key component.

    Parameters
    ----------
    cs : ConformalStructure

    Returns
    -------
    The Weyl tensor W^a_{bcd} (type (1,3) tensor), which is the
    middle-slot component of the tractor curvature endomorphism.
    On conformally flat spaces this vanishes.

    In dimension <= 2 the Weyl tensor vanishes identically, and in
    dimension 3 it also vanishes (the Cotton tensor is the obstruction
    to conformal flatness instead).  We return a zero tensor in those cases.
    """
    n = cs.dimension
    if n <= 2:
        # Weyl tensor is identically zero in dimension <= 2;
        # SageManifolds refuses to compute it, so build a zero (1,3) tensor.
        M = cs.manifold
        W = M.tensor_field(1, 3, name='W')
        frame = list(M.frames())[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        W[frame, i, j, k, l] = 0
        return W
    return cs.weyl()


def cotton_tensor(cs):
    """Compute the Cotton tensor C_abc = nabla_a P_bc - nabla_b P_ac.

    Parameters
    ----------
    cs : ConformalStructure

    Returns
    -------
    (0,3)-tensor antisymmetric in (a, b).
    """
    n = cs.dimension
    M = cs.manifold
    nabla = cs.connection()
    P = cs.schouten()

    nab_P = nabla(P)
    C = M.tensor_field(0, 3, name='C', antisym=(0, 1))
    frame = list(M.frames())[0]

    for a in range(n):
        for b in range(n):
            for c in range(n):
                val = nab_P[frame, a, b, c] - nab_P[frame, b, a, c]
                if hasattr(val, 'expr'):
                    C[frame, a, b, c] = val.expr()
                else:
                    C[frame, a, b, c] = val

    return C
