"""Bach tensor computation.

B_ab = nabla^c C_cab + W_{cadb} P^{cd}

where C_abc = nabla_a P_bc - nabla_b P_ac is the Cotton tensor,
W is the Weyl tensor, P is the Schouten tensor.
"""


def compute_bach(cs):
    """Compute the Bach tensor B_ab."""
    n = cs.dimension
    if n < 4:
        raise ValueError(f"Bach tensor requires dimension >= 4, got {n}")

    g = cs.metric
    nabla = cs.connection()
    P = cs.schouten()
    W = cs.weyl()

    nab_P = nabla(P)

    M = cs.manifold
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

    nab_C = nabla(C)
    nab_C_up = nab_C.up(g, 0)
    term1 = nab_C_up.trace(0, 1)

    P_up = P.up(g)
    W_down = W.down(g, 0)

    B = M.tensor_field(0, 2, name='B', sym=(0, 1))
    chart = list(M.top_charts())[0]

    for a in range(n):
        for b in range(a, n):
            term2_val = 0
            for c in range(n):
                for d in range(n):
                    w_comp = W_down[frame, c, a, d, b]
                    p_comp = P_up[frame, c, d]
                    if hasattr(w_comp, 'expr'):
                        w_comp = w_comp.expr()
                    if hasattr(p_comp, 'expr'):
                        p_comp = p_comp.expr()
                    term2_val += w_comp * p_comp

            term1_comp = term1[frame, a, b]
            if hasattr(term1_comp, 'expr'):
                term1_comp = term1_comp.expr()

            bach_comp = (term1_comp + term2_val).simplify_full()
            B[frame, a, b] = bach_comp
            if a != b:
                B[frame, b, a] = bach_comp

    return B
