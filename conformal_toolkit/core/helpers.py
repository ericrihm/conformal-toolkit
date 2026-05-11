"""Tensor calculus helper functions for SageManifolds."""


def divergence(nabla, g, one_form):
    """Compute div(omega) = nabla^a omega_a for a 1-form omega."""
    d_omega = nabla(one_form)
    d_omega_up = d_omega.up(g, 0)
    return d_omega_up.trace(0, 1)


def laplacian(nabla, g, f):
    """Compute Laplacian: Delta(f) = nabla^a nabla_a f."""
    df = nabla(f)
    return divergence(nabla, g, df)


def tensor_norm_squared(g, T):
    """Compute |T|^2 = T_{a1...ak} T^{a1...ak} for a (0,k) tensor."""
    T_up = T.up(g)
    result = T
    for i in range(T.tensor_type()[1]):
        result = result.contract(0, T_up, 0)
    return result
