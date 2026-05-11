"""Normal tractor connection on the standard tractor bundle.

The tractor connection nabla^T_a acting on a standard tractor (sigma, mu_b, rho):

    nabla^T_a (sigma, mu_b, rho) = (
        nabla_a sigma - mu_a,
        nabla_a mu_b + P_ab sigma + g_ab rho,
        nabla_a rho - P_a^b mu_b
    )

where P is the Schouten tensor and nabla is the Levi-Civita connection.
"""


def tractor_connection(cs, tractor):
    """Apply the normal tractor connection to a standard tractor.

    Parameters
    ----------
    cs : ConformalStructure
    tractor : StandardTractor

    Returns
    -------
    dict with keys 'sigma', 'mu', 'rho':
        - 'sigma' : 1-form (nabla_a sigma - mu_a)
        - 'mu' : (0,2)-tensor (nabla_a mu_b + P_ab sigma + g_ab rho)
        - 'rho' : 1-form (nabla_a rho - P_a^b mu_b)
    """
    g = cs.metric
    nabla = cs.connection()
    P = cs.schouten()
    M = cs.manifold
    n = cs.dimension

    sigma = tractor.sigma
    mu = tractor.mu
    rho = tractor.rho

    # --- sigma slot: nabla_a sigma - mu_a  (1-form) ---
    d_sigma = nabla(sigma)   # 1-form
    sigma_result = d_sigma - mu

    # --- mu slot: nabla_a mu_b + P_ab sigma + g_ab rho  (0,2)-tensor ---
    d_mu = nabla(mu)         # (0,2) tensor
    mu_result = d_mu + P * sigma + g * rho

    # --- rho slot: nabla_a rho - P_a^b mu_b  (1-form) ---
    d_rho = nabla(rho)       # 1-form
    # P_a^b mu_b: raise mu to get mu^b, then contract P_{ab} mu^b
    mu_up = mu.up(g)         # vector field mu^b
    P_mu = P.contract(1, mu_up, 0)  # 1-form: P_{ab} g^{bc} mu_c = P_a^b mu_b
    rho_result = d_rho - P_mu

    return {'sigma': sigma_result, 'mu': mu_result, 'rho': rho_result}
