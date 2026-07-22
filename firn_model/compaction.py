"""Constitutive laws for firn compaction."""


def linear_compaction(rho, temperature, params):
    """Return C = c a_s (rho_i - rho)."""
    return params.c * params.a_s * (params.rho_i - rho)