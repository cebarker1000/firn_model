"""Governing UFL forms for the firn model."""

from firedrake import FacetNormal, conditional, dx, dS_h, ds_t, ds_b
from irksome import Dt


def column_forms(
    state,
    tests,
    params,
    compaction,
    horizontal_divergence=0.0,
):
    """
    Return forms for a locally horizontally uniform firn column.

    horizontal_divergence is D = div_h(u), assumed independent of zeta.
    The reduction assumes grad_h(rho) = grad_h(h) = 0, so

        div_h(h rho u) = h rho D,
        div_h(h u)     = h D.
    """
    rho, h, p = state
    phi, eta, psi = tests

    C = compaction(rho, getattr(params, "temperature", None), params)
    fractional_compaction = C / rho
    D = horizontal_divergence

    normal = FacetNormal(rho.ufl_domain())
    zeta_axis = normal.ufl_shape[0] - 1

    normal_speed = p("+") * normal[zeta_axis]("+")
    rho_upwind = conditional(normal_speed >= 0.0, rho("+"), rho("-"))

    time_forms = {
        "rho": Dt(h * rho) * phi * dx,
        "h": Dt(h) * eta * dx,
        "p": Dt(h) * psi * dx,
    }

    spatial_forms = {
        "rho": (
            h * rho * D * phi * dx
            - rho * p * phi.dx(zeta_axis) * dx
            + (phi("+") - phi("-")) * normal_speed * rho_upwind * dS_h
            + params.rho_s * p * phi * ds_t
            - rho * p * phi * ds_b
        ),
        "h": (
            (-params.a_s + params.a_b + h * D + h * fractional_compaction)
            * eta * dx
        ),
        "p": (
            (
                h * D
                + p.dx(zeta_axis)
                + h * fractional_compaction
            )
            * psi * dx
        ),
    }

    return time_forms, spatial_forms


def vertical_forms(state, tests, params, compaction, measures=None):
    """Zero-horizontal-divergence wrapper."""
    return column_forms(
        state,
        tests,
        params,
        compaction,
        horizontal_divergence=0.0,
    )