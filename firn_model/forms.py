"""Governing UFL forms for the firn model."""

from numbers import Real

from firedrake import (
    Constant,
    FacetNormal,
    conditional,
    dx,
    dS_h,
    ds_t,
    ds_b,
)
from irksome import Dt


def _coefficient(value):
    """Convert ordinary numerical parameters to Firedrake Constants."""
    if isinstance(value, Real):
        return Constant(float(value))
    return value


def column_forms(
    state,
    tests,
    params,
    compaction,
    horizontal_divergence=None,
):
    """
    Return forms for a locally horizontally uniform firn column.

    State:
        (rho, h, p)
    or
        (rho, h, p, T)

    Here p = h * omega and D = div_h(u), with D assumed independent
    of zeta. In the thermal model, c_p must be constant and params
    must contain c_p and k. The temperature equation is written in
    conservative form:

        c_p Dt(h rho T)
        + h rho c_p T D
        + c_p d(rho p T)/dzeta
        - (1/h) d(k dT/dzeta)/dzeta = 0.

    Using the mass equation, this is equivalent to the material form

        h rho c_p Dt(T)
        + rho c_p p dT/dzeta
        - (1/h) d(k dT/dzeta)/dzeta = 0.

    params.q_b is optional and is positive upward into the firn.
    Surface temperature is normally imposed with a DirichletBC.
    """
    if len(state) == 3:
        rho, h, p = state
        phi, eta, psi = tests
        T = getattr(params, "temperature", None)
        thermal = False

    elif len(state) == 4:
        rho, h, p, T = state
        phi, eta, psi, chi = tests
        thermal = True

    else:
        raise ValueError(
            "state must be (rho, h, p) or (rho, h, p, T)"
        )

    # Use UFL coefficients rather than literal Python zeros. This keeps
    # the integration domain attached even when a term is numerically zero.
    if horizontal_divergence is None:
        D = Constant(0.0)
    else:
        D = _coefficient(horizontal_divergence)

    C = compaction(rho, T, params)
    fractional_compaction = C / rho

    normal = FacetNormal(rho.ufl_domain())
    zeta_axis = normal.ufl_shape[0] - 1

    normal_speed = p("+") * normal[zeta_axis]("+")

    rho_upwind = conditional(
        normal_speed >= 0.0,
        rho("+"),
        rho("-"),
    )

    time_forms = {
        "rho": Dt(h * rho) * phi * dx,
        "h": Dt(h) * eta * dx,
        "p": Dt(h) * psi * dx,
    }

    spatial_forms = {
        "rho": (
            h * rho * D * phi * dx
            - rho * p * phi.dx(zeta_axis) * dx
            + (phi("+") - phi("-"))
            * normal_speed
            * rho_upwind
            * dS_h
            + params.rho_s * p * phi * ds_t
            - rho * p * phi * ds_b
        ),

        "h": (
            (
                -params.a_s
                + params.a_b
                + h * D
                + h * fractional_compaction
            )
            * eta
            * dx
        ),

        "p": (
            (
                h * D
                + p.dx(zeta_axis)
                + h * fractional_compaction
            )
            * psi
            * dx
        ),
    }

    if thermal:
        c_p = _coefficient(params.c_p)
        k = _coefficient(params.k)
        q_b = _coefficient(getattr(params, "q_b", 0.0))

        T_upwind = conditional(
            normal_speed >= 0.0,
            T("+"),
            T("-"),
        )

        # If a separate prescribed inflow temperature is supplied, use it
        # at the surface. Otherwise the strongly imposed trace of T is used.
        T_s = _coefficient(getattr(params, "T_s", T))

        time_forms["temperature"] = (
            c_p * Dt(h * rho * T) * chi * dx
        )

        spatial_forms["temperature"] = (
            # Horizontal conservative energy-flux divergence.
            h * rho * c_p * T * D * chi * dx

            # Vertical conservative energy flux.
            - c_p
            * rho
            * p
            * T
            * chi.dx(zeta_axis)
            * dx

            # Interior upwind energy flux:
            # upwind mass flux times upwind specific thermal energy.
            + c_p
            * (chi("+") - chi("-"))
            * normal_speed
            * rho_upwind
            * T_upwind
            * dS_h

            # Surface inflow and basal outflow.
            + c_p
            * params.rho_s
            * p
            * T_s
            * chi
            * ds_t

            - c_p
            * rho
            * p
            * T
            * chi
            * ds_b

            # Vertical conduction.
            + (k / h)
            * T.dx(zeta_axis)
            * chi.dx(zeta_axis)
            * dx

            # Basal conductive heat flux, positive upward into the firn.
            - q_b * chi * ds_b
        )

    return time_forms, spatial_forms


def vertical_forms(
    state,
    tests,
    params,
    compaction,
    measures=None,
):
    """Zero-horizontal-divergence wrapper."""
    return column_forms(
        state,
        tests,
        params,
        compaction,
        horizontal_divergence=Constant(0.0),
    )