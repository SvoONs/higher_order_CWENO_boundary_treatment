# -*- coding: utf-8 -*-

import numpy as np

from higher_order_CWENO_boundary_treatment.uniform_CWENO_1D import (
    CWENO3_1D,
    CWENO5_1D,
    CWENO7_1D,
)
from higher_order_CWENO_boundary_treatment.uniform_CWENO_2D import (
    CWENO3_2D,
    CWENO5_2D,
    CWENO7_2D,
)


def right_hand_side(data, h, flux, dflux, params, order):
    """Method implementing the RHS of the semi discretized system of ODEs
    where periodic boundaries are assumed."""

    # CWENO reconstruction
    left, right = reconstructed_interfaces(data, h, params, order)

    # periodic boundaries
    left = np.append(left, left[0])
    right = np.insert(right, 0, right[-1])

    # right border numerical fluxes:
    H_plus = local_lax_friedrichs_flux(left[1:], right[1:], flux, dflux)
    # left border numerical fluxes:
    H_minus = local_lax_friedrichs_flux(left[:-1], right[:-1], flux, dflux)

    return -(H_plus - H_minus) / h


def reconstructed_interfaces(data, h, params, order):
    """Method calling the CWENO operators of corresponding order for
    reconstruction of cell interfaces. The feasible orders are 3, 5 or 7."""

    if order == 3:
        CWENO = CWENO3_1D(data, h, params[0], params[1], params[2], params[-1])
    elif order == 5:
        CWENO = CWENO5_1D(data, h, params[0], params[1], params[2], params[-1])
    elif order == 7:
        CWENO = CWENO7_1D(data, h, params[0], params[1], params[2], params[-1])

    left, right = CWENO.reconstruct_cell_interfaces()

    return left, right


def right_hand_side_2D(data, hx, hy, flux_f, dflux_f, flux_g, dflux_g, params, order):
    """Method implementing the RHS of the semi discretized system of ODEs in
    two space dimensions where periodic boundaries are assumed."""

    # CWENO reconstruction
    left, right, down, up = reconstructed_interfaces_2D(data, hx, hy, params, order)

    # periodic boundaries
    left, right = np.c_[left, left[:, 0]], np.c_[right[:, -1], right]
    down, up = np.r_["0,2", down, down[0, :]], np.r_["0,2", up[-1, :], up]

    # right border numerical fluxes:
    H_right = local_lax_friedrichs_flux(left[:, 1:], right[:, 1:], flux_f, dflux_f)
    # left border numerical fluxes:
    H_left = local_lax_friedrichs_flux(left[:, :-1], right[:, :-1], flux_f, dflux_f)
    # upper border numerical fluxes:
    H_up = local_lax_friedrichs_flux(down[1:, :], up[1:, :], flux_g, dflux_g)
    # lower border numerical fluxes:
    H_down = local_lax_friedrichs_flux(down[:-1, :], up[:-1, :], flux_g, dflux_g)

    return -(H_right - H_left) / hx - (H_up - H_down) / hy


def reconstructed_interfaces_2D(data, hx, hy, params, order):
    """Method calling the CWENO operators of corresponding order for
    reconstruction of cell interfaces in two space dimensions.
    The feasible orders are 3, 5 or 7."""

    if order == 3:
        CWENO_2D = CWENO3_2D(data, hx, hy, params[0], params[1], params[2], params[-1])
    elif order == 5:
        CWENO_2D = CWENO5_2D(data, hx, hy, params[0], params[1], params[2], params[-1])
    elif order == 7:
        CWENO_2D = CWENO7_2D(data, hx, hy, params[0], params[1], params[2], params[-1])

    left, right, down, up = CWENO_2D.reconstruct_cell_interfaces()

    return left, right, down, up


def local_lax_friedrichs_flux(u, v, func, spec):
    """Method implementing the local Lax Friedrichs numerical flux function for the
    linear degenerate or genuinely nonlinear case."""

    sigma = np.maximum(np.abs(spec(u)), np.abs(spec(v)))

    return 1 / 2 * (func(u) + func(v) - sigma * (u - v))
