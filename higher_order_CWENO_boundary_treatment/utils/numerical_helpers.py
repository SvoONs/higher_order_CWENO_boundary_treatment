# -*- coding: utf-8 -*-

import numpy as np
from scipy.special.orthogonal import p_roots


def convergence_rate(error, refinement):
    """ Computes the experimental convergence rate provided errors and cell sizes. """

    if len(error) != len(refinement):
        raise ValueError("The dimensions of data provided does not match.")

    convergence_rate = [float("inf")]
    for i in range(len(error) - 1):
        convergence_rate.append(
            np.log2(error[i + 1] / error[i])
            / np.log2(refinement[i + 1] / refinement[i])
        )

    return convergence_rate


def undivided_differences(data, order):
    """ Computes the undivided differences up to a provided order. """

    undiv_diffs = np.zeros((data.shape[0], order))
    undiv_diffs[:, 0] = data
    undiv_diffs[:-1, 1] = (undiv_diffs[1:, 0] - undiv_diffs[:-1, 0]) / 2
    for i in range(2, order):
        undiv_diffs[:-i, i] = (
            undiv_diffs[1 : -(i - 1), i - 1] - undiv_diffs[:-i, i - 1]
        ) / (i + 1)

    return undiv_diffs


def jacobian(func, x0):
    """ Computes the numerical Jacobian matrix of the function 'func' at x0. """

    f0 = func(x0)
    n = len(x0)
    m = len(f0)
    J = np.empty([n, m])
    dx = np.zeros(n)
    epsilon = np.sqrt(np.finfo(float).eps)
    for i in range(n):
        dx[i] = epsilon
        J[i] = (func(x0 + dx) - f0) / epsilon
        dx[i] = 0.0

    return J.transpose()


def trapezoidal_rule(func, x0, x1):
    """ Numerical quadrature using the trapezoidal rule. """

    return (x1 - x0) / 2 * (func(x0) + func(1))


def simpson_rule(func, x0, x2):
    """ Numerical quadrature using Simpson's rule. """

    h = x2[0] - x0[0] / 2

    return h / 3 * (func(x0) + 4 * func(x0 + h) + func(x2))


def gaussian_quadrature(func, n, a, b):
    """ Integrates a function using the Gaussian quadratures of n nodes 
    with accuracy O((b-a)^2*n). """

    x, w = p_roots(n + 1)
    G = 0.5 * (b - a) * sum(w * func(0.5 * (b - a) * x + 0.5 * (b + a)))

    return G
