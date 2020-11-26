# -*- coding: utf-8 -*-

import numpy as np


class Runge_Kutta:
    """ Base class for one step Runge Kutta time integrator methods. """

    def __init__(self, u0, T, tau, function, A=None, b=None):
        """ The constructor. """
        self.u0 = u0
        self.T = T
        self.tau = tau
        self.f = function
        self.A = A
        self.b = b
        self.d = 1 if len(u0.shape) == 1 else u0.shape[0]
        self.N = u0.shape[0] if len(u0.shape) == 1 else u0.shape[1]
        self.s = int(0 if b is None else len(self.b))

    def step(self, u):
        """ Method executing time stepping. """
        if self.d == 1:
            k = np.zeros([self.N, self.s])
            for i in range(self.s):
                k[:, i] = self.f(u + self.tau * np.dot(k, self.A[i, :]))
        else:
            k = np.zeros([self.d, self.N, len(self.b)])
            for i in range(len(self.b)):
                k[:, :, i] = self.f(u + self.tau * np.dot(k, self.A[i, :]))

        return u + self.tau * np.dot(k, self.b)

    def solve(self):
        """ Method solving problem up to desired time horizon. """
        if self.d == 1:
            solution = np.zeros([self.N, int(self.T / self.tau) + 1])
            solution[:, 0] = self.u0
            for t in range(1, int(self.T / self.tau) + 1):
                solution[:, t] = self.step(solution[:, t - 1])
        else:
            solution = np.zeros([self.d, self.N, int(self.T / self.tau) + 1])
            solution[:, :, 0] = self.u0
            for t in range(1, int(self.T / self.tau) + 1):
                solution[:, :, t] = self.step(solution[:, :, t - 1])

        return solution


class Explicit_RK3(Runge_Kutta):
    """Subclass implementing the Strong Stability Preserving Runge Kutta method
    of order 3 (SSP RK3)."""

    def __init__(self, u0, T, tau, function):
        """ The constructor. """
        super().__init__(u0, T, tau, function)
        self.A = np.array([[0, 0, 0], [1, 0, 0], [1 / 4, 1 / 4, 0]])
        self.b = np.array([1 / 6, 1 / 6, 2 / 3])
        self.s = len(self.b)


class Explicit_RK5(Runge_Kutta):
    """ Subclass implementing the Runge Kutta method of order 5. """

    def __init__(self, u0, T, tau, function):
        """ The constructor. """
        super().__init__(u0, T, tau, function)
        self.A = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1 / 3, 0, 0, 0, 0, 0],
                [4 / 25, 6 / 25, 0, 0, 0, 0],
                [1 / 4, -3, 15 / 4, 0, 0, 0],
                [2 / 27, 10 / 9, -50 / 81, 8 / 81, 0, 0],
                [2 / 25, 12 / 25, 2 / 15, 8 / 75, 0, 0],
            ]
        )
        self.b = np.array([23 / 192, 0, 125 / 192, 0, -27 / 64, 125 / 192])
        self.s = len(self.b)

    def apply_alternative_solver(self):
        """ Alternative Butcher tableau for explicit RK method of order 5. """
        self.A = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1 / 4, 0, 0, 0, 0, 0],
                [1 / 8, 1 / 8, 0, 0, 0, 0],
                [0, 0, 1 / 2, 0, 0, 0],
                [3 / 16, -3 / 8, 3 / 8, 9 / 16, 0, 0],
                [-3 / 7, 8 / 7, 6 / 7, -12 / 7, 8 / 7, 0],
            ]
        )
        self.b = np.array([7 / 90, 0, 16 / 45, 2 / 15, 16 / 45, 7 / 90])
        self.s = len(self.b)


class Explicit_RK7(Runge_Kutta):
    """ Subclass implementing the Runge Kutta method of order 7. """

    def __init__(self, u0, T, tau, function):
        """ The constructor. """
        super().__init__(u0, T, tau, function)
        self.A = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1 / 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1 / 3, 0, 0, 0, 0, 0, 0, 0],
                [1 / 8, 0, 3 / 8, 0, 0, 0, 0, 0, 0],
                [148 / 1331, 0, 150 / 1331, -56 / 1331, 0, 0, 0, 0, 0],
                [-404 / 243, 0, -170 / 27, 4024 / 1701, 10648 / 1701, 0, 0, 0, 0],
                [
                    2466 / 2401,
                    0,
                    1242 / 343,
                    -19176 / 16807,
                    -51909 / 16807,
                    1053 / 2401,
                    0,
                    0,
                    0,
                ],
                [5 / 154, 0, 0, 96 / 539, -1815 / 20384, -405 / 2464, 49 / 1144, 0, 0],
                [
                    -113 / 32,
                    0,
                    -195 / 22,
                    32 / 7,
                    29403 / 3584,
                    -729 / 512,
                    1029 / 1408,
                    21 / 16,
                    0,
                ],
            ]
        )
        self.b = np.array(
            [
                0,
                0,
                0,
                32 / 105,
                1771561 / 6289920,
                243 / 2560,
                16807 / 74880,
                77 / 1440,
                11 / 270,
            ]
        )
        self.s = len(self.b)
