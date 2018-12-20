# -*- coding: utf-8 -*-

import numpy as np

from higher_order_CWENO_boundary_treatment.uniform_CWENO_1D import (
    CWENO3_1D,
    CWENO5_1D,
    CWENO7_1D,
)


def test_operator_initialization():

    dummy_avgs = np.array([0, 0])
    dummy_h = 0.1
    dummy_eps = dummy_h ** 2
    dummy_p = 2
    dummy_d0 = 0.5

    cweno3 = CWENO3_1D(dummy_avgs, dummy_h, dummy_eps, dummy_p, dummy_d0)
    assert 2 == cweno3.G
    assert 1 == cweno3.g

    cweno5 = CWENO5_1D(dummy_avgs, dummy_h, dummy_eps, dummy_p, dummy_d0)
    assert 4 == cweno5.G
    assert 2 == cweno5.g

    cweno7 = CWENO7_1D(dummy_avgs, dummy_h, dummy_eps, dummy_p, dummy_d0)
    assert 6 == cweno7.G
    assert 3 == cweno7.g
