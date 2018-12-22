# -*- coding: utf-8 -*-

import numpy as np
import pytest

from higher_order_CWENO_boundary_treatment.uniform_CWENO_1D import (
    CWENO3_1D,
    CWENO5_1D,
    CWENO7_1D,
)


@pytest.fixture
def data_params():
    return {
        "CWENO3": np.array([0, 1, 0]),
        "CWENO5": np.array([1, 0, 1, 0, 1]),
        "CWENO7": np.array([0, 1, 0, 1, 0, 1, 0]),
        "h": 0.1,
        "eps": 0.01,
        "p": 2,
        "d0": 0.5,
    }


@pytest.fixture
def cweno3(data_params):
    return CWENO3_1D(
        data_params["CWENO3"],
        data_params["h"],
        data_params["eps"],
        data_params["p"],
        data_params["d0"],
    )


@pytest.fixture
def cweno5(data_params):
    return CWENO5_1D(
        data_params["CWENO5"],
        data_params["h"],
        data_params["eps"],
        data_params["p"],
        data_params["d0"],
    )


@pytest.fixture
def cweno7(data_params):
    return CWENO7_1D(
        data_params["CWENO7"],
        data_params["h"],
        data_params["eps"],
        data_params["p"],
        data_params["d0"],
    )


def test_operator_initialization(cweno3, cweno5, cweno7):

    assert 2 == cweno3.G
    assert 1 == cweno3.g

    assert 4 == cweno5.G
    assert 2 == cweno5.g

    assert 6 == cweno7.G
    assert 3 == cweno7.g


def test_operator_result_fine(cweno3, cweno5, cweno7):

    p_rec_cweno3 = cweno3.compute_reconstruction_polynomial()
    p_rec_cweno5 = cweno5.compute_reconstruction_polynomial()
    p_rec_cweno7 = cweno7.compute_reconstruction_polynomial()

    expected_p_rec_cweno3 = [1.00056332e00, 3.75247204e-18, -6.75985268e-01]
    assert [
        np.isclose(expected_coeff, actual_coeff)
        for expected_coeff in expected_p_rec_cweno3
        for actual_coeff in p_rec_cweno3
    ]

    expected_p_rec_cweno5 = [
        1.04785750e00,
        2.76475774e-16,
        -5.74321099e01,
        1.72797088e-17,
        2.07522374e00,
    ]
    assert [
        np.isclose(expected_coeff, actual_coeff)
        for expected_coeff in expected_p_rec_cweno5
        for actual_coeff in p_rec_cweno5
    ]

    expected_p_rec_cweno7 = [
        1.06630815e00,
        -5.55224086e-16,
        -7.95702389e01,
        -3.46451877e-15,
        3.07275704e-01,
        -8.72244617e-18,
        -2.23473239e00,
    ]
    assert [
        np.isclose(expected_coeff, actual_coeff)
        for expected_coeff in expected_p_rec_cweno7
        for actual_coeff in p_rec_cweno7
    ]
