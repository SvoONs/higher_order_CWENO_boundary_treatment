# -*- coding: utf-8 -*-

from numpy import array
import pytest

from higher_order_CWENO_boundary_treatment.uniform_CWENO_1D import (
    CWENO3_1D,
    CWENO5_1D,
    CWENO7_1D,
)


@pytest.fixture
def data():
    return {
        "CWENO3": array([0, 1, 0]),
        "CWENO5": array([1, 0, 1, 0, 1]),
        "CWENO7": array([0, 1, 0, 1, 0, 1, 0]),
    }


@pytest.fixture
def params():
    return {"h": 0.1, "eps": 0.01, "p": 2, "d0": 0.5}


@pytest.fixture
def cweno3(data, params):
    return CWENO3_1D(
        data["CWENO3"], params["h"], params["eps"], params["p"], params["d0"]
    )


@pytest.fixture
def cweno5(data, params):
    return CWENO5_1D(
        data["CWENO5"], params["h"], params["eps"], params["p"], params["d0"]
    )


@pytest.fixture
def cweno7(data, params):
    return CWENO7_1D(
        data["CWENO7"], params["h"], params["eps"], params["p"], params["d0"]
    )
