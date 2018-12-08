import numpy as np
import pytest

import higher_order_CWENO_boundary_treatment.runge_kutta as rk

def test_rk_method_consistency():
    
    dummy_initial_data = np.array([0, 0])
    dummy_T = 1
    dummy_tau = 0.5
    def dummy_func():
        pass

    RK3 = rk.Explicit_RK3(dummy_initial_data, dummy_T, dummy_tau, dummy_func)
    assert np.isclose(sum(RK3.b), 1)

    RK5 = rk.Explicit_RK5(dummy_initial_data, dummy_T, dummy_tau, dummy_func)
    assert np.isclose(sum(RK5.b), 1)

    RK7 = rk.Explicit_RK7(dummy_initial_data, dummy_T, dummy_tau, dummy_func)
    assert np.isclose(sum(RK7.b), 1)
