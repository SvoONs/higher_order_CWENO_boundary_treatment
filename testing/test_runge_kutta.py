import numpy as np
import pytest

import higher_order_CWENO_boundary_treatment.runge_kutta as rk

def test_rk_method_consistency():
    RK3 = rk.Explicit_RK3(np.array([0,0]), 1 , 0.5, _dummy_func)
    assert sum(RK3.b) == 1

def _dummy_func():
    pass