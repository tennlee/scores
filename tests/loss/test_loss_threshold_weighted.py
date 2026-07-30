"""
Contains unit tests for scores.loss.threshold_weighted
"""

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("torch not installed")

from scores.continuous.threshold_weighted_impl import (
    _auxiliary_funcs,
    _g_j_rect,
    _g_j_trap,
    _phi_j_prime_rect,
    _phi_j_prime_trap,
    _phi_j_rect,
    _phi_j_trap,
    tw_huber_loss,
)
from tests.continuous.continuous_test_data import (
    TW_A,
    TW_A_TRAP,
    TW_B,
    TW_B_TRAP,
    TW_C_TRAP,
    TW_D_TRAP,
    TW_FCST1_DA,
    TW_HUBER_PARAM,
    TW_OBS1_DA,
    TW_X1,
    TW_X2,
    TW_X3,
    TW_X_TRAP,
)
from tests.loss.loss_test_utils import TEST_DEVICE_PARAMS


@pytest.fixture(params=[(-1.0, 2.0, TW_X1), (TW_A, TW_B, TW_X2)])
def rect_aux_func_input_args(request):
    "Fixture for a, b intervals; and input arrays for rectangular auxiliary functions"
    return request.param


@pytest.fixture(params=[_g_j_rect, _phi_j_rect, _phi_j_prime_rect])
def rect_aux_func(request):
    "Fixture for rectangular auxiliary functions to be tested"
    return request.param


@pytest.fixture
def rect_aux_func_test_args(rect_aux_func_input_args, rect_aux_func):
    "Fixture aggregating inputs and calculating expected value for rectangular auxiliary functions"
    a, b, x = rect_aux_func_input_args
    return a, b, x, rect_aux_func(a, b, x), rect_aux_func


@pytest.mark.parametrize(("device",), TEST_DEVICE_PARAMS)
def test_rect_aux_funcs(rect_aux_func_test_args, device):
    """Tests that rectangular auxiliary functions gives results as expected."""
    a, b, x, expected = tuple(
        x if isinstance(x, float) else torch.tensor(x.values, dtype=torch.float, device=device)
        for x in rect_aux_func_test_args[:-1]
    )
    func = rect_aux_func_test_args[-1]
    result = func(a, b, x)
    assert isinstance(result, torch.Tensor)
    assert result.device == expected.device
    torch.testing.assert_close(result, expected, equal_nan=True)


@pytest.fixture(params=[(-2, 1, 5, 8, TW_X3), (TW_A_TRAP, TW_B_TRAP, TW_C_TRAP, TW_D_TRAP, TW_X_TRAP)])
def trap_aux_func_input_args(request):
    "Fixture for a, b, c, d intervals; and input arrays for trapezoidal auxiliary functions"
    return request.param


@pytest.fixture(params=[_g_j_trap, _phi_j_trap, _phi_j_prime_trap])
def trap_aux_func(request):
    "Fixture for trapezoidal auxiliary functions to be tested"
    return request.param


@pytest.fixture
def aux_func_trap_test_args(trap_aux_func_input_args, trap_aux_func):
    "Fixture aggregating inputs and calculating expected value for trapezoidal auxiliary functions"
    a, b, c, d, x = trap_aux_func_input_args
    return a, b, c, d, x, trap_aux_func(a, b, c, d, x), trap_aux_func


@pytest.mark.parametrize(("device",), TEST_DEVICE_PARAMS)
def test_aux_trap_funcs(aux_func_trap_test_args, device):
    """Tests that trapezoidal auxiliary functions gives results as expected."""
    a, b, c, d, x, expected = tuple(
        x if isinstance(x, (float, int)) else torch.tensor(x.values, dtype=torch.float, device=device)
        for x in aux_func_trap_test_args[:-1]
    )
    func = aux_func_trap_test_args[-1]
    result = func(a, b, c, d, x)
    assert isinstance(result, torch.Tensor)
    assert result.device == expected.device
    torch.testing.assert_close(result, expected, equal_nan=True)


@pytest.mark.parametrize(("device",), TEST_DEVICE_PARAMS)
@pytest.mark.parametrize(
    ("interval_where_one", "a", "b"),
    [
        ((1, 4), 1, 4),
        ((-np.inf, 5), -6, 5),
        ((-6, np.inf), -6, 11),
        # ((DA_A_INF, DA_B_INF), DA_A_FINITE, DA_B_FINITE),
        # ((DA_INTERVAL_WHERE, np.inf), DA_INTERVAL_WHERE, 11),
        # ((-np.inf, DA_INTERVAL_WHERE), -6, DA_INTERVAL_WHERE),
    ],
)
def test__auxiliary_funcs1(interval_where_one, a, b, device):
    """
    Tests that `_auxiliary_funcs` gives expected results for "rectangular" weights.
    """
    interval = tuple(
        x if isinstance(x, (int, float)) else torch.tensor(x.values, dtype=torch.float, device=device)
        for x in interval_where_one
    )
    a_ = a if isinstance(a, (int, float)) else torch.tensor(a.values, dtype=torch.float, device=device)
    b_ = b if isinstance(b, (int, float)) else torch.tensor(b.values, dtype=torch.float, device=device)
    g, phi, phi_prime = _auxiliary_funcs(
        torch.tensor([-5, 4], dtype=torch.float, device=device),
        torch.tensor([0, 10], dtype=torch.float, device=device),
        interval,
        None,
    )

    x = torch.linspace(-10, 12, 100, dtype=torch.float, device=device)
    torch.testing.assert_close(g(x), _g_j_rect(a_, b_, x), equal_nan=True)
    torch.testing.assert_close(phi(x), _phi_j_rect(a_, b_, x), equal_nan=True)
    torch.testing.assert_close(phi_prime(x), _phi_j_prime_rect(a_, b_, x), equal_nan=True)


@pytest.mark.parametrize(("device",), TEST_DEVICE_PARAMS)
@pytest.mark.parametrize(
    ("interval_where_one", "interval_where_positive", "a", "b", "c", "d"),
    [
        ((1, 4), (-1, 5), -1, 1, 4, 5),
        ((-np.inf, 4), (-np.inf, 5), -7, -6, 4, 5),
        ((-1, np.inf), (-3, np.inf), -3, -1, 11, 12),
    ],
)
# pylint: disable=too-many-positional-arguments
def test__auxiliary_funcs2(interval_where_one, interval_where_positive, a, b, c, d, device):
    """
    Tests that `_auxiliary_funcs` gives expected results for "trapezoidal" weights.
    """
    g, phi, phi_prime = _auxiliary_funcs(
        torch.tensor([0, 10], dtype=torch.float, device=device),
        torch.tensor([-5, 9], dtype=torch.float, device=device),
        interval_where_one,
        interval_where_positive,
    )
    x = torch.linspace(-10, 12, 50, dtype=torch.float, device=device)
    torch.testing.assert_close(g(x), _g_j_trap(a, b, c, d, x), equal_nan=True)
    torch.testing.assert_close(phi(x), _phi_j_trap(a, b, c, d, x), equal_nan=True)
    torch.testing.assert_close(phi_prime(x), _phi_j_prime_trap(a, b, c, d, x), equal_nan=True)


@pytest.fixture(params=[(tw_huber_loss, {"huber_param": TW_HUBER_PARAM})])
def scoring_func_args(request):
    """Fixture representing the scoring functions to test."""
    return request.param


@pytest.fixture(
    params=[
        ((-np.inf, np.inf), None),
        ((-np.inf, 0), None),
        ((0, np.inf), None),
    ]
)
def intervals(request):
    """Fixture representing interval_where_one and interval_where_positive args in scoring functions."""
    return request.param


@pytest.mark.parametrize(("device"), TEST_DEVICE_PARAMS)
def test_threshold_weighted_scores(scoring_func_args, intervals, device):
    """
    Tests that the scoring functions with torch inputs replicate xarray inputs.
    Replicates tests.continuous.threshold_weighted
    """
    scoring_func, kwargs = scoring_func_args
    interval_where_one, interval_where_positive = intervals

    # DA_FCST and TW_OBS1_DA have different, but overlapping dims
    # this extracts the dims that overlap with fcst, since torch
    # doesn't do dim matching.
    da_obs = TW_OBS1_DA[:-1]

    expected = scoring_func(
        TW_FCST1_DA,
        da_obs,
        interval_where_one=interval_where_one,
        interval_where_positive=interval_where_positive,
        **kwargs,
    )

    fcst = torch.tensor(TW_FCST1_DA.values, dtype=torch.float, device=device)
    obs = torch.tensor(da_obs.values, dtype=torch.float, device=device)

    result = scoring_func(
        fcst,
        obs,
        interval_where_one=interval_where_one,
        interval_where_positive=interval_where_positive,
        **kwargs,
    )

    assert isinstance(result, torch.Tensor)
    assert result.device == fcst.device
    torch.testing.assert_close(result, torch.tensor(expected.values, dtype=result.dtype, device=result.device))
