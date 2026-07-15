"""
Contains unit tests for scores.loss.threshold_weighted
"""

import numpy as np
import pytest

try:
    import torch

    _SKIP_TORCH_TESTS = False
except ModuleNotFoundError:
    _SKIP_TORCH_TESTS = True

from scores.continuous.threshold_weighted_impl import (
    _auxiliary_funcs,
    _g_j_rect,
    _g_j_trap,
    _phi_j_prime_rect,
    _phi_j_prime_trap,
    _phi_j_rect,
    _phi_j_trap,
)
from tests.continuous.test_threshold_weighted import (
    DA_A,
    DA_A_TRAP,
    DA_B,
    DA_B_TRAP,
    DA_C_TRAP,
    DA_D_TRAP,
    DA_X1,
    DA_X2,
    DA_X3,
    DA_X_TRAP,
)

TEST_DEVICE_PARAMS = [
    pytest.param(
        device,
        marks=pytest.mark.skipif(
            not getattr(torch, device).is_available(), reason=f"torch device {device} not available."
        ),
    )
    for device in ("cpu", "cuda", "mps")
]


@pytest.fixture(params=[(-1.0, 2.0, DA_X1), (DA_A, DA_B, DA_X2)])
def rect_aux_func_input_args(request):
    "Fixture representing a, b intervals; and input arrays for functions"
    return request.param


@pytest.fixture(params=[_g_j_rect, _phi_j_rect, _phi_j_prime_rect])
def rect_aux_func(request):
    "Fixture representing functions to be tested"
    return request.param


@pytest.fixture
def rect_aux_func_test_args(rect_aux_func_input_args, rect_aux_func):
    "Fixture aggregating inputs and calculating expected value"
    a, b, x = rect_aux_func_input_args
    return a, b, x, rect_aux_func(a, b, x), rect_aux_func


@pytest.mark.skipif(_SKIP_TORCH_TESTS, reason="torch not installed")
@pytest.mark.parametrize(("device",), TEST_DEVICE_PARAMS)
def test_rect_aux_funcs(rect_aux_func_test_args, device):
    """Tests that `_g_j_rect`, `_phi_j_rect`, and `_phi_j_prime_rect` gives results as expected."""
    a, b, x, expected = tuple(
        x if isinstance(x, float) else torch.tensor(x.values, dtype=torch.float, device=device)
        for x in rect_aux_func_test_args[:-1]
    )
    func = rect_aux_func_test_args[-1]
    result = func(a, b, x)
    assert isinstance(result, torch.Tensor)
    assert result.device == expected.device
    torch.testing.assert_close(result, expected, equal_nan=True)


@pytest.fixture(params=[(-2, 1, 5, 8, DA_X3), (DA_A_TRAP, DA_B_TRAP, DA_C_TRAP, DA_D_TRAP, DA_X_TRAP)])
def trap_aux_func_input_args(request):
    return request.param


@pytest.fixture(params=[_g_j_trap, _phi_j_trap, _phi_j_prime_trap])
def trap_aux_func(request):
    return request.param


@pytest.fixture
def aux_func_trap_test_args(trap_aux_func_input_args, trap_aux_func):
    a, b, c, d, x = trap_aux_func_input_args
    return a, b, c, d, x, trap_aux_func(a, b, c, d, x), trap_aux_func


@pytest.mark.skipif(_SKIP_TORCH_TESTS, reason="torch not installed")
@pytest.mark.parametrize(("device",), TEST_DEVICE_PARAMS)
def test_aux_trap_funcs(aux_func_trap_test_args, device):
    """Tests that `_g_j_trap`, `_phi_j_trap`, and `_phi_j_prime_trap` gives results as expected."""
    a, b, c, d, x, expected = tuple(
        x if isinstance(x, (float, int)) else torch.tensor(x.values, dtype=torch.float, device=device)
        for x in aux_func_trap_test_args[:-1]
    )
    func = aux_func_trap_test_args[-1]
    result = func(a, b, c, d, x)
    assert isinstance(result, torch.Tensor)
    assert result.device == expected.device
    torch.testing.assert_close(result, expected, equal_nan=True)


@pytest.mark.skipif(_SKIP_TORCH_TESTS, reason="torch not installed")
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


@pytest.mark.skipif(_SKIP_TORCH_TESTS, reason="torch not installed")
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
