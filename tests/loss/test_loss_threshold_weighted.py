"""
Contains unit tests for scores.loss.threshold_weighted
"""

import pytest

try:
    import torch

    _SKIP_TORCH_TESTS = False
except ModuleNotFoundError:
    _SKIP_TORCH_TESTS = True

from scores.continuous.threshold_weighted_impl import (
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
