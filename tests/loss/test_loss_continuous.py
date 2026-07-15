"""
Contains unit tests for scores.continuous.standard
"""

# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import math

import numpy as np
import pandas as pd
import pytest

try:
    import torch

    _SKIP_TORCH_TESTS = False
except ModuleNotFoundError:
    _SKIP_TORCH_TESTS = True

import scores
from tests.continuous.continuous_test_data import (
    BIAS_FCST_DA,
    BIAS_OBS_DA,
    BIAS_WEIGHTS_DA,
    MSE_FCST_DA,
    MSE_OBS_DA,
    MSE_WEIGHTS_DA,
)
from tests.loss.loss_test_utils import TEST_DEVICE_PARAMS

PRECISION = 4

# Mean Squared Error
#


@pytest.fixture(params=[True, False])
def is_angular(request):
    "Fixture for whether the MSE calculation is angular"
    return request.param


@pytest.fixture(params=[None, MSE_WEIGHTS_DA])
def weights(request):
    "Fixture for MSE weights, including the unweighted case"
    return request.param


@pytest.fixture
def mse_test_args(is_angular, weights):
    "Fixture for inputs and calculating expected MSE value"
    scale = 180.0 / math.pi if is_angular else 1.0
    fcst = scale * MSE_FCST_DA
    obs = scale * MSE_OBS_DA
    args = {
        "fcst": fcst.values,
        "obs": obs.values,
        "is_angular": is_angular,
        "weights": weights if weights is None else weights.values,
        "expected": float(
            scores.continuous.mse(
                fcst,
                obs,
                is_angular=is_angular,
                weights=weights,
            )
        ),
    }
    return args


@pytest.mark.skipif(_SKIP_TORCH_TESTS, reason="torch not installed")
@pytest.mark.parametrize(("device",), TEST_DEVICE_PARAMS)
def test_mse_torch_host_device_consistency(mse_test_args, device):
    """
    Tests execution on available devices match host
    serial answers within tolerance.
    """

    fcst = mse_test_args["fcst"]
    obs = mse_test_args["obs"]
    expected = mse_test_args["expected"]
    is_angular = mse_test_args["is_angular"]
    weights = mse_test_args["weights"]
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float, device=device)

    fcst_tensor = torch.tensor(fcst, dtype=torch.float, device=device)
    obs_tensor = torch.tensor(obs, dtype=torch.float, device=device)
    device_result = scores.loss.mse(
        fcst_tensor,
        obs_tensor,
        is_angular=is_angular,
        weights=weights,
    )
    assert isinstance(device_result, torch.Tensor)
    assert device_result.device == fcst_tensor.device
    torch.testing.assert_close(
        device_result,
        torch.tensor(expected, device=device),
    )


@pytest.mark.parametrize(
    ("arr_type",),
    [(np.array,), (pd.Series,)],
)
def test_mse_array_consistency(arr_type, mse_test_args):
    """
    Tests that different numpy-like arrays give the same result.
    """
    fcst = arr_type(mse_test_args["fcst"])
    obs = arr_type(mse_test_args["obs"])
    is_angular = mse_test_args["is_angular"]
    weights = mse_test_args["weights"]
    if weights is not None:
        weights = arr_type(weights)

    if arr_type is pd.Series:
        if weights is not None:
            pytest.skip("weighted mse not yet supported for pd.Series input")
        elif is_angular:
            pytest.skip("angular mse not yet supported for pd.Series input")

    result = scores.loss.mse(fcst, obs, is_angular=is_angular, weights=weights)
    np.testing.assert_allclose(mse_test_args["expected"], result)


@pytest.fixture(params=[None, BIAS_WEIGHTS_DA])
def additive_bias_expected(request):
    "Fixture aggregating weights and calculating expected additive bias value"
    weights = request.param
    return weights, float(
        scores.continuous.additive_bias(
            BIAS_FCST_DA,
            BIAS_OBS_DA,
            weights=weights,
        ).data
    )


@pytest.mark.skipif(_SKIP_TORCH_TESTS, reason="torch not installed")
@pytest.mark.parametrize(("device",), TEST_DEVICE_PARAMS)
def test_additive_bias_torch_host_device_consistency(additive_bias_expected, device):
    """
    Tests loss.additive_bias that execution on available devices match host
    serial answers within tolerance.
    """
    weights, expected = additive_bias_expected
    if weights is not None:
        weights = torch.tensor(weights.values, dtype=torch.float, device=device)
    fcst, obs = tuple(
        torch.tensor(obj, dtype=torch.float, device=device) for obj in (BIAS_FCST_DA.values, BIAS_OBS_DA.values)
    )
    result = scores.loss.additive_bias(fcst, obs, weights=weights)
    assert isinstance(result, torch.Tensor)
    assert result.device == fcst.device
    torch.testing.assert_close(result, torch.tensor(expected, device=device))


# def test_mse_dataframe():
#     """
#     Test calculation works correctly on dataframe columns
#     """

#     fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
#     obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
#     df = pd.DataFrame({"fcst": fcst_pd_series, "obs": obs_pd_series})
#     expected = 1.0909
#     result = scores.continuous.mse(df["fcst"], df["obs"])
#     assert isinstance(result, float)
#     assert round(result, PRECISION) == expected


# # Root Mean Squared Error


# @pytest.fixture
# def rmse_fcst_pandas():
#     """Creates forecast Pandas series for test."""
#     return pd.Series([-1, 3, 1, 3, 0, 2, 2, 1, 1, 2, 3])


# @pytest.fixture
# def rmse_fcst_nan_pandas():
#     """Creates forecast Pandas series containing NaNs for test."""
#     return pd.Series([-1, 3, 1, 3, np.nan, 2, 2, 1, 1, 2, 3])


# @pytest.fixture
# def rmse_obs_pandas():
#     """Creates observation Pandas series for test."""
#     return pd.Series([1, 1, 1, 2, 1, 2, 1, 1, -1, 3, 1])


# @pytest.mark.parametrize(
#     "forecast, observations, expected, request_kwargs",
#     [
#         ("rmse_fcst_pandas", "rmse_obs_pandas", 1.3484, {}),
#         ("rmse_fcst_pandas", 1, 1.3484, {}),
#         ("rmse_fcst_nan_pandas", "rmse_obs_pandas", 1.3784, {}),
#     ],
#     ids=[
#         "pandas-series-1d",
#         "pandas-to-point",
#         "pandas-series-nan-1d",
#     ],
# )
# def test_rmse_pandas_1d(forecast, observations, expected, request_kwargs, request):
#     """
#     Test RMSE for the following cases:
#        * Calculates the correct value for a simple pandas 1d series
#     """
#     if isinstance(forecast, str):
#         forecast = request.getfixturevalue(forecast)
#     if isinstance(observations, str):
#         observations = request.getfixturevalue(observations)
#     result = scores.continuous.rmse(forecast, observations, **request_kwargs)
#     if not isinstance(result, float):
#         assert (result.round(PRECISION) == expected).all()
#     else:
#         assert np.round(result, PRECISION) == expected


# # Mean Absolute Error


# def test_mae_pandas_series():
#     """
#     Test calculation works correctly on pandas series
#     """

#     fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
#     obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
#     expected = 0.7273
#     result = scores.continuous.mae(fcst_pd_series, obs_pd_series)
#     assert isinstance(result, float)
#     assert round(result, 4) == expected


# def test_mae_dataframe():
#     """
#     Test calculation works correctly on dataframe columns
#     """

#     fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
#     obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
#     df = pd.DataFrame({"fcst": fcst_pd_series, "obs": obs_pd_series})
#     expected = 0.7273
#     result = scores.continuous.mae(df["fcst"], df["obs"])
#     assert isinstance(result, float)
#     assert round(result, PRECISION) == expected
