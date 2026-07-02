"""
Contains unit tests for scores.continuous.standard
"""

# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import math
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

import scores

PRECISION = 4

# Mean Squared Error
#
DA1_BIAS = torch.tensor(
    np.array([[1, 1, np.nan], [0, 0, 0], [0.5, -0.5, 0.5]]),
).float()

DA2_BIAS = torch.tensor(
    np.array([[2, 2, 6], [2, 10, 0], [-0.5, 0.5, -0.5]]),
).float()

BIAS_WEIGHTS = torch.tensor(
    np.array([[1, 1, 1], [3, 0, 0], [3, 0, 0]]),
).float()

EXP_BIAS2 = torch.tensor(np.array([-1.33333])).float()
EXP_BIAS3 = torch.tensor(np.array(-1.625)).float()

@pytest.fixture(params=[True, False])
def mse_test_args(request):
    is_angular = request.param
    args = {
        "fcst": [180*i/math.pi if is_angular else i for i in [1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3]],
        "obs": [180*i/math.pi if is_angular else i for i in [1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1]],
        "is_angular": is_angular
    }
    args["expected"] = float(scores.continuous.mse(
        xr.DataArray(args["fcst"]),
        xr.DataArray(args["obs"]),
        is_angular = args["is_angular"]
    ))
    return args


@pytest.mark.parametrize(
    ("device",),
    [
        pytest.param(
            device, 
            marks=pytest.mark.skipif(
                not getattr(torch, device).is_available(),
                reason=f"torch device {device} not available."
            )
        ) for device in ("cpu", "cuda", "mps")
    ]
)
def test_mse_torch_host_device_consistency(mse_test_args, device):
    """
    Tests that execution on available devices match host
    serial answers within tolerance.
    """

    fcst = mse_test_args["fcst"]
    obs = mse_test_args["obs"]
    expected = mse_test_args["expected"]

    fcst_tensor = torch.tensor(fcst, dtype=torch.float).to(device)
    obs_tensor = torch.tensor(obs, dtype=torch.float).to(device)
    device_result = scores.loss.mse(fcst_tensor, obs_tensor)
    assert isinstance(device_result, torch.Tensor)
    assert device_result.device == fcst_tensor.device
    torch.testing.assert_close(
        device_result, 
        torch.tensor(expected, device=device),
    )


@pytest.mark.parametrize(
    ("arr_type", ),
    [(np.array,), (pd.Series,)],
)
def test_mse_array_consistency(arr_type, mse_test_args):
    """
    Tests that different numpy-like arrays give the same result.
    """
    fcst = arr_type(mse_test_args["fcst"])
    obs = arr_type(mse_test_args["obs"])
    result = scores.loss.mse(fcst, obs)
    np.testing.assert_allclose(mse_test_args["expected"], result)

@pytest.mark.parametrize(
    ("fcst", "obs", "weights", "expected"),
    [
        # Check weighting works
        # (DA1_BIAS, DA2_BIAS, BIAS_WEIGHTS, EXP_BIAS2),
        (DA1_BIAS, DA2_BIAS, None, EXP_BIAS3),
    ],
)
def test_additive_bias(fcst, obs, weights, expected):
    """
    Tests continuous.additive_bias
    Also tests mean_error (which is an identical function)
    """

    fcst = fcst.rename(None)
    obs = obs.rename(None)

    if weights is None:
        weights = torch.ones(fcst.shape)

    weights = weights.rename(None)

    weights = weights * (~torch.isnan(fcst))  # mask out nans from fcst
    weights = weights * (~torch.isnan(obs))  # mask out nans from obs
    tensor_result = scores.loss.additive_bias(fcst, obs, weights=weights)

    tensor_result = tensor_result.rename(None)
    assert (torch.round(tensor_result, decimals=4) == torch.tensor(expected)).all()

    fcst_gpu = fcst.to(device="mps")
    obs_gpu = obs.to(device="mps")
    weights_gpu = weights.to(device="mps")
    _gpu_result = scores.loss.additive_bias(fcst_gpu, obs_gpu, weights=weights_gpu)


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
