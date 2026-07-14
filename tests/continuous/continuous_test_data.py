"""
Shared test input data for `scores.continuous.mse` / `scores.loss.mse` and
`scores.continuous.additive_bias` / `scores.loss.additive_bias`.

The canonical inputs are stored as `xr.DataArray` objects. Update the
DataArrays here to have the change propagate to both loss and continuous
tests.
"""

import numpy as np
import xarray as xr

# Mean squared error test vectors
MSE_FCST_DA = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
MSE_OBS_DA = xr.DataArray([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
MSE_WEIGHTS_DA = xr.DataArray(np.random.default_rng(42).random(MSE_FCST_DA.size))

# Additive bias test arrays
_BIAS_COORDS = [("space", ["w", "x", "y"]), ("time", [1, 2, 3])]

BIAS_FCST_DA = xr.DataArray(
    np.array([[1, 1, np.nan], [0, 0, 0], [0.5, -0.5, 0.5]]),
    dims=("space", "time"),
    coords=_BIAS_COORDS,
)

BIAS_OBS_DA = xr.DataArray(
    np.array([[2, 2, 6], [2, 10, 0], [-0.5, 0.5, -0.5]]),
    dims=("space", "time"),
    coords=_BIAS_COORDS,
)

BIAS_WEIGHTS_DA = xr.DataArray(
    np.array([[1, 1, 1], [3, 0, 0], [3, 0, 0]]),
    dims=("space", "time"),
    coords=_BIAS_COORDS,
)

TW_FCST_DA = xr.DataArray(
    data=[[[3.0, 1.0, np.nan, 2], [3.0, 1.0, np.nan, 2]], [[-4.0, 0.0, 1.0, 2], [-4.0, 0.0, 1.0, 2]]],
    dims=["date", "lead_day", "station"],
    coords=dict(
        date=["1", "2"],
        lead_day=[1, 1],
        station=[100, 101, 102, 0],
    ),
)

TW_OBS_DA = xr.DataArray(
    data=[[np.nan, 3.0, 5.0], [-4.0, 10.0, -1.0], [3.0, 2.0, -0.2]],
    dims=["date", "station"],
    coords=dict(date=["1", "2", "3"], station=[100, 101, 102]),
)
