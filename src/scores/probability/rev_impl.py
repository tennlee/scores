"""
Relative Economic Value metrics for forecast evaluation.

Three public functions are provided, each suited to a different workflow:

- :py:func:`relative_economic_value_from_rates` — pure math from pre-computed
  POD, POFD and climatology.
- :py:func:`relative_economic_value_from_contingency` — from a pre-built
  :py:class:`~scores.categorical.BinaryContingencyManager` or
  :py:class:`~scores.categorical.BasicContingencyManager`.
- :py:func:`relative_economic_value_from_threshold` — from raw forecast /
  observation arrays with optional probability-threshold discretisation.
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import xarray as xr

from scores.categorical import BasicContingencyManager, BinaryContingencyManager
from scores.processing import binary_discretise, broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike, all_same_xarraylike
from scores.utils import check_binary, check_weights, gather_dimensions

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_dimensions(
    fcst: XarrayLike,
    obs: XarrayLike,
    weights: Optional[xr.DataArray],
    threshold_dim: str,
    cost_loss_dim: str,
) -> None:
    """Check for dimension conflicts in inputs."""
    inputs = [("fcst", fcst), ("obs", obs)]
    if weights is not None:
        inputs.append(("weights", weights))

    for dim_name in (threshold_dim, cost_loss_dim):
        for input_name, input_data in inputs:
            if dim_name in input_data.dims:
                raise ValueError(f"'{dim_name}' cannot be a dimension in {input_name}")


def _validate_thresholds(
    threshold: Optional[Union[float, Sequence[float]]],
    threshold_outputs: Optional[Sequence[float]],
) -> None:
    """Validate threshold and threshold_outputs configuration."""
    if threshold is not None:
        try:
            check_monotonic_array(np.atleast_1d(threshold))
        except (ValueError, TypeError) as ex:
            raise type(ex)(f"for threshold, {ex}") from ex

        if threshold_outputs is not None:
            thresh_array = np.atleast_1d(threshold)
            if not set(threshold_outputs) <= set(thresh_array):
                raise ValueError("values in threshold_outputs must be in the supplied threshold parameter")
    else:
        if threshold_outputs:
            raise ValueError("threshold_outputs can only be used when threshold parameter is provided")


def _validate_cost_loss_ratios(cost_loss_ratios: Union[float, Sequence[float]]) -> None:
    """Validate cost-loss ratio values are in [0,1] and monotonically increasing."""
    if cost_loss_ratios is None:
        raise ValueError("cost_loss_ratios must not be None")
    try:
        check_monotonic_array(np.atleast_1d(cost_loss_ratios))
    except (ValueError, TypeError) as ex:
        raise type(ex)(f"for cost_loss_ratios, {ex}") from ex


def _validate_forecasts(
    fcst: XarrayLike,
    threshold: Optional[Union[float, Sequence[float]]],
) -> None:
    """
    Validate forecast values and threshold configuration.

    Raises ValueError if the threshold is provided but forecast is not
    between 0 and 1 or threshold is None but the forecast is not 0, 1 or NaN.
    """
    if isinstance(fcst, xr.Dataset):
        fcst_min = min(var.min().item() for var in fcst.data_vars.values())
        fcst_max = max(var.max().item() for var in fcst.data_vars.values())
    else:
        fcst_vals = fcst.values
        fcst_min = fcst_vals.min().item()
        fcst_max = fcst_vals.max().item()

    if threshold is not None:
        if fcst_min < 0 or fcst_max > 1:
            raise ValueError("When threshold is provided, fcst must contain values between 0 and 1")
    else:
        check_binary(fcst, "fcst")


def _validate_derived_metrics(
    derived_metrics: Optional[Sequence[str]],
    threshold: Optional[Union[float, Sequence[float]]],
) -> None:
    """Validate derived metrics configuration."""
    if derived_metrics is None:
        return

    valid_special = {"maximum", "rational_user"}
    invalid = set(derived_metrics) - valid_special
    if invalid:
        raise ValueError(f"Invalid derived_metrics values: {invalid}. Valid options are {valid_special}")

    if "rational_user" in derived_metrics and threshold is None:
        raise ValueError("derived_metrics 'rational_user' can only be used when threshold parameter is provided")


def _validate_threshold_inputs(
    fcst: XarrayLike,
    obs: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    threshold: Optional[Union[float, Sequence[float]]],
    threshold_dim: str,
    cost_loss_dim: str,
    weights: Optional[xr.DataArray],
    derived_metrics: Optional[Sequence[str]],
    threshold_outputs: Optional[Sequence[float]],
) -> None:
    """Validate inputs for the threshold-based REV calculation."""
    if isinstance(weights, xr.Dataset):
        raise ValueError("Weights cannot be Datasets. Convert to a DataArray or calculate separately.")

    _validate_dimensions(fcst, obs, weights, threshold_dim, cost_loss_dim)
    _validate_cost_loss_ratios(cost_loss_ratios)
    _validate_thresholds(threshold, threshold_outputs)
    _validate_derived_metrics(derived_metrics, threshold)
    check_binary(obs, "obs")
    _validate_forecasts(fcst, threshold)

    if weights is not None:
        check_weights(weights)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _calculate_rev_core(
    binary_fcst: XarrayLike,
    obs: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    dims_to_reduce: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    cost_loss_dim: str = "cost_loss_ratio",
) -> XarrayLike:
    """Core REV calculation from binary forecasts via BinaryContingencyManager."""
    binary_fcst, obs = broadcast_and_match_nan(binary_fcst, obs)

    manager = BinaryContingencyManager(binary_fcst, obs)
    basic = manager.transform(reduce_dims=dims_to_reduce, weights=weights)

    return relative_economic_value_from_rates(
        pod=basic.hit_rate(),
        pofd=basic.false_alarm_rate(),
        climatology=basic.base_rate(),
        cost_loss_ratios=cost_loss_ratios,
        cost_loss_dim=cost_loss_dim,
    )


def calculate_climatology(
    obs: XarrayLike,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> XarrayLike:
    """
    Calculates the climatological base rate (mean of observations).

    Args:
        obs: An array containing binary values (typically {0, 1, np.nan})
        reduce_dims: Dimensions to reduce. Default reduces all.
        preserve_dims: Dimensions to preserve. Default reduces all.
        weights: Optional weights for weighted mean.

    Returns:
        A DataArray of the climatological base rate.
    """
    from scores.processing import aggregate

    dims_to_reduce = gather_dimensions(
        fcst_dims=(),
        obs_dims=obs.dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights_dims=weights.dims if weights is not None else None,
    )
    return aggregate(obs, reduce_dims=dims_to_reduce, weights=weights, method="mean")


def _create_output_dataset(
    rev: xr.DataArray,
    thresholds: Sequence[float],
    cost_loss_ratios: Sequence[float],
    derived_metrics: Optional[Sequence[str]],
    threshold_outputs: Optional[Sequence[float]],
    threshold_dim: str,
    cost_loss_dim: str,
) -> xr.Dataset:
    """Create output Dataset with derived metrics and threshold slices."""
    derived_metrics = [] if derived_metrics is None else list(derived_metrics)
    threshold_outputs = [] if threshold_outputs is None else list(threshold_outputs)

    result = xr.Dataset(attrs=rev.attrs)

    for mode in derived_metrics:
        if mode == "maximum":
            result["maximum"] = rev.max(dim=threshold_dim)
        elif mode == "rational_user":  # pragma: no cover
            if list(thresholds) != list(cost_loss_ratios):
                raise ValueError(
                    "Can only specify derived_metrics 'rational_user' if thresholds and cost_loss_ratios are identical"
                )
            actual_values = []
            for alpha in cost_loss_ratios:
                val = rev.sel({threshold_dim: alpha, cost_loss_dim: alpha})
                actual_values.append(val)

            result["rational_user"] = xr.concat(actual_values, dim=cost_loss_dim)
            result["rational_user"][cost_loss_dim] = cost_loss_ratios

            if threshold_dim in result["rational_user"].coords:
                result["rational_user"] = result["rational_user"].drop_vars(threshold_dim)

    for thresh in threshold_outputs:
        var_name = f"threshold_{thresh}".replace(".", "_")
        result[var_name] = rev.sel({threshold_dim: thresh}).drop_vars(threshold_dim)

    return result


def check_monotonic_array(array: Union[Sequence[float], np.ndarray]) -> None:
    """Checks array values are in range [0, 1] and monotonically increasing."""
    try:
        np_array = np.array(array, dtype=float)
    except Exception as ex:
        raise TypeError("could not convert array into a numpy ndarray of floats") from ex

    if len(np_array.shape) != 1:
        raise ValueError("array must be one-dimensional")

    if max(np_array) > 1 or min(np_array) < 0:
        raise ValueError("array values should be between 0 and 1.")

    if len(np_array) > 1 and not (np_array[1:] - np_array[:-1] >= 0).all():
        raise ValueError("the supplied array is not monotonically increasing.")


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def relative_economic_value_from_rates(
    pod: XarrayLike,
    pofd: XarrayLike,
    climatology: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    cost_loss_dim: str = "cost_loss_ratio",
) -> XarrayLike:
    """
    Calculates Relative Economic Value (REV) from pre-computed detection rates.

    REV measures the economic benefit of using forecasts compared to climatology,
    relative to perfect forecasts. This function computes REV directly from
    probability of detection (POD), probability of false detection (POFD), and
    climatological frequency.

    The relative economic value is calculated using:

    .. math::
        \\begin{split}
        \\text{REV} = \\frac{\\min(\\alpha, \\bar{o}) - F\\alpha(1-\\bar{o})
                              + H\\bar{o}(1-\\alpha) - \\bar{o}}
                             {\\min(\\alpha, \\bar{o}) - \\bar{o}\\alpha}
        \\end{split}

    where:
        - :math:`\\alpha` is the cost-loss ratio
        - :math:`\\bar{o}` is the climatological frequency (base rate)
        - :math:`F` is the probability of false detection (false alarm rate)
        - :math:`H` is the probability of detection (hit rate)

    Args:
        pod: Probability of detection (hit rate). Values should be between 0 and 1,
            where 1 indicates all events were correctly forecast.
        pofd: Probability of false detection (false alarm rate). Values should be
            between 0 and 1, where 0 indicates no false alarms.
        climatology: Climatological frequency of the event (base rate). Values should
            be between 0 and 1, representing the proportion of time the event occurs.
        cost_loss_ratios: Cost-loss ratio(s) at which to calculate REV. Must be
            strictly monotonically increasing values between 0 and 1. Can be a single
            float or sequence of floats.
        cost_loss_dim: Name of the cost-loss ratio dimension in output. Default is
            'cost_loss_ratio'. Must not exist as a dimension in any input array.

    Returns:
        xarray.DataArray or xarray.Dataset: REV values with an additional
            'cost_loss_ratio' dimension.

    Raises:
        TypeError: If 'pod' and 'pofd' are not both DataArrays or both Datasets.
        ValueError: If 'cost_loss_ratio' dimension already exists in any input.

    References:
        - Richardson, D. S. (2000). Skill and relative economic value of the ECMWF
          ensemble prediction system. *Q. J. R. Meteorol. Soc.*, 126(563), 649-667.

    See Also:
        - :py:func:`relative_economic_value_from_contingency`
        - :py:func:`relative_economic_value_from_threshold`
    """
    if not all_same_xarraylike([pod, pofd]):
        raise TypeError("Both pod and pofd must be either xarray DataArrays or xarray Datasets.")

    if cost_loss_dim in (set(pod.dims) | set(pofd.dims) | set(climatology.dims)):
        raise ValueError(f"dimension '{cost_loss_dim}' must not be in input data")

    # Handle Dataset inputs by applying to each variable
    if isinstance(pod, xr.Dataset):
        result_dict = {}
        for var in pod.data_vars:
            result_dict[var] = relative_economic_value_from_rates(
                pod[var],
                pofd[var] if isinstance(pofd, xr.Dataset) else pofd,
                climatology[var] if isinstance(climatology, xr.Dataset) else climatology,
                cost_loss_ratios,
                cost_loss_dim=cost_loss_dim,
            )
        return xr.Dataset(result_dict)

    alphas = xr.DataArray(
        cost_loss_ratios,
        dims=[cost_loss_dim],
        coords={cost_loss_dim: cost_loss_ratios},
    )

    obar, alphas = xr.broadcast(climatology, alphas)
    climatological_term = xr.where(obar < alphas, obar, alphas)

    rev = (climatological_term - pofd * alphas * (1 - obar) + pod * obar * (1 - alphas) - obar) / (
        climatological_term - obar * alphas
    )

    # Tidy up floating point infinities from near-zero denominators at alpha=0 or alpha=1
    rev = rev.where(~np.isinf(rev))

    return rev


def relative_economic_value_from_contingency(
    contingency_manager: Union[BinaryContingencyManager, BasicContingencyManager],
    cost_loss_ratios: Union[float, Sequence[float]],
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    cost_loss_dim: str = "cost_loss_ratio",
) -> XarrayLike:
    """
    Calculates Relative Economic Value (REV) from a contingency manager.

    This is the recommended entry point when you want to use a custom
    :py:class:`~scores.categorical.EventOperator` (e.g.
    :py:class:`~scores.categorical.ThresholdEventOperator`) to define events,
    or when you already have a contingency table from another analysis.

    When a :py:class:`~scores.categorical.BinaryContingencyManager` is supplied,
    it is transformed (with optional ``reduce_dims``, ``preserve_dims`` and
    ``weights``) to produce a
    :py:class:`~scores.categorical.BasicContingencyManager`.  When a
    :py:class:`~scores.categorical.BasicContingencyManager` is supplied
    directly (already transformed), ``reduce_dims``, ``preserve_dims`` and
    ``weights`` are ignored.

    .. math::
        \\begin{split}
        \\text{REV} = \\frac{\\min(\\alpha, \\bar{o}) - F\\alpha(1-\\bar{o})
                              + H\\bar{o}(1-\\alpha) - \\bar{o}}
                             {\\min(\\alpha, \\bar{o}) - \\bar{o}\\alpha}
        \\end{split}

    Args:
        contingency_manager: A pre-built contingency manager. Can be either a
            :py:class:`~scores.categorical.BinaryContingencyManager` (which will
            be transformed) or a
            :py:class:`~scores.categorical.BasicContingencyManager` (used as-is).
        cost_loss_ratios: Cost-loss ratio(s) at which to calculate REV. Must be
            monotonically increasing values between 0 and 1.
        reduce_dims: Dimensions to reduce when transforming a
            ``BinaryContingencyManager``. Ignored for ``BasicContingencyManager``.
        preserve_dims: Dimensions to preserve when transforming a
            ``BinaryContingencyManager``. Ignored for ``BasicContingencyManager``.
        weights: Optional weights for weighted aggregation when transforming a
            ``BinaryContingencyManager``. Ignored for ``BasicContingencyManager``.
        cost_loss_dim: Name of the cost-loss ratio dimension in output.

    Returns:
        xarray.DataArray: REV values with a ``cost_loss_ratio`` dimension.

    Examples:
        Using a :py:class:`~scores.categorical.ThresholdEventOperator`:

        >>> from scores.categorical import ThresholdEventOperator
        >>> event_op = ThresholdEventOperator(default_event_threshold=10)
        >>> manager = event_op.make_contingency_manager(fcst, obs, event_threshold=10)
        >>> rev = relative_economic_value_from_contingency(
        ...     manager, cost_loss_ratios=[0.1, 0.3, 0.5, 0.7, 0.9]
        ... )

        Using a pre-built :py:class:`~scores.categorical.BinaryContingencyManager`
        with weights:

        >>> from scores.categorical import BinaryContingencyManager
        >>> manager = BinaryContingencyManager(binary_fcst, binary_obs)
        >>> rev = relative_economic_value_from_contingency(
        ...     manager,
        ...     cost_loss_ratios=[0.3, 0.5, 0.7],
        ...     weights=lat_weights,
        ... )

    See Also:
        - :py:func:`relative_economic_value_from_rates`
        - :py:func:`relative_economic_value_from_threshold`
        - :py:class:`scores.categorical.BinaryContingencyManager`
        - :py:class:`scores.categorical.ThresholdEventOperator`
    """
    _validate_cost_loss_ratios(cost_loss_ratios)

    if isinstance(cost_loss_ratios, (float, int)):
        cost_loss_ratios = [cost_loss_ratios]

    if isinstance(contingency_manager, BinaryContingencyManager):
        basic = contingency_manager.transform(
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
            weights=weights,
        )
    else:
        basic = contingency_manager

    return relative_economic_value_from_rates(
        pod=basic.hit_rate(),
        pofd=basic.false_alarm_rate(),
        climatology=basic.base_rate(),
        cost_loss_ratios=cost_loss_ratios,
        cost_loss_dim=cost_loss_dim,
    )


def relative_economic_value_from_threshold(
    fcst: XarrayLike,
    obs: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    *,
    threshold: Optional[Union[float, Sequence[float]]] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    threshold_dim: str = "threshold",
    cost_loss_dim: str = "cost_loss_ratio",
    derived_metrics: Optional[Sequence[str]] = None,
    threshold_outputs: Optional[Sequence[float]] = None,
    check_args: bool = True,
) -> XarrayLike:
    """
    Calculates Relative Economic Value (REV) from forecast and observation arrays.

    For probabilistic forecasts, multiple decision thresholds are evaluated to find
    the optimal strategy. For binary forecasts, a single decision has already been
    made.

    For ensemble forecasts, consider using
    :py:func:`scores.processing.binary_discretise_proportion` to convert ensembles
    to empirical probabilities before calculating REV.

    .. math::
        \\begin{split}
        \\text{REV} = \\frac{\\min(\\alpha, \\bar{o}) - F\\alpha(1-\\bar{o})
                              + H\\bar{o}(1-\\alpha) - \\bar{o}}
                             {\\min(\\alpha, \\bar{o}) - \\bar{o}\\alpha}
        \\end{split}

    where:
        - :math:`\\bar{o}` is the climatological frequency (base rate)
        - :math:`\\alpha` is the cost-loss ratio
        - :math:`F` is the probability of false detection (false alarm rate)
        - :math:`H` is the probability of detection (hit rate)

    Args:
        fcst: Forecast data. Can be:
            - Probabilistic: values between 0 and 1 (requires ``threshold``)
            - Binary: values of 0 or 1
        obs: Binary observations (0 or 1).
        cost_loss_ratios: Cost-loss ratio(s) at which to calculate REV. Must be
            monotonically increasing values between 0 and 1.
        threshold: Decision threshold(s) for converting probabilistic forecasts to
            binary decisions. Each threshold converts forecasts to 1 where
            fcst >= threshold, 0 otherwise. If None, assumes fcst is already binary.
        reduce_dims: Dimensions to reduce.
        preserve_dims: Dimensions to preserve.
        weights: Optional weights for weighted averaging.
        threshold_dim: Name of the threshold dimension in output.
        cost_loss_dim: Name of the cost-loss ratio dimension in output.
        derived_metrics: Optional list of derived metrics to compute:
            - ``'maximum'``: Maximum REV across all thresholds.
            - ``'rational_user'``: REV when threshold equals cost-loss ratio
              (requires thresholds to match cost_loss_ratios exactly).
        threshold_outputs: Specific threshold values to extract as separate
            Dataset variables.
        check_args: If True, validates input arguments.

    Returns:
        XarrayLike:
            - If ``derived_metrics`` or ``threshold_outputs``: xr.Dataset
            - If ``threshold`` is provided: xr.DataArray with threshold_dim
            - Otherwise: xr.DataArray with cost_loss_dim only

    Raises:
        ValueError: For invalid inputs (see parameter descriptions).

    References:
        - Richardson, D. S. (2000). Skill and relative economic value of the ECMWF
          ensemble prediction system. *Q. J. R. Meteorol. Soc.*, 126(563), 649-667.

    Examples:
        Binary forecasts:

        >>> rev = relative_economic_value_from_threshold(
        ...     fcst, obs, cost_loss_ratios=[0.1, 0.3, 0.5, 0.7, 0.9]
        ... )

        Probabilistic forecasts with maximum value:

        >>> result = relative_economic_value_from_threshold(
        ...     fcst_prob, obs, cost_loss_ratios,
        ...     threshold=[0.3, 0.5, 0.7],
        ...     derived_metrics=['maximum'],
        ... )

    See Also:
        - :py:func:`relative_economic_value_from_rates`
        - :py:func:`relative_economic_value_from_contingency`
    """
    # Input validation
    if check_args:
        _validate_threshold_inputs(
            fcst,
            obs,
            cost_loss_ratios,
            threshold,
            threshold_dim,
            cost_loss_dim,
            weights,
            derived_metrics,
            threshold_outputs,
        )

    # --- Dataset dispatch ---
    if isinstance(fcst, xr.Dataset) and isinstance(obs, xr.Dataset):
        fcst_aligned, obs_aligned = xr.align(fcst, obs, join="inner")
        name_sep = "__vs__"
        result_dict = {}
        for fvar in sorted(fcst_aligned.data_vars):
            for ovar in sorted(obs_aligned.data_vars):
                out_name = f"{fvar}{name_sep}{ovar}"
                result_dict[out_name] = relative_economic_value_from_threshold(
                    fcst_aligned[fvar],
                    obs_aligned[ovar],
                    cost_loss_ratios,
                    threshold=threshold,
                    reduce_dims=reduce_dims,
                    preserve_dims=preserve_dims,
                    weights=weights,
                    threshold_dim=threshold_dim,
                    cost_loss_dim=cost_loss_dim,
                    derived_metrics=derived_metrics,
                    threshold_outputs=threshold_outputs,
                    check_args=False,
                )
        return xr.Dataset(result_dict)

    if isinstance(fcst, xr.Dataset):
        result_dict = {}
        for var in fcst.data_vars:
            result_dict[var] = relative_economic_value_from_threshold(
                fcst[var],
                obs,
                cost_loss_ratios,
                threshold=threshold,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
                threshold_dim=threshold_dim,
                cost_loss_dim=cost_loss_dim,
                derived_metrics=derived_metrics,
                threshold_outputs=threshold_outputs,
                check_args=False,
            )
        return xr.Dataset(result_dict)

    if isinstance(obs, xr.Dataset):
        result_dict = {}
        for var in obs.data_vars:
            result_dict[var] = relative_economic_value_from_threshold(
                fcst,
                obs[var],
                cost_loss_ratios,
                threshold=threshold,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
                threshold_dim=threshold_dim,
                cost_loss_dim=cost_loss_dim,
                derived_metrics=derived_metrics,
                threshold_outputs=threshold_outputs,
                check_args=False,
            )
        return xr.Dataset(result_dict)

    # --- Scalar path ---
    if isinstance(cost_loss_ratios, (float, int)):
        cost_loss_ratios = [cost_loss_ratios]

    weights_dims = weights.dims if weights is not None else None
    dims_to_reduce = gather_dimensions(
        fcst.dims,
        obs.dims,
        weights_dims=weights_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

    if threshold is not None:
        if isinstance(threshold, (float, int)):
            threshold = [threshold]

        binary_fcst = binary_discretise(fcst, threshold, ">=")

        if threshold_dim != "threshold":
            binary_fcst = binary_fcst.rename({"threshold": threshold_dim})

        rev = _calculate_rev_core(
            binary_fcst,
            obs,
            cost_loss_ratios,
            dims_to_reduce=dims_to_reduce,
            weights=weights,
            cost_loss_dim=cost_loss_dim,
        )

        if derived_metrics or threshold_outputs:
            return _create_output_dataset(
                rev,
                threshold,
                cost_loss_ratios,
                derived_metrics,
                threshold_outputs,
                threshold_dim,
                cost_loss_dim,
            )

        return rev

    # Binary forecast path
    return _calculate_rev_core(
        fcst,
        obs,
        cost_loss_ratios,
        dims_to_reduce=dims_to_reduce,
        weights=weights,
    )
