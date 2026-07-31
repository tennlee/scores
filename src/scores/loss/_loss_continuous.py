from scores import continuous as __continuous


def mse(
    fcst,
    obs,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
    weights=None,
):
    """Calculates the mean squared error from forecast and observed data.

    A detailed explanation is on https://en.wikipedia.org/wiki/Mean_squared_error

    .. math ::
        \\frac{1}{n} \\sum_{i=1}^n (\\text{forecast}_i - \\text{observed}_i)^2


    Notes:
        Dimensional reduction is not supported for pandas and the user should
        convert their data to xarray to formulate the call to the base metric,
        `scores.continuous.mse`.

    Args:
        fcst: Forecast or predicted variables in pandas.
        obs: Observed variables in pandas.
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.

    Returns:
        pandas.Series:
            An object containing a single floating point number representing the mean squared
            error for the supplied data. All dimensions will be reduced.

    """
    return __continuous.mse(fcst, obs, is_angular=is_angular, weights=weights)


def additive_bias(
    fcst,
    obs,
    *,
    reduce_dims=None,
    weights=None,
):
    """
    Calculates the additive bias which is also sometimes called the mean error.

    It is defined as

    .. math::
        \\text{Additive bias} =\\frac{1}{N}\\sum_{i=1}^{N}(x_i - y_i)
        \\text{where } x = \\text{the forecast, and } y = \\text{the observation}


    See "Mean error" section at https://jwgfvr.github.io/forecastverification/index.html#meanerror
    for more information.

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the additive bias. All other dimensions will be preserved.
        weights: An array of weights to apply to the score (e.g., weighting a grid by latitude).
            If None, no weights are applied. If provided, the weights must be broadcastable
            to the data dimensions and must not contain negative or NaN values. If
            appropriate, NaN values in weights  can be replaced by ``weights.fillna(0)``.
            The weighting approach follows :py:class:`xarray.computation.weighted.DataArrayWeighted`.
            See the ``scores`` weighting tutorial for more information on how to use weights.

    Returns:
        An xarray object with the additive bias of a forecast.

    References:
        -   https://jwgfvr.github.io/forecastverification/index.html#meanerror

    """

    score = __continuous.additive_bias(fcst, obs, weights=weights)

    return score


def tw_huber_loss(
    fcst,
    obs,
    huber_param,
    interval_where_one,
    *,
    interval_where_positive=None,
    # TODO: implement preserve_dims = "all"
    # TODO: implement weights
):
    """
    Returns the threshold weighted Huber loss.

    For more flexible threshold weighting schemes,
    see :py:func:`scores.continuous.consistent_huber_score`.

    Two types of threshold weighting are supported: rectangular and trapezoidal.
        - To specify a rectangular threshold weight, set ``interval_where_positive=None`` and set
            ``interval_where_one`` to be the interval where the threshold weight is 1.
            For example, if  ``interval_where_one=(0, 10)`` then a threshold weight of 1
            is applied to decision thresholds satisfying 0 <= threshold < 10, and a threshold weight of 0 is
            applied otherwise. Interval endpoints can be ``-numpy.inf`` or ``numpy.inf``.
        - To specify a trapezoidal threshold weight, specify ``interval_where_positive`` and ``interval_where_one``
            using desired endpoints. For example, if ``interval_where_positive=(-2, 10)`` and
            ``interval_where_one=(2, 4)`` then a threshold weight of 1 is applied to decision thresholds
            satisfying 2 <= threshold < 4. The threshold weight increases linearly from 0 to 1 on the interval
            [-2, 2) and decreases linearly from 1 to 0 on the interval [4, 10], and is 0 otherwise.
            Interval endpoints can only be infinite if the corresponding ``interval_where_one`` endpoint
            is infinite. End points of ``interval_where_positive`` and ``interval_where_one`` must differ
            except when the endpoints are infinite.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        huber_param: the Huber transition parameter.
        interval_where_one: endpoints of the interval where the threshold weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        interval_where_positive: endpoints of the interval where the threshold weights are positive.
            Must be increasing. Infinite endpoints are only permissible when the corresponding
            ``interval_where_one`` endpoint is infinite. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_expectile_score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_expectile_score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_expectile_score
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of ``reduce_dims``
            and ``preserve_dims`` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: An array of weights to apply to the score (e.g., weighting a grid by latitude).
            If None, no weights are applied. If provided, the weights must be broadcastable
            to the data dimensions and must not contain negative or NaN values. If
            appropriate, users can choose to replace NaN values in weights by calling ``weights.fillna(0)``.
            The weighting approach follows :py:class:`xarray.computation.weighted.DataArrayWeighted`.
            See the scores weighting tutorial for more information on how to use weights.

    Returns:
        xarray data array of the threshold weighted expectile error
    """

    score = __continuous.tw_huber_loss(
        fcst, obs, huber_param, interval_where_one, interval_where_positive=interval_where_positive
    )

    return score
