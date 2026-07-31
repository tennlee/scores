"""
Provides interfaces for common array operations compatible with Python Array API
and xarray. Purpose is to provide a single interface for these array operations
to a) reduce code duplication, and b) ensure same treatment of NaNs for different
backends.
"""

from array_api_compat import array_namespace

from scores.typing import is_xarraylike


def where(condition, a, b, namespace=None):
    """
    Selects elementwise from ``a`` where ``condition`` is True, and from ``b``
    otherwise.

    Dispatches to ``a.where(condition, b)`` for xarray objects and to
    ``xp.where(condition, a, b)`` for Python Array API objects, so that callers
    need not know which backend they hold. Note the differing argument order of
    the two backends is handled here.

    .. note::

        For internal use only.

    Args:
        condition: boolean array selecting between ``a`` and ``b``. Must be
            broadcastable against both.
        a: values to take where ``condition`` is True.
        b: values to take where ``condition`` is False. May be a scalar
            (e.g. ``numpy.nan``) for either backend.
        namespace: array namespace to dispatch to, when it is already known to the
            caller. Supplying it skips type inspection, so it is the preferred
            form inside loops or hot paths. Pass ``None`` for xarray inputs, or
            to have the namespace inferred from the arguments.

    Returns:
        An array of the same backend as the inputs, holding elements of ``a``
        where ``condition`` is True and elements of ``b`` elsewhere.
    """
    if namespace is None:
        if is_xarraylike(condition):
            # xr.where exists, but unclear about how it treats coord dropping
            return a.where(condition, b)
        # raises an exception if condition isn't supported
        xp = array_namespace(condition)
    else:
        xp = namespace
    return xp.where(condition, a, b)
