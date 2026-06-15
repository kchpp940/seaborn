"""Unified parameter validation and deprecation helpers.

This module provides a consistent interface for parameter validation,
deprecation warnings, and error handling across seaborn's plotting functions.
"""
import textwrap
import warnings
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union


__all__ = [
    "_check_argument",
    "_check_figure_level_ax",
    "_check_mutually_exclusive",
    "_deprecate_ci",
    "_handle_deprecated_param",
    "_handle_ignored_param",
]


def _handle_deprecated_param(
    param: str,
    value: Any,
    *,
    new_param: Optional[str] = None,
    new_value: Any = None,
    target: Optional[Mapping] = None,
    target_key: Optional[str] = None,
    message: Optional[str] = None,
    since: str = "0.12.0",
    remove_version: str = "0.14.0",
    warning_type: type = UserWarning,
    stacklevel: int = 2,
    action: str = "set",
    transform: Optional[callable] = None,
    error_type: Optional[type] = None,
) -> Any:
    """Handle a deprecated parameter with consistent messaging.

    Parameters
    ----------
    param : str
        Name of the deprecated parameter.
    value : Any
        The value passed for the deprecated parameter. If None, no warning is issued.
    new_param : str, optional
        Name of the replacement parameter. If provided, the warning will suggest
        using this parameter instead.
    new_value : Any, optional
        Value to use for the new parameter. If not provided, uses the old value.
    target : Mapping, optional
        A dict to update with the new parameter and value.
    target_key : str, optional
        Key to use in the target dict. Defaults to new_param or param.
    message : str, optional
        Custom warning message. If provided, overrides the default message.
    since : str, optional
        Version when the parameter was deprecated.
    remove_version : str, optional
        Version when the parameter will be removed.
    warning_type : type, optional
        Warning class to use (UserWarning, FutureWarning, etc.).
    stacklevel : int, optional
        Stack level for the warning.
    action : {"set", "pop"}, optional
        Whether to "set" the new value in target or "pop" from kwargs first.
        Use "pop" when the deprecated param is in **kwargs.
    transform : callable, optional
        A function that takes (value, locals_dict) and returns (new_value, message_suffix).
        Used for complex deprecations where the new value depends on custom logic.
    error_type : type, optional
        If provided, raises this error type instead of warning. Use for parameters
        that have been fully removed.

    Returns
    -------
    Any
        The value that should be used for the new parameter, or None if no
        deprecation was triggered.

    Examples
    --------
    Simple rename:
    >>> fill = _handle_deprecated_param("shade", shade, new_param="fill")

    Move to dict:
    >>> _handle_deprecated_param("sharex", sharex, target=facet_kws)

    Pop from kwargs and set new param:
    >>> bw_method = _handle_deprecated_param("bw", kwargs.pop("bw", None),
    ...                                      new_param="bw_method")
    """
    if value is None:
        return None

    if action == "pop" and isinstance(value, dict):
        value = value.pop(param, None)
        if value is None:
            return None

    message_suffix = ""
    if transform is not None:
        new_value, message_suffix = transform(value)
    elif new_value is None:
        new_value = value

    if target is not None:
        key = target_key or new_param or param
        target[key] = new_value

    if error_type is not None:
        if message is None:
            msg = textwrap.dedent(f"""\n
            The `{param}` parameter has been removed; {message_suffix}
            Please update your code.
            """)
        else:
            msg = message
        raise error_type(msg)

    if message is None:
        if new_param is not None:
            msg = textwrap.dedent(f"""\n
            The `{param}` parameter is deprecated in favor of `{new_param}`;
            {message_suffix}setting `{new_param}={new_value}`.
            This will become an error in seaborn v{remove_version};
            please update your code.
            """)
        elif target is not None:
            key = target_key or new_param or param
            msg = textwrap.dedent(f"""\n
            `{param}` is deprecated from the function signature.
            Please update your code to pass it using `{key}`.
            """)
        else:
            msg = textwrap.dedent(f"""\n
            The `{param}` parameter is deprecated. {message_suffix}
            This will become an error in seaborn v{remove_version};
            please update your code.
            """)
    else:
        msg = message

    warnings.warn(msg, warning_type, stacklevel=stacklevel + 1)
    return new_value


def _check_figure_level_ax(
    func_name: str,
    kwargs: dict,
    kind: Optional[str] = None,
    message: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """Check for and warn about `ax` parameter in figure-level functions.

    Parameters
    ----------
    func_name : str
        Name of the figure-level function (e.g., "relplot", "catplot").
    kwargs : dict
        The kwargs dict to check and remove `ax` from.
    kind : str, optional
        The plot kind, if applicable, for suggesting the axes-level alternative.
        If provided, the suggestion will be "{kind}plot".
    message : str, optional
        Custom warning message. If provided, overrides the default message.
    stacklevel : int, optional
        Stack level for the warning.

    Examples
    --------
    >>> _check_figure_level_ax("relplot", kwargs, kind="scatter")
    """
    if "ax" in kwargs:
        if message is not None:
            msg = message
        elif kind is not None:
            axes_func = f"{kind}plot"
            msg = (
                f"`{func_name}` is a figure-level function and does not accept "
                f"the `ax` parameter. You may wish to try {axes_func}."
            )
        else:
            msg = f"Ignoring `ax`; {func_name} is a figure-level function."

        warnings.warn(msg, UserWarning, stacklevel=stacklevel + 1)
        kwargs.pop("ax")


def _check_mutually_exclusive(
    params: Sequence[Tuple[str, Any]],
    *,
    func_name: Optional[str] = None,
    message: Optional[str] = None,
    error_type: type = ValueError,
    stacklevel: int = 2,
) -> None:
    """Check that no more than one of the mutually exclusive parameters is provided.

    Parameters
    ----------
    params : sequence of (str, Any) tuples
        List of (param_name, param_value) tuples to check.
    func_name : str, optional
        Name of the function calling this check, for the error message.
    message : str, optional
        Custom error message. If provided, overrides the default message.
    error_type : type, optional
        Exception type to raise.
    stacklevel : int, optional
        Stack level for traceback.

    Raises
    ------
    ValueError
        If more than one parameter has a truthy value.

    Examples
    --------
    >>> _check_mutually_exclusive([
    ...     ("order", order > 1),
    ...     ("logistic", logistic),
    ...     ("robust", robust),
    ...     ("lowess", lowess),
    ...     ("logx", logx),
    ... ])
    """
    count = sum(1 for _, val in params if val)
    if count > 1:
        if message is not None:
            msg = message
        else:
            names = [name for name, val in params if val]
            if func_name:
                msg = f"In {func_name}: "
            else:
                msg = ""
            msg += f"Mutually exclusive parameters: {', '.join(names)}."
        raise error_type(msg)


def _check_argument(
    param: str,
    options: Iterable[Any],
    value: Any,
    *,
    prefix: bool = False,
    error_type: type = ValueError,
) -> Any:
    """Raise if value for param is not in options.

    Parameters
    ----------
    param : str
        Name of the parameter being validated.
    options : iterable
        Allowed values for the parameter.
    value : Any
        The value to validate.
    prefix : bool, optional
        If True, check if value starts with any of the options (for string values).
    error_type : type, optional
        Exception type to raise.

    Returns
    -------
    Any
        The validated value.

    Raises
    ------
    ValueError
        If value is not in options.

    Examples
    --------
    >>> _check_argument("multiple", ["layer", "stack", "fill"], multiple)
    """
    options = list(options)
    if prefix and value is not None:
        failure = not any(
            value.startswith(p) for p in options if isinstance(p, str)
        )
    else:
        failure = value not in options

    if failure:
        raise error_type(
            f"The value for `{param}` must be one of {options}, "
            f"but {repr(value)} was passed."
        )
    return value


def _deprecate_ci(
    errorbar: Any,
    ci: Any,
    *,
    stacklevel: int = 2,
) -> Any:
    """Warn on usage of ci= and convert to appropriate errorbar= arg.

    ci was deprecated when errorbar was added in 0.12. It should not be removed
    completely for some time, but it can be moved out of function definitions
    (and extracted from kwargs) after one cycle.

    Parameters
    ----------
    errorbar : Any
        The errorbar parameter value.
    ci : Any
        The deprecated ci parameter value.
    stacklevel : int, optional
        Stack level for the warning.

    Returns
    -------
    Any
        The updated errorbar value.

    Examples
    --------
    >>> errorbar = _deprecate_ci(errorbar, ci)
    """
    from seaborn._core.typing import deprecated

    if ci is not deprecated and ci != "deprecated":
        if ci is None:
            errorbar = None
        elif ci == "sd":
            errorbar = "sd"
        else:
            errorbar = ("ci", ci)
        msg = (
            "\n\nThe `ci` parameter is deprecated. "
            f"Use `errorbar={repr(errorbar)}` for the same effect.\n"
        )
        warnings.warn(msg, FutureWarning, stacklevel=stacklevel + 1)

    return errorbar


def _handle_ignored_param(
    param: str,
    value: Any,
    *,
    reason: str,
    warning_type: type = UserWarning,
    stacklevel: int = 2,
) -> None:
    """Warn when a parameter is ignored in a certain context.

    Parameters
    ----------
    param : str
        Name of the parameter being ignored.
    value : Any
        The value passed (only warns if not None).
    reason : str
        Explanation of why the parameter is ignored.
    warning_type : type, optional
        Warning class to use.
    stacklevel : int, optional
        Stack level for the warning.

    Examples
    --------
    >>> _handle_ignored_param("units", units,
    ...                       reason="has no effect with kind='scatter'")
    """
    if value is not None:
        msg = f"The `{param}` parameter {reason}."
        warnings.warn(msg, warning_type, stacklevel=stacklevel + 1)
