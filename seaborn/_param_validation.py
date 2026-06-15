"""Unified parameter validation and deprecation helpers.

This module provides a consistent interface for parameter validation,
deprecation warnings, and error handling across seaborn's plotting functions.

Each helper generates its message from structured data, so that adding a new
parameter only requires passing the right keyword arguments — never hand-writing
a warning or error string in the calling module.

Helper categories
-----------------
1. **_deprecate_param**          – parameter renamed / migrated / removed
2. **_check_mutually_exclusive** – at most one of a set may be truthy
3. **_check_figure_level_ax**    – ``ax=`` in a figure-level function
4. **_check_argument**           – enum / allow-list value check
5. **_handle_ignored_param**     – param has no effect in current context
6. **_deprecate_ci**             – ci→errorbar convenience wrapper
7. **_warn_singular**            – singular-data KDE / variance warning
8. **_warn_deprecated_function** – whole-function deprecation (e.g. distplot)
9. **_check_required_param**     – required param is missing
10. **_check_param_constraint**  – generic conditional ValueError/TypeError
"""
import textwrap
import warnings
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


__all__ = [
    "_check_argument",
    "_check_figure_level_ax",
    "_check_mutually_exclusive",
    "_check_param_constraint",
    "_check_required_param",
    "_deprecate_ci",
    "_deprecate_param",
    "_handle_ignored_param",
    "_warn_deprecated_function",
    "_warn_singular",
]


def _deprecate_param(
    param: str,
    value: Any,
    *,
    new_param: Optional[str] = None,
    new_value: Any = None,
    target: Optional[Mapping] = None,
    target_key: Optional[str] = None,
    since: Optional[str] = None,
    remove_version: str = "0.14.0",
    warning_type: type = UserWarning,
    stacklevel: int = 2,
    suggest: Optional[str] = None,
    error_type: Optional[type] = None,
) -> Any:
    """Handle a deprecated parameter; message is auto-generated.

    The message is always built from structured fields so callers never
    hand-write a format string.  Available message shapes:

    * **Rename** (new_param given, no target):
      ``The `{param}` parameter is deprecated in favor of `{new_param}`;
      setting `{new_param}={new_value}`.
      This will become an error in seaborn v{remove_version};
      please update your code.``

    * **Migrate to dict** (target given, no new_param):
      ``{param}` is deprecated from the function signature.
      Please update your code to pass it using `{target_key}`.``

    * **Removed** (error_type given):
      ``The `{param}` parameter has been removed (replaced by `{new_param}`);
      please update your code.``

    * **Generic** (neither new_param nor target):
      ``The `{param}` parameter is deprecated. {suggest}
      This will become an error in seaborn v{remove_version};
      please update your code.``

    Parameters
    ----------
    param : str
        Name of the deprecated parameter.
    value : Any
        The value passed.  If *None* (or the ``deprecated`` sentinel),
        the function returns immediately with no warning.
    new_param : str, optional
        Name of the replacement parameter.
    new_value : Any, optional
        Override value for the new parameter.  Defaults to *value*.
    target : Mapping, optional
        Dict to update with ``target_key → new_value``.
    target_key : str, optional
        Key in *target*.  Defaults to *new_param* or *param*.
    since : str, optional
        Version when the deprecation started.
    remove_version : str, optional
        Version when it becomes a hard error.
    warning_type : type, optional
        UserWarning or FutureWarning.
    stacklevel : int, optional
        Stack level passed to ``warnings.warn``.
    suggest : str, optional
        Extra guidance appended to generic messages (e.g.
        ``"Setting `bw_method=scott`, but please see the docs for the new parameters."``).
    error_type : type, optional
        If given, *raise* instead of warn (parameter fully removed).

    Returns
    -------
    Any
        *new_value* if a warning was issued, else *None*.
    """
    from seaborn._core.typing import deprecated as _deprecated_sentinel

    if value is None or value is _deprecated_sentinel:
        return None

    if isinstance(value, str) and value == "deprecated":
        return None

    if new_value is None:
        new_value = value

    key = target_key or new_param or param
    if target is not None:
        target[key] = new_value

    if error_type is not None:
        if new_param is not None:
            msg = (
                f"The `{param}` parameter has been removed (replaced by "
                f"`{new_param}`); please update your code."
            )
        else:
            msg = (
                f"The `{param}` parameter has been removed; "
                "please update your code."
            )
        raise error_type(msg)

    if new_param is not None and target is None:
        suggest_part = f" {suggest}" if suggest else ""
        msg = (
            f"The `{param}` parameter is deprecated in favor of `{new_param}`; "
            f"setting `{new_param}={new_value}`.{suggest_part}\n"
            f"This will become an error in seaborn v{remove_version}; "
            "please update your code."
        )
    elif target is not None and new_param is None:
        msg = (
            f"`{param}` is deprecated from the function signature. "
            f"Please update your code to pass it using `{key}`."
        )
    else:
        suggest_part = f" {suggest}" if suggest else ""
        msg = (
            f"The `{param}` parameter is deprecated.{suggest_part}\n"
            f"This will become an error in seaborn v{remove_version}; "
            "please update your code."
        )

    warnings.warn(msg, warning_type, stacklevel=stacklevel + 1)
    return new_value


def _check_figure_level_ax(
    func_name: str,
    kwargs: dict,
    *,
    kind: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """Warn and remove ``ax`` when passed to a figure-level function.

    The message is always auto-generated from *func_name* and *kind*:

    * With *kind*: ``{func_name}` is a figure-level function and does not
      accept the `ax` parameter. You may wish to try {kind}plot.``
    * Without *kind*: ``Ignoring `ax`; {func_name}` is a figure-level function.``
    """
    if "ax" not in kwargs:
        return

    if kind is not None:
        axes_func = f"{kind}plot"
        msg = (
            f"`{func_name}` is a figure-level function and does not accept "
            f"the `ax` parameter. You may wish to try {axes_func}."
        )
    else:
        msg = f"Ignoring `ax`; {func_name}` is a figure-level function."

    warnings.warn(msg, UserWarning, stacklevel=stacklevel + 1)
    kwargs.pop("ax")


def _check_mutually_exclusive(
    params: Sequence[Tuple[str, Any]],
    *,
    func_name: Optional[str] = None,
    error_type: type = ValueError,
    stacklevel: int = 2,
) -> None:
    """Raise if more than one parameter in *params* is truthy.

    Message: ``Mutually exclusive {label}: {names}.``
    where *label* defaults to "parameters" and *names* lists the active ones.
    """
    active = [(name, val) for name, val in params if val]
    if len(active) <= 1:
        return

    names = ", ".join(name for name, _ in active)
    prefix = f"In {func_name}: " if func_name else ""
    msg = f"{prefix}Mutually exclusive parameters: {names}."
    raise error_type(msg)


def _check_argument(
    param: str,
    options: Iterable[Any],
    value: Any,
    *,
    prefix: bool = False,
    error_type: type = ValueError,
) -> Any:
    """Raise if *value* for *param* is not in *options*.

    Message: ``The value for `{param}` must be one of {options}, but {value!r} was passed.``
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
    """Convert deprecated ``ci=`` to ``errorbar=`` with a FutureWarning.

    Message: ``The `ci` parameter is deprecated. Use `errorbar={repr(errorbar)}` for the same effect.``
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
    """Warn when *param* (non-None *value*) is ignored in the current context.

    Message: ``The `{param}` parameter {reason}.``
    """
    if value is not None:
        msg = f"The `{param}` parameter {reason}."
        warnings.warn(msg, warning_type, stacklevel=stacklevel + 1)


def _warn_singular(
    context: str = "univariate",
    *,
    stacklevel: int = 2,
) -> None:
    """Warn about singular data (zero variance / perfect covariance).

    Message shape is determined by *context*:
    * ``"univariate"``: ``Dataset has 0 variance; skipping density estimate. Pass `warn_singular=False` to disable this warning.``
    * ``"bivariate"``:  ``KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.``
    """
    if context == "univariate":
        msg = (
            "Dataset has 0 variance; skipping density estimate. "
            "Pass `warn_singular=False` to disable this warning."
        )
    else:
        msg = (
            "KDE cannot be estimated (0 variance or perfect covariance). "
            "Pass `warn_singular=False` to disable this warning."
        )
    warnings.warn(msg, UserWarning, stacklevel=stacklevel + 1)


def _warn_deprecated_function(
    func_name: str,
    *,
    replacement: Optional[str] = None,
    removal_version: str = "0.14.0",
    extra_guidance: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """Warn that an entire function is deprecated.

    Message: ``{func_name}` is a deprecated function and will be removed in seaborn v{removal_version}. {extra_guidance}``
    """
    parts = [
        f"`{func_name}` is a deprecated function and will be removed "
        f"in seaborn v{removal_version}.",
    ]
    if extra_guidance:
        parts.append(extra_guidance)
    msg = " ".join(parts)
    warnings.warn(msg, UserWarning, stacklevel=stacklevel + 1)


def _check_required_param(
    param: str,
    value: Any,
    *,
    func_name: Optional[str] = None,
    error_type: type = TypeError,
    condition: Optional[str] = None,
) -> Any:
    """Raise if a required parameter is missing (None).

    Message shapes:
    * With *condition*: ``Must pass `{param}` when {condition}.``
    * With *func_name*: ``Missing required keyword argument `{param}` in {func_name}.``
    * Default: ``Missing required keyword argument `{param}`.``
    """
    if value is not None:
        return value

    if condition is not None:
        msg = f"Must pass `{param}` when {condition}."
    elif func_name is not None:
        msg = f"Missing required keyword argument `{param}` in `{func_name}`."
    else:
        msg = f"Missing required keyword argument `{param}`."

    raise error_type(msg)


def _check_param_constraint(
    constraint: str,
    *,
    param: Optional[str] = None,
    value: Any = None,
    error_type: type = ValueError,
) -> None:
    """Raise a ValueError/TypeError for a generic parameter constraint.

    This is the escape hatch for one-off checks that don't fit the other
    helpers.  The message is always auto-generated from structured fields:

    * With *param* and *value*: ``{constraint} (got `{param}`={value!r})``
    * With *param* only:       ``{constraint} (parameter: `{param}`)``
    * Neither:                 ``{constraint}``
    """
    if param is not None and value is not None:
        msg = f"{constraint} (got `{param}`={value!r})"
    elif param is not None:
        msg = f"{constraint} (parameter: `{param}`)"
    else:
        msg = constraint
    raise error_type(msg)
