"""Unified parameter validation and deprecation helpers.

This module provides a consistent interface for parameter validation,
deprecation warnings, and error handling across seaborn's plotting functions.

Each helper generates its message from **named styles** and structured
parameters — never from a free-form ``message`` string.  Adding a new
parameter deprecation or check only requires picking the right style and
passing the corresponding structured fields.

History / compatibility
-----------------------
Every style corresponds to an exact historical message format (warning text,
error type, warning category).  Calling code picks the style whose historical
behavior matches what it needs, so public-facing output never changes even as
the helper layer evolves.

Helper categories
-----------------
* ``_deprecate_param``       – parameter renamed / migrated / removed
* ``_check_figure_level_ax`` – ``ax=`` in a figure-level function
* ``_check_mutually_exclusive`` – at most one of a set may be truthy
* ``_check_argument``        – enum / allow-list value check
* ``_handle_ignored_param``  – param has no effect in current context
* ``_deprecate_ci``          – ``ci`` → ``errorbar`` convenience wrapper
* ``_warn_singular``         – singular-data KDE / variance warning
* ``_warn_deprecated_function`` – whole-function deprecation (e.g. distplot)
* ``_check_required_param``  – required param is missing
* ``_check_param_constraint`` – generic conditional ValueError/TypeError
"""
import textwrap
import warnings
from typing import Any, Iterable, Mapping, Optional, Sequence


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


# ---------------------------------------------------------------------------
# Deprecated parameter handling
# ---------------------------------------------------------------------------

def _deprecate_param(
    param: str,
    value: Any,
    *,
    style: str = "rename",
    new_param: Optional[str] = None,
    new_value: Any = None,
    set_param: Optional[str] = None,
    target: Optional[Mapping] = None,
    target_key: Optional[str] = None,
    version: str = "0.14.0",
    warning_type: Optional[type] = None,
    stacklevel: int = 2,
    error_type: Optional[type] = None,
    suggestion: Optional[str] = None,
    context: Optional[str] = None,
) -> Any:
    """Handle a deprecated parameter; message is chosen by *style*.

    Available styles
    ----------------
    ``"migrate_dict"``
        Migrate a signature-level param into a dict kwarg (e.g. ``sharex`` →
        ``facet_kws``).  Message::

            `{param}` is deprecated from the function signature.
            Please update your code to pass it using `{target_key}`.

    ``"removed"``
        Parameter has been fully removed — raise *error_type*.  Message::

            `{param}` has been removed (replaced by `{new_param}`);
            please update your code.

    ``"dedented_rename"``
        Heavy-form rename with ``textwrap.dedent`` (kdeplot ``bw`` / ``kernel``
        / ``shade_lowest`` style).  Uses UserWarning.

    ``"dedented_shade"``
        Like ``dedented_rename`` but uses FutureWarning (``shade`` → ``fill``
        soft-deprecation style).

    ``"dedented_generic"``
        Heavy-form generic deprecation (rugplot ``a`` / ``axis`` style).
        Message mentions "has been replaced".

    ``"dedented_vertical"``
        Vertical-axis deprecation with custom action text (kdeplot / rugplot
        ``vertical`` style).  *suggestion* carries the "assigning data to..."
        clause.

    ``"brief_renamed"``
        Light-form rename (violin ``scale`` → ``density_norm`` style).
        "\n\n...v{version}. Pass `{new_param}={new_value!r}` for the same effect."
        Uses FutureWarning.

    ``"brief_replaced"``
        Light-form replace (violin ``scale_hue`` → ``common_norm`` style).
        "has been replaced" instead of "has been renamed".

    ``"palette_color_trick"``
        ``color=`` seeding a gradient palette (categorical backcompat).
        "\n\nSetting a gradient palette using color= is deprecated...\n"

    ``"palette_no_hue``
        Passing ``palette`` without ``hue`` (categorical backcompat).
        "\n\nPassing `palette` without assigning `hue` is deprecated...\n"
        *context* gives the orient variable name.

    ``"capsize_none"``
        ``capsize=None`` → ``capsize=0`` (categorical backcompat).

    ``"err_kwarg"``
        ``errcolor`` / ``errwidth`` → ``err_kws`` (categorical backcompat).

    ``"point_scale"``
        ``scale=`` deprecation in pointplot.

    ``"point_join"``
        ``join=`` deprecation in pointplot.  *suggestion* is appended when
        the caller wants to show the "remove the line" hint.

    ``"gray_color"``
        ``linecolor="gray"`` → ``"auto"`` (categorical backcompat).

    Parameters
    ----------
    param : str
        Name of the deprecated parameter.
    value : Any
        The value passed.  If *None* or the ``deprecated`` sentinel, the
        function returns immediately with no warning.
    style : str
        Named message format (see above).
    new_param : str, optional
        Name of the replacement parameter.
    new_value : Any, optional
        Override value for the new parameter.  Defaults to *value*.
    target : Mapping, optional
        Dict to update with ``target_key → new_value``.
    target_key : str, optional
        Key in *target*.  Defaults to *new_param* or *param*.
    version : str, optional
        Version when the parameter will be / was removed.
    warning_type : type, optional
        Warning category.  May be overridden by some styles.
    stacklevel : int, optional
        Base stack level; the helper adds +1 internally.
    error_type : type, optional
        For ``"removed"`` style: exception type to raise.
    suggestion : str, optional
        Extra hint text inserted into the message (style-dependent).
    context : str, optional
        Additional context (style-dependent — e.g. orient variable name).

    Returns
    -------
    Any
        *new_value* if a warning was issued, else *None*.
    """
    from seaborn._core.typing import deprecated as _deprecated_sentinel

    _wt_override = warning_type

    # Guard: skip if value is the "not provided" sentinel
    if value is None or value is _deprecated_sentinel:
        return None
    if isinstance(value, str) and value == "deprecated":
        return None

    if new_value is None:
        new_value = value

    key = target_key or new_param or param
    if target is not None:
        target[key] = new_value

    # ---- Style dispatch ----------------------------------------------------

    if style == "migrate_dict":
        msg = (
            f"`{param}` is deprecated from the function signature. "
            f"Please update your code to pass it using `{key}`."
        )
        warning_type = UserWarning

    elif style == "removed":
        if new_param is not None:
            msg = (
                f"`{param}` has been removed (replaced by `{new_param}`); "
                "please update your code."
            )
        else:
            msg = f"`{param}` has been removed; please update your code."
        if error_type is None:
            error_type = TypeError
        raise error_type(msg)

    elif style == "dedented_rename":
        suggestion_part = suggestion or (
            "but please see the docs for the new parameters "
            "and update your code."
        )
        msg = textwrap.dedent(f"""\n
            The `{param}` parameter is deprecated in favor of `{new_param}`.
            Setting `{new_param}={new_value}`, {suggestion_part}
            This will become an error in seaborn v{version}.
            """)
        warning_type = UserWarning

    elif style == "dedented_shade":
        msg = textwrap.dedent(f"""\n
            `{param}` is now deprecated in favor of `{new_param}`; setting `{new_param}={new_value}`.
            This will become an error in seaborn v{version}; please update your code.
            """)
        warning_type = FutureWarning

    elif style == "dedented_replaced":
        msg = textwrap.dedent(f"""\n
            `{param}` has been replaced by `{new_param}`; setting `{new_param}={new_value}.
            This will become an error in seaborn v{version}; please update your code.
            """)
        warning_type = UserWarning

    elif style == "dedented_generic":
        suggest = suggestion or "use the new API instead"
        msg = textwrap.dedent(f"""\n
            The `{param}` parameter has been replaced; {suggest}
            Please update your code; This will become an error in seaborn v{version}.
            """)
        warning_type = UserWarning

    elif style == "dedented_generic_deprecated":
        suggest = suggestion or "use the new API instead"
        msg = textwrap.dedent(f"""\n
            The `{param}` parameter has been deprecated; {suggest}
            Please update your code; this will become an error in seaborn v{version}.
            """)
        warning_type = UserWarning

    elif style == "dedented_vertical":
        action = suggestion or "keeping the current orientation."
        msg = textwrap.dedent(f"""\n
            The `{param}` parameter is deprecated; {action}
            This will become an error in seaborn v{version}; please update your code.
            """)
        warning_type = UserWarning

    elif style == "dedented_bw_kde":
        set_p = set_param or new_param or param
        msg = textwrap.dedent(f"""\n
            The `{param}` parameter is deprecated in favor of {new_param}.
            Setting `{set_p}={new_value}`, but please see the docs for the new parameters
            and update your code. This will become an error in seaborn v{version}.
            """)
        warning_type = UserWarning

    elif style == "dedented_bw_violin":
        set_p = set_param or new_param or param
        suggestion_part = suggestion or (
            "but please see docs for the new parameters "
            "and update your code."
        )
        msg = textwrap.dedent(f"""\n
            The `{param}` parameter is deprecated in favor of {new_param}.
            Setting `{set_p}={new_value!r}`, {suggestion_part}
            This will become an error in seaborn v{version}.
            """)
        warning_type = FutureWarning

    elif style == "dedented_kernel_kde":
        msg = textwrap.dedent(f"""\n
            {suggestion}
            This will become an error in seaborn v{version}; please update your code.
            """)
        warning_type = UserWarning

    elif style == "brief_renamed":
        suggestion_part = suggestion or "for the same effect."
        msg = (
            f"\n\nThe `{param}` parameter has been renamed and will be removed "
            f"in v{version}. Pass `{new_param}={new_value!r}` {suggestion_part}"
        )
        warning_type = FutureWarning

    elif style == "brief_replaced":
        suggestion_part = suggestion or "for the same effect."
        msg = (
            f"\n\nThe `{param}` parameter has been replaced and will be removed "
            f"in v{version}. Pass `{new_param}={new_value!r}` {suggestion_part}"
        )
        warning_type = FutureWarning

    elif style == "palette_color_trick":
        palette_val = suggestion or f"dark:{value}"
        msg = (
            f"\n\nSetting a gradient palette using color= is deprecated "
            f"and will be removed in v{version}. "
            f"Set `palette='{palette_val}'` for the same effect.\n"
        )
        warning_type = FutureWarning

    elif style == "palette_no_hue":
        orient = context or "x"
        msg = (
            f"\n\nPassing `palette` without assigning `hue` is deprecated "
            f"and will be removed in v{version}. "
            f"Assign the `{orient}` variable to `hue` "
            f"and set `legend=False` for the same effect.\n"
        )
        warning_type = FutureWarning

    elif style == "capsize_none":
        msg = (
            f"\n\nPassing `capsize=None` is deprecated and will be removed "
            f"in v{version}. Pass `capsize=0` to disable caps.\n"
        )
        warning_type = FutureWarning

    elif style == "err_kwarg":
        target_key_display = suggestion or f"err_kws={{...}}"
        msg = (
            f"\n\nThe `{param}` parameter is deprecated. "
            f"And will be removed in v{version}. Pass `{target_key_display}` instead.\n"
        )
        warning_type = FutureWarning

    elif style == "point_scale":
        msg = (
            f"\n\nThe `scale` parameter is deprecated and will be removed "
            f"in v{version}. "
            "You can now control the size of each plot element using matplotlib "
            "`Line2D` parameters (e.g., `linewidth`, `markersize`, etc.).\n"
        )
        warning_type = UserWarning

    elif style == "point_join":
        msg = (
            f"\n\nThe `join` parameter is deprecated and will be removed "
            f"in v{version}."
        )
        if suggestion:
            msg += f" {suggestion}"
        msg += "\n"
        warning_type = UserWarning

    elif style == "gray_color":
        msg = (
            'Use "auto" to set automatic grayscale colors. '
            f'From v{version}, "gray" will default to matplotlib\'s definition.'
        )
        warning_type = FutureWarning

    else:
        raise ValueError(f"Unknown deprecation style: {style!r}")

    if _wt_override is not None:
        warning_type = _wt_override

    warnings.warn(msg, warning_type, stacklevel=stacklevel + 1)
    return new_value


# ---------------------------------------------------------------------------
# Figure-level ``ax=`` warning
# ---------------------------------------------------------------------------

def _check_figure_level_ax(
    func_name: str,
    kwargs: dict,
    *,
    kind: Optional[str] = None,
    style: str = "default",
    stacklevel: int = 2,
) -> None:
    """Warn and remove ``ax`` when passed to a figure-level function.

    Styles
    ------
    ``"default"`` (relplot style)
        ``relplot is a figure-level function and does not accept the `ax`
        parameter. You may wish to try scatterplot``

    ``"catplot"``
        ``catplot is a figure-level function and does not accept target axes.
        You may wish to try stripplot``

    ``"displot"``
        ```displot` is a figure-level function and does not accept the ax=
        parameter. You may wish to try histplot.``  (Note backticks around
        function name and ``ax=`` format.)
    """
    if "ax" not in kwargs:
        return

    axes_func = f"{kind}plot" if kind is not None else "the axes-level function"

    if style == "default":
        msg = (
            f"{func_name} is a figure-level function and does not accept "
            f"the `ax` parameter. You may wish to try {axes_func}"
        )
    elif style == "catplot":
        msg = (
            f"{func_name} is a figure-level function and does not accept "
            f"target axes. You may wish to try {axes_func}"
        )
    elif style == "displot":
        msg = (
            f"`{func_name}` is a figure-level function and does not accept "
            f"the ax= parameter. You may wish to try {axes_func}."
        )
    else:
        raise ValueError(f"Unknown figure-level ax style: {style!r}")

    warnings.warn(msg, UserWarning, stacklevel=stacklevel + 1)
    kwargs.pop("ax")


# ---------------------------------------------------------------------------
# Mutually exclusive parameters
# ---------------------------------------------------------------------------

def _check_mutually_exclusive(
    params: Sequence[tuple],
    *,
    style: str = "default",
    func_name: Optional[str] = None,
    error_type: type = ValueError,
    stacklevel: int = 2,
) -> None:
    """Raise if more than one parameter in *params* is truthy.

    Styles
    ------
    ``"default"``
        ``Mutually exclusive parameters: x, y.``

    ``"regression_options"``
        ``Mutually exclusive regression options.``

    ``"cannot_pass_both"``
        ``Cannot pass values for both `x` and `y`.``
        (Works with any pair — lists the first two active names.)
    """
    active = [(name, val) for name, val in params if val]
    if len(active) <= 1:
        return

    names = ", ".join(name for name, _ in active)

    if style == "default":
        msg = f"Mutually exclusive parameters: {names}."
    elif style == "regression_options":
        msg = "Mutually exclusive regression options."
    elif style == "cannot_pass_both":
        # Classic phrasing — only makes sense for pairs
        first, second = active[0][0], active[1][0]
        msg = f"Cannot pass values for both `{first}` and `{second}`."
    else:
        raise ValueError(f"Unknown mutually-exclusive style: {style!r}")

    raise error_type(msg)


# ---------------------------------------------------------------------------
# Argument value validation
# ---------------------------------------------------------------------------

def _check_argument(
    param: str,
    options: Iterable[Any],
    value: Any,
    *,
    style: str = "default",
    prefix: bool = False,
    error_type: type = ValueError,
) -> Any:
    """Raise if *value* for *param* is not in *options*.

    Styles
    ------
    ``"default"``
        ``The value for `{param}` must be one of {options}, but {value!r} was passed.``

    ``"must_be"``
        `` `{param}` must be 'opt1' or 'opt2', not {value}``

    ``"must_be_either"``
        `` `{param}` must be either 'x' or 'y', not {value!r}.``

    ``"kind_not_recognized"``
        ``Plot kind {value} not recognized``

    ``"invalid_kind_list"``
        ``Invalid `kind`: {value!r}. Options are 'a', 'b', and 'c'.``
        (Oxford comma + "and" before last item.)
    """
    options = list(options)

    if prefix and value is not None:
        failure = not any(
            value.startswith(p) for p in options if isinstance(p, str)
        )
    else:
        failure = value not in options

    if not failure:
        return value

    if style == "default":
        msg = (
            f"The value for `{param}` must be one of {options}, "
            f"but {repr(value)} was passed."
        )
    elif style == "must_be":
        opt_str = "' or '".join(str(o) for o in options)
        msg = f"`{param}` must be '{opt_str}', not {value}"
    elif style == "must_be_either":
        opt_str = "' or '".join(str(o) for o in options)
        msg = f"`{param}` must be either '{opt_str}', not {value!r}."
    elif style == "kind_not_recognized":
        msg = f"Plot kind {value} not recognized"
    elif style == "invalid_kind_list":
        if len(options) == 1:
            opt_str = repr(options[0])
        elif len(options) == 2:
            opt_str = f"{options[0]!r} and {options[1]!r}"
        else:
            opt_str = ", ".join(repr(o) for o in options[:-1])
            opt_str += f", and {options[-1]!r}"
        msg = f"Invalid `{param}`: {value!r}. Options are {opt_str}."
    else:
        raise ValueError(f"Unknown check_argument style: {style!r}")

    raise error_type(msg)


# ---------------------------------------------------------------------------
# ci → errorbar deprecation
# ---------------------------------------------------------------------------

def _deprecate_ci(
    errorbar: Any,
    ci: Any,
    *,
    stacklevel: int = 2,
) -> Any:
    """Convert deprecated ``ci=`` to ``errorbar=`` with a FutureWarning.

    Message (exact historical):
        ``\n\nThe `ci` parameter is deprecated. Use `errorbar={repr(errorbar)}` for the same effect.\n``
    """
    from seaborn._core.typing import deprecated

    if ci is not deprecated and (not isinstance(ci, str) or ci != "deprecated"):
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


# ---------------------------------------------------------------------------
# Ignored-parameter warning
# ---------------------------------------------------------------------------

def _handle_ignored_param(
    param: str,
    value: Any,
    *,
    style: str = "has_no_effect",
    context: Optional[str] = None,
    reason: Optional[str] = None,
    warning_type: type = UserWarning,
    stacklevel: int = 2,
) -> None:
    """Warn when *param* (non-None *value*) is ignored.

    Styles
    ------
    ``"has_no_effect"``
        ``The `{param}` parameter has no effect with {context}.``

    ``"ignored_when"``
        ``{param} parameter ignored when using {context}.``
        (No backticks around param name.)

    ``"bins_auto_weights"``
        `` `bins` cannot be 'auto' when using weights. Setting `bins=10`,
        but you will likely want to adjust.``
        (*reason* is the full second sentence; *value* is unused for the
        check — caller passes the string ``"auto"`` or similar.)
    """
    if value is None:
        return

    if style == "has_no_effect":
        msg = f"The `{param}` parameter has no effect with {context}."
    elif style == "ignored_when":
        msg = f"{param} parameter ignored when using {context}."
    elif style == "bins_auto_weights":
        msg = (
            f"`{param}` cannot be 'auto' when using weights. "
            f"{reason}"
        )
    else:
        raise ValueError(f"Unknown ignored-param style: {style!r}")

    warnings.warn(msg, warning_type, stacklevel=stacklevel + 1)


# ---------------------------------------------------------------------------
# Singular-data warning
# ---------------------------------------------------------------------------

def _warn_singular(
    context: str = "univariate",
    *,
    stacklevel: int = 2,
) -> None:
    """Warn about singular data (zero variance / perfect covariance).

    *context* selects the exact message:
    * ``"univariate"`` — hist/kde1d style
    * ``"bivariate"``  — kde2d style
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


# ---------------------------------------------------------------------------
# Whole-function deprecation
# ---------------------------------------------------------------------------

def _warn_deprecated_function(
    func_name: str,
    *,
    style: str = "default",
    version: str = "0.14.0",
    suggestion: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """Warn that an entire function is deprecated.

    Styles
    ------
    ``"default"``
        Heavy dedented format with replacement suggestion (distplot style).
        *suggestion* holds the full "Please adapt your code..." paragraph.
    """
    if style == "default":
        body = suggestion or "Please update your code."
        msg = textwrap.dedent(f"""

        `{func_name}` is a deprecated function and will be removed in seaborn v{version}.

        {body}
        """)
    else:
        raise ValueError(f"Unknown deprecated-function style: {style!r}")

    warnings.warn(msg, UserWarning, stacklevel=stacklevel + 1)


# ---------------------------------------------------------------------------
# Required-parameter check
# ---------------------------------------------------------------------------

def _check_required_param(
    param: str,
    value: Any,
    *,
    style: str = "missing_kwarg",
    error_type: type = TypeError,
    condition: Optional[str] = None,
    func_name: Optional[str] = None,
) -> Any:
    """Raise if a required parameter is missing (None).

    Styles
    ------
    ``"missing_kwarg"``
        ``Missing required keyword argument `{param}`.``

    ``"when_condition"``
        ``Must pass `{param}` when {condition}.``
        (ValueError, not TypeError.)
    """
    if value is not None:
        return value

    if style == "missing_kwarg":
        msg = f"Missing required keyword argument `{param}`."
    elif style == "when_condition":
        msg = f"Must pass `{param}` when {condition}."
        if error_type is TypeError:
            error_type = ValueError
    else:
        raise ValueError(f"Unknown required-param style: {style!r}")

    raise error_type(msg)


# ---------------------------------------------------------------------------
# Generic parameter constraint check
# ---------------------------------------------------------------------------

def _check_param_constraint(
    constraint: str,
    *,
    style: str = "plain",
    param: Optional[str] = None,
    value: Any = None,
    error_type: type = ValueError,
) -> None:
    """Raise for a generic parameter constraint.

    This is the escape hatch for one-off checks that don't fit the other
    helpers.  Even here, the message is always built from structured fields
    so callers never hand-write a warning/error string.

    Styles
    ------
    ``"plain"``
        Just *constraint*.

    ``"with_param"``
        ``{constraint} (parameter: `{param}`)``

    ``"with_value"``
        ``{constraint} (got `{param}`={value!r})``

    ``"statsmodels"``
        `` `{param}=True` requires statsmodels, an optional dependency,
        to be installed.``  (*constraint* is unused.)
    """
    if style == "plain":
        msg = constraint
    elif style == "with_param":
        msg = f"{constraint} (parameter: `{param}`)"
    elif style == "with_value":
        msg = f"{constraint} (got `{param}`={value!r})"
    elif style == "statsmodels":
        msg = (
            f"`{param}=True` requires statsmodels, an optional dependency, "
            "to be installed."
        )
        error_type = RuntimeError
    else:
        raise ValueError(f"Unknown param-constraint style: {style!r}")

    raise error_type(msg)
