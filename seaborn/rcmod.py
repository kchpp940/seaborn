"""Control plot style and scaling using the matplotlib rcParams interface."""
import copy
import functools
import warnings
import matplotlib as mpl
from cycler import cycler
from . import palettes


__all__ = ["set_theme", "set", "reset_defaults", "reset_orig",
           "axes_style", "set_style", "plotting_context", "set_context",
           "set_palette",
           "register_theme_profile", "get_theme_profile",
           "list_theme_profiles", "unregister_theme_profile"]


_theme_profile_registry: dict = {}


_style_keys = [

    "axes.facecolor",
    "axes.edgecolor",
    "axes.grid",
    "axes.axisbelow",
    "axes.labelcolor",

    "figure.facecolor",

    "grid.color",
    "grid.linestyle",

    "text.color",

    "xtick.color",
    "ytick.color",
    "xtick.direction",
    "ytick.direction",
    "lines.solid_capstyle",

    "patch.edgecolor",
    "patch.force_edgecolor",

    "image.cmap",
    "font.family",
    "font.sans-serif",

    "xtick.bottom",
    "xtick.top",
    "ytick.left",
    "ytick.right",

    "axes.spines.left",
    "axes.spines.bottom",
    "axes.spines.right",
    "axes.spines.top",

]

_context_keys = [

    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "legend.title_fontsize",

    "axes.linewidth",
    "grid.linewidth",
    "lines.linewidth",
    "lines.markersize",
    "patch.linewidth",

    "xtick.major.width",
    "ytick.major.width",
    "xtick.minor.width",
    "ytick.minor.width",

    "xtick.major.size",
    "ytick.major.size",
    "xtick.minor.size",
    "ytick.minor.size",

]

_VALID_PROFILE_KEYS = frozenset({
    "context", "style", "palette", "font", "font_scale",
    "color_codes", "rc",
})

_STYLE_KEY_SET = frozenset(_style_keys)
_CONTEXT_KEY_SET = frozenset(_context_keys)


def _validate_rc_dict(rc, category=None):
    """Validate a dict of rc parameters.

    Parameters
    ----------
    rc : dict
        Dictionary of rc parameter mappings to validate.
    category : {"style", "context", None}
        If set, only allow keys in that category and warn about others.

    Returns
    -------
    validated : dict
        The validated (possibly filtered) rc dict.

    Raises
    ------
    TypeError
        If rc is not a dict.
    ValueError
        If no keys are valid matplotlib rcParams.
    """
    if rc is None:
        return {}
    if not isinstance(rc, dict):
        raise TypeError(
            f"`rc` must be a dict of matplotlib rcParams, got {type(rc).__name__}"
        )

    if category == "style":
        invalid = {k for k in rc if k not in _STYLE_KEY_SET}
        if invalid:
            warnings.warn(
                f"Ignoring {len(invalid)} rc key(s) not in style definition: "
                f"{', '.join(sorted(invalid))}",
                UserWarning,
                stacklevel=3,
            )
        return {k: v for k, v in rc.items() if k in _STYLE_KEY_SET}

    if category == "context":
        invalid = {k for k in rc if k not in _CONTEXT_KEY_SET}
        if invalid:
            warnings.warn(
                f"Ignoring {len(invalid)} rc key(s) not in context definition: "
                f"{', '.join(sorted(invalid))}",
                UserWarning,
                stacklevel=3,
            )
        return {k: v for k, v in rc.items() if k in _CONTEXT_KEY_SET}

    bad_keys = [k for k in rc if k not in mpl.rcParams]
    if bad_keys:
        raise ValueError(
            f"Unrecognized matplotlib rcParam key(s): {', '.join(sorted(bad_keys))}. "
            "Check matplotlib.rcParams for valid keys."
        )
    return dict(rc)


def _validate_profile_params(params):
    """Validate and normalize a theme profile parameter dict.

    A profile may contain any subset of the keys accepted by
    :func:`set_theme`: context, style, palette, font, font_scale,
    color_codes, and rc (a flat dict of matplotlib rcParams).

    Parameters
    ----------
    params : dict
        Profile parameter dictionary.

    Returns
    -------
    normalized : dict
        Normalized copy with validated values.
    """
    if not isinstance(params, dict):
        raise TypeError(
            f"Theme profile params must be a dict, got {type(params).__name__}"
        )

    unknown = frozenset(params) - _VALID_PROFILE_KEYS
    if unknown:
        raise ValueError(
            f"Theme profile contains unknown key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys are: {', '.join(sorted(_VALID_PROFILE_KEYS))}"
        )

    normalized = {}

    if "context" in params:
        ctx = params["context"]
        if not (isinstance(ctx, (str, dict)) or ctx is None):
            raise TypeError(
                "profile 'context' must be a string name, dict of rcParams, or None"
            )
        if isinstance(ctx, str) and ctx not in ["paper", "notebook", "talk", "poster"]:
            raise ValueError(
                f"profile 'context' string must be one of "
                f"paper, notebook, talk, poster; got {ctx!r}"
            )
        normalized["context"] = ctx
    else:
        normalized["context"] = "notebook"

    if "style" in params:
        sty = params["style"]
        if not (isinstance(sty, (str, dict)) or sty is None):
            raise TypeError(
                "profile 'style' must be a string name, dict of rcParams, or None"
            )
        if isinstance(sty, str) and sty not in ["white", "dark", "whitegrid", "darkgrid", "ticks"]:
            raise ValueError(
                f"profile 'style' string must be one of "
                f"white, dark, whitegrid, darkgrid, ticks; got {sty!r}"
            )
        normalized["style"] = sty
    else:
        normalized["style"] = "darkgrid"

    if "palette" in params:
        normalized["palette"] = params["palette"]
    else:
        normalized["palette"] = "deep"

    if "font" in params:
        if not isinstance(params["font"], (str, list)):
            raise TypeError("profile 'font' must be a string or list of strings")
        normalized["font"] = params["font"]
    else:
        normalized["font"] = "sans-serif"

    if "font_scale" in params:
        fs = params["font_scale"]
        if not isinstance(fs, (int, float)):
            raise TypeError("profile 'font_scale' must be a number")
        if fs <= 0:
            raise ValueError("profile 'font_scale' must be positive")
        normalized["font_scale"] = fs
    else:
        normalized["font_scale"] = 1

    if "color_codes" in params:
        if not isinstance(params["color_codes"], bool):
            raise TypeError("profile 'color_codes' must be a bool")
        normalized["color_codes"] = params["color_codes"]
    else:
        normalized["color_codes"] = True

    if "rc" in params:
        normalized["rc"] = _validate_rc_dict(params["rc"])
    else:
        normalized["rc"] = {}

    return normalized


def register_theme_profile(name, params, *, overwrite=False):
    """Register a named theme profile for reuse across plots.

    A theme profile bundles any of the parameters accepted by
    :func:`set_theme` so you can refer to them by a short string name
    in :func:`set_theme`, :func:`axes_style`, :func:`plotting_context`,
    and figure-level plotting functions.

    Registered profiles are global for the Python process and therefore
    ideal for codifying a project's visual identity.

    Parameters
    ----------
    name : str
        Unique name used to look up the profile (e.g. ``"corporate"``).
    params : dict
        Theme parameters. Recognized keys (all optional):

        - ``context`` : ``"paper"`` | ``"notebook"`` | ``"talk"`` | ``"poster"`` | dict
        - ``style`` : ``"white"`` | ``"dark"`` | ``"whitegrid"`` | ``"darkgrid"`` | ``"ticks"`` | dict
        - ``palette`` : palette name, list of colors, or seaborn palette
        - ``font`` : font family name or list of family names
        - ``font_scale`` : positive number (independent font scaling)
        - ``color_codes`` : bool (whether to remap ``"b"``, ``"g"``, ...)
        - ``rc`` : dict of additional matplotlib rcParams
    overwrite : bool, default False
        If True, replace an existing profile with the same ``name``.
        Otherwise raise ``ValueError``.

    Raises
    ------
    ValueError
        If ``name`` is already registered and ``overwrite`` is False,
        or if ``params`` contains invalid keys / values.
    TypeError
        If ``params`` is not a dict.

    Examples
    --------
    Register a custom profile and apply it globally:

    .. code:: python

        import seaborn as sns

        sns.register_theme_profile("corp", {
            "style": "white",
            "context": "talk",
            "palette": "Blues_d",
            "font": "serif",
            "rc": {"axes.linewidth": 2},
        })

        sns.set_theme(profile="corp")

    Use it inside a temporary context:

    .. code:: python

        with sns.axes_style(profile="corp"):
            ...
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Theme profile name must be a non-empty string")

    if name in ["white", "dark", "whitegrid", "darkgrid", "ticks",
                "paper", "notebook", "talk", "poster"]:
        raise ValueError(
            f"Theme profile name {name!r} collides with a built-in style/context "
            f"name; please choose a different identifier."
        )

    if name in _theme_profile_registry and not overwrite:
        raise ValueError(
            f"Theme profile {name!r} is already registered. "
            "Use overwrite=True to replace it."
        )

    normalized = _validate_profile_params(params)
    _theme_profile_registry[name] = normalized


def get_theme_profile(name):
    """Return a copy of the registered theme profile *name*.

    Parameters
    ----------
    name : str
        Name of the profile previously registered with
        :func:`register_theme_profile`.

    Returns
    -------
    profile : dict
        A copy of the stored profile parameters.

    Raises
    ------
    ValueError
        If no profile with the given name exists.

    See Also
    --------
    register_theme_profile, list_theme_profiles, unregister_theme_profile
    """
    if name not in _theme_profile_registry:
        available = list_theme_profiles()
        if available:
            raise ValueError(
                f"Unknown theme profile {name!r}. "
                f"Available profiles: {', '.join(available)}"
            )
        raise ValueError(
            f"Unknown theme profile {name!r}. "
            "No profiles are currently registered; use "
            "sns.register_theme_profile() to add one."
        )
    return copy.deepcopy(_theme_profile_registry[name])


def list_theme_profiles():
    """Return the names of all registered theme profiles.

    Returns
    -------
    names : list of str
        Alphabetically sorted list of profile names.

    See Also
    --------
    register_theme_profile, get_theme_profile, unregister_theme_profile
    """
    return sorted(_theme_profile_registry)


def unregister_theme_profile(name):
    """Remove a theme profile from the registry.

    Parameters
    ----------
    name : str
        Name of the profile to remove.

    Raises
    ------
    ValueError
        If the profile does not exist.

    See Also
    --------
    register_theme_profile, list_theme_profiles
    """
    if name not in _theme_profile_registry:
        raise ValueError(f"Cannot unregister unknown theme profile {name!r}")
    del _theme_profile_registry[name]


def _resolve_profile(profile, explicit, *, kind=None):
    """Resolve a profile (by name) and merge with explicit overrides.

    Parameters
    ----------
    profile : str, dict, or None
        Profile name (string), inline profile dict, or None.
    explicit : dict
        Explicit keyword arguments that should win over profile values.
    kind : {None, "style", "context"}
        If "style" or "context", only return the relevant subset of
        parameters.  Otherwise return the full set for ``set_theme``.

    Returns
    -------
    merged : dict
    """
    base = {
        "context": "notebook", "style": "darkgrid", "palette": "deep",
        "font": "sans-serif", "font_scale": 1, "color_codes": True, "rc": {},
    }

    if profile is None:
        resolved = base
    elif isinstance(profile, str):
        resolved = get_theme_profile(profile)
    elif isinstance(profile, dict):
        resolved = _validate_profile_params(profile)
    else:
        raise TypeError(
            f"`profile` must be a string name, dict, or None; "
            f"got {type(profile).__name__}"
        )

    merged = {**resolved}

    for k, v in explicit.items():
        if v is None:
            continue
        if k == "rc" and isinstance(v, dict):
            merged["rc"] = {**merged.get("rc", {}), **v}
        else:
            merged[k] = v

    if kind == "style":
        return {"style": merged["style"], "rc": merged["rc"]}
    if kind == "context":
        return {"context": merged["context"], "font_scale": merged["font_scale"],
                "rc": merged["rc"]}
    return merged


def set_theme(context=None, style=None, palette=None,
              font=None, font_scale=None, color_codes=None, rc=None,
              profile=None):
    """
    Set aspects of the visual theme for all matplotlib and seaborn plots.

    This function changes the global defaults for all plots using the
    matplotlib rcParams system. The themeing is decomposed into several distinct
    sets of parameter values.

    The options are illustrated in the :doc:`aesthetics <../tutorial/aesthetics>`
    and :doc:`color palette <../tutorial/color_palettes>` tutorials.

    Parameters
    ----------
    context : string or dict
        Scaling parameters, see :func:`plotting_context`.
    style : string or dict
        Axes style parameters, see :func:`axes_style`.
    palette : string or sequence
        Color palette, see :func:`color_palette`.
    font : string
        Font family, see matplotlib font manager.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    color_codes : bool
        If ``True`` and ``palette`` is a seaborn palette, remap the shorthand
        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.
    rc : dict or None
        Dictionary of rc parameter mappings to override the above.
    profile : str, dict, or None
        Name of a registered theme profile (see
        :func:`register_theme_profile`), or an inline profile dict.
        Explicit arguments above take precedence over the profile values.

    Examples
    --------

    .. include:: ../docstrings/set_theme.rst

    """
    explicit = dict(
        context=context, style=style, palette=palette,
        font=font, font_scale=font_scale, color_codes=color_codes, rc=rc,
    )
    merged = _resolve_profile(profile, explicit)

    all_rc = _validate_rc_dict(merged["rc"])
    style_rc = {k: v for k, v in all_rc.items() if k in _STYLE_KEY_SET}
    context_rc = {k: v for k, v in all_rc.items() if k in _CONTEXT_KEY_SET}
    extra_rc = {k: v for k, v in all_rc.items()
                if k not in _STYLE_KEY_SET and k not in _CONTEXT_KEY_SET}

    set_context(merged["context"], merged["font_scale"], rc=context_rc)
    set_style(merged["style"], rc={
        **({"font.family": merged["font"]} if merged["font"] is not None else {}),
        **style_rc,
    })
    set_palette(merged["palette"], color_codes=merged["color_codes"])

    if extra_rc:
        mpl.rcParams.update(extra_rc)


def set(*args, **kwargs):
    """
    Alias for :func:`set_theme`, which is the preferred interface.

    This function may be removed in the future.
    """
    set_theme(*args, **kwargs)


def reset_defaults():
    """Restore all RC params to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def reset_orig():
    """Restore all RC params to original settings (respects custom rc)."""
    from . import _orig_rc_params
    mpl.rcParams.update(_orig_rc_params)


def axes_style(style=None, rc=None, *, profile=None):
    """
    Get the parameters that control the general style of the plots.

    The style parameters control properties like the color of the background and
    whether a grid is enabled by default. This is accomplished using the
    matplotlib rcParams system.

    The options are illustrated in the
    :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    This function can also be used as a context manager to temporarily
    alter the global defaults. See :func:`set_theme` or :func:`set_style`
    to modify the global defaults for all plots.

    Parameters
    ----------
    style : None, dict, or one of {darkgrid, whitegrid, dark, white, ticks}
        A dictionary of parameters or the name of a preconfigured style.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        style dictionaries. This only updates parameters that are
        considered part of the style definition.
    profile : str, dict, or None
        Name of a registered theme profile (see
        :func:`register_theme_profile`), or an inline profile dict.
        The ``style`` and ``rc`` arguments (if given) take precedence
        over values coming from the profile.

    Examples
    --------

    .. include:: ../docstrings/axes_style.rst

    """
    if profile is not None:
        resolved = _resolve_profile(profile, {"style": style, "rc": rc},
                                    kind="style")
        style = resolved["style"]
        rc = resolved["rc"]

    if style is None:
        style_dict = {k: mpl.rcParams[k] for k in _style_keys}

    elif isinstance(style, dict):
        style_dict = style

    else:
        styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]
        if style not in styles:
            raise ValueError(f"style must be one of {', '.join(styles)}")

        # Define colors here
        dark_gray = ".15"
        light_gray = ".8"

        # Common parameters
        style_dict = {

            "figure.facecolor": "white",
            "axes.labelcolor": dark_gray,

            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": dark_gray,
            "ytick.color": dark_gray,

            "axes.axisbelow": True,
            "grid.linestyle": "-",


            "text.color": dark_gray,
            "font.family": ["sans-serif"],
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans",
                                "Bitstream Vera Sans", "sans-serif"],


            "lines.solid_capstyle": "round",
            "patch.edgecolor": "w",
            "patch.force_edgecolor": True,

            "image.cmap": "rocket",

            "xtick.top": False,
            "ytick.right": False,

        }

        # Set grid on or off
        if "grid" in style:
            style_dict.update({
                "axes.grid": True,
            })
        else:
            style_dict.update({
                "axes.grid": False,
            })

        # Set the color of the background, spines, and grids
        if style.startswith("dark"):
            style_dict.update({

                "axes.facecolor": "#EAEAF2",
                "axes.edgecolor": "white",
                "grid.color": "white",

                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.spines.right": True,
                "axes.spines.top": True,

            })

        elif style == "whitegrid":
            style_dict.update({

                "axes.facecolor": "white",
                "axes.edgecolor": light_gray,
                "grid.color": light_gray,

                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.spines.right": True,
                "axes.spines.top": True,

            })

        elif style in ["white", "ticks"]:
            style_dict.update({

                "axes.facecolor": "white",
                "axes.edgecolor": dark_gray,
                "grid.color": light_gray,

                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.spines.right": True,
                "axes.spines.top": True,

            })

        # Show or hide the axes ticks
        if style == "ticks":
            style_dict.update({
                "xtick.bottom": True,
                "ytick.left": True,
            })
        else:
            style_dict.update({
                "xtick.bottom": False,
                "ytick.left": False,
            })

    style_dict = {k: v for k, v in style_dict.items() if k in _STYLE_KEY_SET}

    # Override these settings with the provided rc dictionary
    if rc is not None:
        rc_valid = _validate_rc_dict(rc, category="style")
        style_dict.update(rc_valid)

    # Wrap in an _AxesStyle object so this can be used in a with statement
    style_object = _AxesStyle(style_dict)

    return style_object


def set_style(style=None, rc=None, *, profile=None):
    """
    Set the parameters that control the general style of the plots.

    The style parameters control properties like the color of the background and
    whether a grid is enabled by default. This is accomplished using the
    matplotlib rcParams system.

    The options are illustrated in the
    :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    See :func:`axes_style` to get the parameter values.

    Parameters
    ----------
    style : dict, or one of {darkgrid, whitegrid, dark, white, ticks}
        A dictionary of parameters or the name of a preconfigured style.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        style dictionaries. This only updates parameters that are
        considered part of the style definition.
    profile : str, dict, or None
        Name of a registered theme profile (see
        :func:`register_theme_profile`), or an inline profile dict.

    Examples
    --------

    .. include:: ../docstrings/set_style.rst

    """
    style_object = axes_style(style, rc, profile=profile)
    mpl.rcParams.update(style_object)


def plotting_context(context=None, font_scale=None, rc=None, *, profile=None):
    """
    Get the parameters that control the scaling of plot elements.

    These parameters correspond to label size, line thickness, etc. For more
    information, see the :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    The base context is "notebook", and the other contexts are "paper", "talk",
    and "poster", which are version of the notebook parameters scaled by different
    values. Font elements can also be scaled independently of (but relative to)
    the other values.

    This function can also be used as a context manager to temporarily
    alter the global defaults. See :func:`set_theme` or :func:`set_context`
    to modify the global defaults for all plots.

    Parameters
    ----------
    context : None, dict, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        context dictionaries. This only updates parameters that are
        considered part of the context definition.
    profile : str, dict, or None
        Name of a registered theme profile (see
        :func:`register_theme_profile`), or an inline profile dict.
        The ``context``, ``font_scale`` and ``rc`` arguments (if given)
        take precedence over values coming from the profile.

    Examples
    --------

    .. include:: ../docstrings/plotting_context.rst

    """
    if profile is not None:
        resolved = _resolve_profile(
            profile,
            {"context": context, "font_scale": font_scale, "rc": rc},
            kind="context",
        )
        context = resolved["context"]
        font_scale = resolved["font_scale"]
        rc = resolved["rc"]

    if font_scale is None:
        font_scale = 1

    if context is None:
        context_dict = {k: mpl.rcParams[k] for k in _context_keys}

    elif isinstance(context, dict):
        context_dict = context

    else:

        contexts = ["paper", "notebook", "talk", "poster"]
        if context not in contexts:
            raise ValueError(f"context must be in {', '.join(contexts)}")

        # Set up dictionary of default parameters
        texts_base_context = {

            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,

        }

        base_context = {

            "axes.linewidth": 1.25,
            "grid.linewidth": 1,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "patch.linewidth": 1,

            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,

            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,

        }
        base_context.update(texts_base_context)

        # Scale all the parameters by the same factor depending on the context
        scaling = dict(paper=.8, notebook=1, talk=1.5, poster=2)[context]
        context_dict = {k: v * scaling for k, v in base_context.items()}

        # Now independently scale the fonts
        font_keys = texts_base_context.keys()
        font_dict = {k: context_dict[k] * font_scale for k in font_keys}
        context_dict.update(font_dict)

    # Override these settings with the provided rc dictionary
    if rc is not None:
        rc_valid = _validate_rc_dict(rc, category="context")
        context_dict.update(rc_valid)

    # Wrap in a _PlottingContext object so this can be used in a with statement
    context_object = _PlottingContext(context_dict)

    return context_object


def set_context(context=None, font_scale=None, rc=None, *, profile=None):
    """
    Set the parameters that control the scaling of plot elements.

    These parameters correspond to label size, line thickness, etc.
    Calling this function modifies the global matplotlib `rcParams`. For more
    information, see the :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    The base context is "notebook", and the other contexts are "paper", "talk",
    and "poster", which are version of the notebook parameters scaled by different
    values. Font elements can also be scaled independently of (but relative to)
    the other values.

    See :func:`plotting_context` to get the parameter values.

    Parameters
    ----------
    context : dict, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        context dictionaries. This only updates parameters that are
        considered part of the context definition.
    profile : str, dict, or None
        Name of a registered theme profile (see
        :func:`register_theme_profile`), or an inline profile dict.

    Examples
    --------

    .. include:: ../docstrings/set_context.rst

    """
    context_object = plotting_context(context, font_scale, rc, profile=profile)
    mpl.rcParams.update(context_object)


class _RCAesthetics(dict):
    def __enter__(self):
        rc = mpl.rcParams
        self._orig = {k: rc[k] for k in self._keys}
        self._set(self)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._set(self._orig)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class _AxesStyle(_RCAesthetics):
    """Light wrapper on a dict to set style temporarily."""
    _keys = _style_keys
    _set = staticmethod(set_style)


class _PlottingContext(_RCAesthetics):
    """Light wrapper on a dict to set context temporarily."""
    _keys = _context_keys
    _set = staticmethod(set_context)


def set_palette(palette, n_colors=None, desat=None, color_codes=False):
    """Set the matplotlib color cycle using a seaborn palette.

    Parameters
    ----------
    palette : seaborn color palette | matplotlib colormap | hls | husl
        Palette definition. Should be something :func:`color_palette` can process.
    n_colors : int
        Number of colors in the cycle. The default number of colors will depend
        on the format of ``palette``, see the :func:`color_palette`
        documentation for more information.
    desat : float
        Proportion to desaturate each color by.
    color_codes : bool
        If ``True`` and ``palette`` is a seaborn palette, remap the shorthand
        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.

    See Also
    --------
    color_palette : build a color palette or set the color cycle temporarily
                    in a ``with`` statement.
    set_context : set parameters to scale plot elements
    set_style : set the default parameters for figure style

    """
    colors = palettes.color_palette(palette, n_colors, desat)
    cyl = cycler('color', colors)
    mpl.rcParams['axes.prop_cycle'] = cyl
    if color_codes:
        try:
            palettes.set_color_codes(palette)
        except (ValueError, TypeError):
            pass
