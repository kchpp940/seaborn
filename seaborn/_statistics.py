"""Statistical transformations for visualization.

This module is currently private, but is being written to eventually form part
of the public API.

The classes should behave roughly in the style of scikit-learn.

- All data-independent parameters should be passed to the class constructor.
- Each class should implement a default transformation that is exposed through
  __call__. These are currently written for vector arguments, but I think
  consuming a whole `plot_data` DataFrame and return it with transformed
  variables would make more sense.
- Some class have data-dependent preprocessing that should be cached and used
  multiple times (think defining histogram bins off all data and then counting
  observations within each bin multiple times per data subsets). These currently
  have unique names, but it would be good to have a common name. Not quite
  `fit`, but something similar.
- Alternatively, the transform interface could take some information about grouping
  variables and do a groupby internally.
- Some classes should define alternate transforms that might make the most sense
  with a different function. For example, KDE usually evaluates the distribution
  on a regular grid, but it would be useful for it to transform at the actual
  datapoints. Then again, this could be controlled by a parameter at  the time of
  class instantiation.

"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
try:
    from scipy.stats import gaussian_kde
    _no_scipy = False
except ImportError:
    from .external.kde import gaussian_kde
    _no_scipy = True

from .algorithms import bootstrap
from .utils import _check_argument


@dataclass
class BinDiagnostics:
    """Diagnostic information about histogram binning for a single data group.

    Attributes
    ----------
    bin_edges : np.ndarray or tuple of np.ndarray
        Actual bin edges used for this group.
        Univariate: 1D array of shape (n_bins + 1,).
        Bivariate: tuple of (x_edges, y_edges), each 1D array.
    count : int
        Number of valid (non-NaN, finite) samples used for binning.
    weight_sum : float
        Sum of weights for valid samples (1.0 * count if no weights).
    normalization_denominator : float or None
        Denominator used for normalization (hist.sum() for probability/percent,
        total area for density, etc.). None if stat == "count".
    empty_reason : str or None
        If the group produced no valid bins or data, explanation why.
        One of: "no_data", "all_nan", "zero_variance", or None if valid.
    extra : dict
        Additional implementation-specific metadata.
    """
    bin_edges: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None
    count: int = 0
    weight_sum: float = 0.0
    normalization_denominator: float | None = None
    empty_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        parts = []
        if self.bin_edges is not None:
            if isinstance(self.bin_edges, (tuple, list)) and len(self.bin_edges) == 2:
                parts.append(
                    f"bin_edges=(array(shape={self.bin_edges[0].shape}), "
                    f"array(shape={self.bin_edges[1].shape}))"
                )
            else:
                parts.append(f"bin_edges=array(shape={self.bin_edges.shape})")
        parts.append(f"count={self.count}")
        parts.append(f"weight_sum={self.weight_sum}")
        if self.normalization_denominator is not None:
            parts.append(f"norm_denom={self.normalization_denominator}")
        if self.empty_reason is not None:
            parts.append(f"empty_reason={self.empty_reason!r}")
        if self.extra:
            parts.append(f"extra={self.extra}")
        return f"BinDiagnostics({', '.join(parts)})"


class BinDiagnosticsCollector:
    """Unified collector for histogram binning diagnostics.

    This is the single source of truth for computing bin diagnostics.
    Both the legacy Histogram class and the objects-layer Hist class
    delegate to this collector to ensure consistent output.

    Parameters
    ----------
    stat : str
        The normalization statistic being computed.
    cumulative : bool
        Whether the statistic is cumulative.

    Examples
    --------
    >>> collector = BinDiagnosticsCollector(stat="density", cumulative=False)
    >>> collector.add_group(
    ...     group_key=("hue", "A"),
    ...     x=data_array,
    ...     bin_edges=edges,
    ...     hist=counts,
    ...     weights=weights,
    ... )
    >>> collector.diagnostics
    {('hue', 'A'): BinDiagnostics(...)}
    """

    def __init__(self, stat: str = "count", cumulative: bool = False):
        self.stat = stat
        self.cumulative = cumulative
        self._diagnostics: dict[tuple, BinDiagnostics] = {}

    @property
    def diagnostics(self) -> dict[tuple, BinDiagnostics]:
        """Read-only access to collected diagnostics."""
        return self._diagnostics

    def clear(self) -> None:
        """Clear all collected diagnostics."""
        self._diagnostics.clear()

    def add_group(
        self,
        group_key: tuple,
        x: np.ndarray | pd.Series | tuple[np.ndarray, np.ndarray],
        bin_edges: np.ndarray | list | tuple,
        hist: np.ndarray,
        weights: np.ndarray | pd.Series | None = None,
    ) -> BinDiagnostics:
        """Compute and store diagnostics for one data group.

        This is the unified entry point for computing bin diagnostics.
        Both legacy and objects layers should call this method.

        Parameters
        ----------
        group_key : tuple
            Key identifying the group, e.g. (("hue", "A"), ("col", "col1")).
            Use () for the overall / single group case.
        x : array-like or tuple of array-like
            The raw data values. For bivariate, pass (x1, x2).
        bin_edges : array-like or tuple/list of array-like
            The bin edges used. For bivariate, pass (x_edges, y_edges).
        hist : np.ndarray
            The computed histogram values (after normalization if applicable).
        weights : array-like, optional
            Sample weights.

        Returns
        -------
        BinDiagnostics
            The computed diagnostic record for this group.
        """
        # -- 1. Determine valid count and weight sum --
        is_bivariate = isinstance(x, (tuple, list)) and len(x) == 2 and not isinstance(x[0], (int, float))

        if is_bivariate:
            x1, x2 = np.asarray(x[0]), np.asarray(x[1])
            n_samples = len(x1)
            valid_mask = np.isfinite(x1) & np.isfinite(x2)
        else:
            x_arr = np.asarray(x)
            n_samples = len(x_arr)
            valid_mask = np.isfinite(x_arr)

        if weights is not None:
            weights_arr = np.asarray(weights)
            valid_mask = valid_mask & np.isfinite(weights_arr)
            weight_sum = float(weights_arr[valid_mask].sum())
        else:
            weight_sum = float(valid_mask.sum())

        count = int(valid_mask.sum())

        # -- 2. Determine empty reason --
        empty_reason = None
        if count == 0:
            if n_samples == 0:
                empty_reason = "no_data"
            else:
                empty_reason = "all_nan"
        else:
            # Check for zero variance
            if is_bivariate:
                var1 = float(np.nan_to_num(x1[valid_mask].var()))
                var2 = float(np.nan_to_num(x2[valid_mask].var()))
                if var1 == 0 and var2 == 0:
                    empty_reason = "zero_variance"
            else:
                if float(np.nan_to_num(x_arr[valid_mask].var())) == 0:
                    empty_reason = "zero_variance"

        # -- 3. Determine normalization denominator --
        if self.stat == "count":
            norm_denom = None
        elif self.stat in ("probability", "proportion"):
            norm_denom = float(hist.sum()) if hist.size > 0 else 0.0
        elif self.stat == "percent":
            norm_denom = float(hist.sum()) / 100.0 if hist.size > 0 else 0.0
        elif self.stat == "density":
            # density: total area should integrate to 1 (or weight_sum)
            if is_bivariate:
                x_edges = np.asarray(bin_edges[0])
                y_edges = np.asarray(bin_edges[1])
                area = np.outer(np.diff(x_edges), np.diff(y_edges))
                norm_denom = float((hist * area).sum()) if hist.size > 0 else 0.0
            else:
                edges_arr = np.asarray(bin_edges)
                norm_denom = float((hist * np.diff(edges_arr)).sum()) if hist.size > 0 else 0.0
        elif self.stat == "frequency":
            if is_bivariate:
                x_edges = np.asarray(bin_edges[0])
                y_edges = np.asarray(bin_edges[1])
                area = np.outer(np.diff(x_edges), np.diff(y_edges))
                norm_denom = float(area.mean()) if area.size > 0 else 0.0
            else:
                edges_arr = np.asarray(bin_edges)
                norm_denom = float(np.mean(np.diff(edges_arr))) if len(edges_arr) > 1 else 0.0
        else:
            norm_denom = None

        # -- 4. Normalize bin_edges representation --
        if is_bivariate:
            if isinstance(bin_edges, (tuple, list)) and len(bin_edges) == 2:
                edges_out = (np.asarray(bin_edges[0]), np.asarray(bin_edges[1]))
            else:
                edges_out = np.asarray(bin_edges)
        else:
            edges_out = np.asarray(bin_edges)

        # -- 5. Build diagnostics record --
        diag = BinDiagnostics(
            bin_edges=edges_out,
            count=count,
            weight_sum=weight_sum,
            normalization_denominator=norm_denom,
            empty_reason=empty_reason,
            extra={"stat": self.stat, "cumulative": self.cumulative},
        )
        self._diagnostics[group_key] = diag
        return diag

    def add_group_univariate(
        self,
        group_key: tuple,
        x: np.ndarray | pd.Series,
        bin_edges: np.ndarray | list,
        hist: np.ndarray,
        weights: np.ndarray | pd.Series | None = None,
    ) -> BinDiagnostics:
        """Convenience wrapper for univariate case."""
        return self.add_group(group_key, x, bin_edges, hist, weights)

    def add_group_bivariate(
        self,
        group_key: tuple,
        x1: np.ndarray | pd.Series,
        x2: np.ndarray | pd.Series,
        bin_edges: tuple[np.ndarray, np.ndarray] | list[np.ndarray],
        hist: np.ndarray,
        weights: np.ndarray | pd.Series | None = None,
    ) -> BinDiagnostics:
        """Convenience wrapper for bivariate case."""
        return self.add_group(group_key, (x1, x2), bin_edges, hist, weights)

    def add_group_from_result(self, result: HistGroupResult) -> BinDiagnostics:
        """Add diagnostics from a HistGroupResult (zero-copy where possible).

        This is the preferred entry point when a :class:`HistGroupResult`
        is already available, because it avoids re-deriving count /
        weight_sum / empty_reason from the raw data.
        """
        diag = BinDiagnostics(
            bin_edges=result.bin_edges,
            count=result.count,
            weight_sum=result.weight_sum,
            normalization_denominator=result.normalization_denominator,
            empty_reason=result.empty_reason,
            extra={"stat": self.stat, "cumulative": self.cumulative},
        )
        self._diagnostics[result.group_key] = diag
        return diag


@dataclass
class HistGroupResult:
    """Unified intermediate result of histogram computation for one group.

    Every histogram-producing code path (legacy ``Histogram``, objects-layer
    ``Hist``, and the ``plot_univariate_histogram`` / ``plot_bivariate_histogram``
    helpers in :mod:`seaborn.distributions`) should produce one
    :class:`HistGroupResult` per group so that downstream consumers can
    derive plotting data and :class:`BinDiagnostics` from the *same*
    structure, eliminating the risk of divergent bookkeeping.

    Attributes
    ----------
    hist : np.ndarray
        Normalized histogram values (after stat and cumulative transforms).
    bin_edges : np.ndarray or tuple[np.ndarray, np.ndarray]
        Bin edges used.  Univariate: 1-D ``(n_bins+1,)``; bivariate:
        ``(x_edges, y_edges)``.
    group_key : tuple
        Canonical group identifier, e.g. ``(("hue", "A"),)``.
    count : int
        Number of valid (finite, non-NaN) samples.
    weight_sum : float
        Sum of weights for valid samples (equals *count* when unweighted).
    empty_reason : str or None
        ``"no_data"``, ``"all_nan"``, ``"zero_variance"``, or ``None``.
    normalization_denominator : float or None
        Denominator used for the chosen *stat* (``None`` for ``"count"``).
    """

    hist: np.ndarray
    bin_edges: np.ndarray | tuple[np.ndarray, np.ndarray]
    group_key: tuple = ()
    count: int = 0
    weight_sum: float = 0.0
    empty_reason: str | None = None
    normalization_denominator: float | None = None


def _clean_data(x):
    """Return a finite-only numpy view and a boolean mask of valid entries.

    Parameters
    ----------
    x : array-like or Series
        Raw data.

    Returns
    -------
    x_arr : np.ndarray
        Original data as float64 array (inf replaced with NaN).
    valid_mask : np.ndarray[bool]
        True where the element is finite.
    """
    x_arr = np.asarray(x, dtype=float)
    x_arr = np.where(np.isfinite(x_arr), x_arr, np.nan)
    valid_mask = np.isfinite(x_arr)
    return x_arr, valid_mask


def define_bin_edges(x, weights, bins, binwidth, binrange, discrete):
    """Unified bin-edge computation for univariate histograms.

    This is the single source of truth for converting the high-level
    ``bins`` / ``binwidth`` / ``binrange`` / ``discrete`` parameters into an
    array of bin edges.  Both :class:`Histogram` and
    :class:`~seaborn._stats.counting.Hist` delegate to this function so
    that edge-case handling (round-off, single-observation, inf/NaN) is
    consistent across the legacy and objects layers.

    Parameters
    ----------
    x : array-like or Series
        Data values.
    weights : array-like or None
        Sample weights.
    bins : str, int, or array-like
        Bin specification forwarded to :func:`numpy.histogram_bin_edges`.
    binwidth : float or None
        Desired bin width; overrides *bins* but can be combined with
        *binrange*.
    binrange : tuple(float, float) or None
        ``(min, max)`` for bin edges.
    discrete : bool
        If True, produce unit-width bins centred on integers.

    Returns
    -------
    np.ndarray
        1-D array of bin edges with shape ``(n_bins + 1,)``.
    """
    x_arr, valid_mask = _clean_data(x)
    valid = x_arr[valid_mask]

    if len(valid) == 0:
        if binrange is not None:
            start, stop = float(binrange[0]), float(binrange[1])
        else:
            return np.array([0.0, 1.0])
    elif binrange is None:
        start, stop = float(valid.min()), float(valid.max())
    else:
        start, stop = float(binrange[0]), float(binrange[1])

    if discrete:
        bin_edges = np.arange(start - .5, stop + 1.5)
    elif binwidth is not None:
        step = binwidth
        bin_edges = np.arange(start, stop + step, step)
        if bin_edges.max() < stop or len(bin_edges) < 2:
            bin_edges = np.append(bin_edges, bin_edges.max() + step)
    else:
        w = np.asarray(weights, dtype=float) if weights is not None else None
        if w is not None:
            w = w[valid_mask] if len(w) == len(x_arr) else w
        bin_edges = np.histogram_bin_edges(valid, bins, binrange, w)

    return bin_edges


def define_bin_edges_bivariate(x1, x2, weights, bins, binwidth, binrange, discrete):
    """Unified bin-edge computation for bivariate histograms.

    Parameters
    ----------
    x1, x2 : array-like or Series
        Data values for the two axes.
    weights : array-like or None
        Sample weights.
    bins : str, int, array-like, or pair of such values
        Bin specification (possibly per-axis).
    binwidth : float or pair of floats or None
        Desired bin width (possibly per-axis).
    binrange : pair of floats or pair of pairs or None
        Bin range (possibly per-axis).
    discrete : bool or pair of bools
        Discrete flag (possibly per-axis).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(x_edges, y_edges)``, each a 1-D array of bin edges.
    """
    edges_list = []
    for i, x in enumerate([x1, x2]):
        b = bins
        if not b or isinstance(b, (str, Number, np.integer)):
            pass
        elif isinstance(b[i], str):
            b = b[i]
        elif len(b) == 2:
            b = b[i]

        bw = binwidth
        if bw is None:
            pass
        elif not isinstance(bw, Number):
            bw = bw[i]

        br = binrange
        if br is None:
            pass
        elif not isinstance(br[0], Number):
            br = br[i]

        d = discrete
        if not isinstance(d, bool):
            d = d[i]

        edges_list.append(define_bin_edges(x, weights, b, bw, br, d))

    return tuple(edges_list)


def bin_edges_to_kws(bin_edges, bins):
    """Convert bin edges to ``numpy.histogram`` keyword arguments.

    Parameters
    ----------
    bin_edges : np.ndarray
        1-D array of bin edges.
    bins : str, int, or array-like
        Original *bins* parameter (used to choose output format).

    Returns
    -------
    dict
        ``{"bins": int, "range": (float, float)}`` or ``{"bins": ndarray}``.
    """
    if isinstance(bins, (str, int, np.integer)):
        n_bins = len(bin_edges) - 1
        bin_range = float(bin_edges.min()), float(bin_edges.max())
        return dict(bins=n_bins, range=bin_range)
    else:
        return dict(bins=bin_edges)


def normalize_hist_array(hist, edges, stat, cumulative, is_bivariate=False):
    """Apply normalization and cumulative transforms to a raw histogram.

    This is the single source of truth for the stat→normalization mapping
    used by both :class:`Histogram` and
    :class:`~seaborn._stats.counting.Hist`.

    Parameters
    ----------
    hist : np.ndarray
        Raw histogram counts (or density values when *stat* is ``"density"``).
    edges : np.ndarray or tuple[np.ndarray, np.ndarray]
        Bin edges.
    stat : str
        One of ``"count"``, ``"frequency"``, ``"density"``,
        ``"probability"``, ``"proportion"``, ``"percent"``.
    cumulative : bool
        Whether to apply cumulative summation.
    is_bivariate : bool
        Whether the histogram is bivariate.

    Returns
    -------
    np.ndarray
        Normalized (and possibly cumulated) histogram values.
    """
    if stat == "probability" or stat == "proportion":
        total = hist.sum()
        hist = hist.astype(float) / total if total > 0 else hist.astype(float)
    elif stat == "percent":
        total = hist.sum()
        hist = hist.astype(float) / total * 100 if total > 0 else hist.astype(float)
    elif stat == "frequency":
        if is_bivariate:
            area = np.outer(np.diff(edges[0]), np.diff(edges[1]))
            hist = hist.astype(float) / area
        else:
            width = np.diff(edges)
            hist = hist.astype(float) / width

    if cumulative:
        if stat in ["density", "frequency"]:
            if is_bivariate:
                area = np.outer(np.diff(edges[0]), np.diff(edges[1]))
                hist = (hist * area).cumsum(axis=0).cumsum(axis=1)
            else:
                width = np.diff(edges)
                hist = (hist * width).cumsum()
        else:
            if is_bivariate:
                hist = hist.cumsum(axis=0).cumsum(axis=1)
            else:
                hist = hist.cumsum()

    return hist


def compute_valid_stats(x, weights=None):
    """Compute valid count, weight sum, and empty reason for data.

    Parameters
    ----------
    x : array-like, Series, or tuple of two array-likes
        Raw data.  For bivariate data pass ``(x1, x2)``.
    weights : array-like or None
        Sample weights.

    Returns
    -------
    count : int
        Number of valid samples.
    weight_sum : float
        Sum of weights for valid samples.
    empty_reason : str or None
        ``"no_data"``, ``"all_nan"``, ``"zero_variance"``, or ``None``.
    """
    is_bivariate = (
        isinstance(x, (tuple, list)) and len(x) == 2
        and not isinstance(x[0], (int, float))
    )

    if is_bivariate:
        x1_arr = np.asarray(x[0], dtype=float)
        x2_arr = np.asarray(x[1], dtype=float)
        n_samples = len(x1_arr)
        valid_mask = np.isfinite(x1_arr) & np.isfinite(x2_arr)
    else:
        x_arr = np.asarray(x, dtype=float)
        n_samples = len(x_arr)
        valid_mask = np.isfinite(x_arr)

    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        if len(weights_arr) == len(valid_mask):
            valid_mask = valid_mask & np.isfinite(weights_arr)
        weight_sum = float(weights_arr[valid_mask].sum()) if valid_mask.any() else 0.0
    else:
        weight_sum = float(valid_mask.sum())

    count = int(valid_mask.sum())

    empty_reason = None
    if count == 0:
        if n_samples == 0:
            empty_reason = "no_data"
        else:
            empty_reason = "all_nan"
    else:
        if is_bivariate:
            v1 = float(np.nan_to_num(x1_arr[valid_mask].var()))
            v2 = float(np.nan_to_num(x2_arr[valid_mask].var()))
            if v1 == 0 and v2 == 0:
                empty_reason = "zero_variance"
        else:
            if float(np.nan_to_num(x_arr[valid_mask].var())) == 0:
                empty_reason = "zero_variance"

    return count, weight_sum, empty_reason


def compute_normalization_denominator(hist, bin_edges, stat, is_bivariate=False):
    """Compute the normalization denominator for a given stat.

    Parameters
    ----------
    hist : np.ndarray
        Histogram values (after normalization if applicable).
    bin_edges : np.ndarray or tuple[np.ndarray, np.ndarray]
        Bin edges.
    stat : str
        Normalization statistic.
    is_bivariate : bool
        Whether the histogram is bivariate.

    Returns
    -------
    float or None
        Normalization denominator, or ``None`` for ``stat="count"``.
    """
    if stat == "count":
        return None
    elif stat in ("probability", "proportion"):
        return float(hist.sum()) if hist.size > 0 else 0.0
    elif stat == "percent":
        return float(hist.sum()) / 100.0 if hist.size > 0 else 0.0
    elif stat == "density":
        if is_bivariate:
            area = np.outer(np.diff(bin_edges[0]), np.diff(bin_edges[1]))
            return float((hist * area).sum()) if hist.size > 0 else 0.0
        else:
            edges_arr = np.asarray(bin_edges)
            return float((hist * np.diff(edges_arr)).sum()) if hist.size > 0 else 0.0
    elif stat == "frequency":
        if is_bivariate:
            area = np.outer(np.diff(bin_edges[0]), np.diff(bin_edges[1]))
            return float(area.mean()) if area.size > 0 else 0.0
        else:
            edges_arr = np.asarray(bin_edges)
            return float(np.mean(np.diff(edges_arr))) if len(edges_arr) > 1 else 0.0
    else:
        return None


def make_group_key(groupby_vars, key):
    """Create a canonical group key from grouping variable names and a key.

    This ensures that :class:`Histogram`, objects-layer :class:`Hist`,
    and the helpers in :mod:`seaborn.distributions` all produce the same
    key representation for equivalent groups.

    Parameters
    ----------
    groupby_vars : str or list of str
        Name(s) of the grouping variable(s).
    key : scalar or tuple
        Key value(s) from a ``groupby`` iteration.

    Returns
    -------
    tuple
        Tuple of ``(variable_name, value)`` pairs, e.g.
        ``(("hue", "A"), ("col", 1))``.
    """
    if isinstance(groupby_vars, (list, tuple)):
        if not isinstance(key, tuple):
            key = (key,)
        return tuple(zip(groupby_vars, key))
    else:
        if isinstance(key, tuple):
            return tuple(zip([groupby_vars], key))
        return ((groupby_vars, key),)


def compute_hist_group_univariate(
    x, weights, bin_kws, stat, cumulative, group_key=(),
):
    """Compute a univariate histogram for one group and return HistGroupResult.

    This is the unified entry point that combines histogram computation,
    normalization, and diagnostics into a single pass so that downstream
    consumers never need to re-derive any intermediate values.

    Parameters
    ----------
    x : array-like or Series
        Data values.
    weights : array-like or None
        Sample weights.
    bin_kws : dict
        Keyword arguments for :func:`numpy.histogram`.
    stat : str
        Normalization statistic.
    cumulative : bool
        Whether to cumulate.
    group_key : tuple
        Canonical group key.

    Returns
    -------
    HistGroupResult
        Complete intermediate result for this group.
    """
    density = stat == "density"
    hist, edges = np.histogram(x, **bin_kws, weights=weights, density=density)
    hist = normalize_hist_array(hist, edges, stat, cumulative, is_bivariate=False)

    count, weight_sum, empty_reason = compute_valid_stats(x, weights)
    norm_denom = compute_normalization_denominator(hist, edges, stat, is_bivariate=False)

    return HistGroupResult(
        hist=hist,
        bin_edges=edges,
        group_key=group_key,
        count=count,
        weight_sum=weight_sum,
        empty_reason=empty_reason,
        normalization_denominator=norm_denom,
    )


def compute_hist_group_bivariate(
    x1, x2, weights, bin_kws, stat, cumulative, group_key=(),
):
    """Compute a bivariate histogram for one group and return HistGroupResult.

    Parameters
    ----------
    x1, x2 : array-like or Series
        Data values for the two axes.
    weights : array-like or None
        Sample weights.
    bin_kws : dict
        Keyword arguments for :func:`numpy.histogram2d`.
    stat : str
        Normalization statistic.
    cumulative : bool
        Whether to cumulate.
    group_key : tuple
        Canonical group key.

    Returns
    -------
    HistGroupResult
        Complete intermediate result for this group.
    """
    density = stat == "density"
    hist, *bin_edges = np.histogram2d(x1, x2, **bin_kws, weights=weights, density=density)
    bin_edges = tuple(bin_edges)
    hist = normalize_hist_array(hist, bin_edges, stat, cumulative, is_bivariate=True)

    count, weight_sum, empty_reason = compute_valid_stats((x1, x2), weights)
    norm_denom = compute_normalization_denominator(hist, bin_edges, stat, is_bivariate=True)

    return HistGroupResult(
        hist=hist,
        bin_edges=bin_edges,
        group_key=group_key,
        count=count,
        weight_sum=weight_sum,
        empty_reason=empty_reason,
        normalization_denominator=norm_denom,
    )


class KDE:
    """Univariate and bivariate kernel density estimator."""
    def __init__(
        self, *,
        bw_method=None,
        bw_adjust=1,
        gridsize=200,
        cut=3,
        clip=None,
        cumulative=False,
    ):
        """Initialize the estimator with its parameters.

        Parameters
        ----------
        bw_method : string, scalar, or callable, optional
            Method for determining the smoothing bandwidth to use; passed to
            :class:`scipy.stats.gaussian_kde`.
        bw_adjust : number, optional
            Factor that multiplicatively scales the value chosen using
            ``bw_method``. Increasing will make the curve smoother. See Notes.
        gridsize : int, optional
            Number of points on each dimension of the evaluation grid.
        cut : number, optional
            Factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints. When
            set to 0, truncate the curve at the data limits.
        clip : pair of numbers or None, or a pair of such pairs
            Do not evaluate the density outside of these limits.
        cumulative : bool, optional
            If True, estimate a cumulative distribution function. Requires scipy.

        """
        if clip is None:
            clip = None, None

        self.bw_method = bw_method
        self.bw_adjust = bw_adjust
        self.gridsize = gridsize
        self.cut = cut
        self.clip = clip
        self.cumulative = cumulative

        if cumulative and _no_scipy:
            raise RuntimeError("Cumulative KDE evaluation requires scipy")

        self.support = None

    def _define_support_grid(self, x, bw, cut, clip, gridsize):
        """Create the grid of evaluation points depending for vector x."""
        clip_lo = -np.inf if clip[0] is None else clip[0]
        clip_hi = +np.inf if clip[1] is None else clip[1]
        gridmin = max(x.min() - bw * cut, clip_lo)
        gridmax = min(x.max() + bw * cut, clip_hi)
        return np.linspace(gridmin, gridmax, gridsize)

    def _define_support_univariate(self, x, weights):
        """Create a 1D grid of evaluation points."""
        kde = self._fit(x, weights)
        bw = np.sqrt(kde.covariance.squeeze())
        grid = self._define_support_grid(
            x, bw, self.cut, self.clip, self.gridsize
        )
        return grid

    def _define_support_bivariate(self, x1, x2, weights):
        """Create a 2D grid of evaluation points."""
        clip = self.clip
        if clip[0] is None or np.isscalar(clip[0]):
            clip = (clip, clip)

        kde = self._fit([x1, x2], weights)
        bw = np.sqrt(np.diag(kde.covariance).squeeze())

        grid1 = self._define_support_grid(
            x1, bw[0], self.cut, clip[0], self.gridsize
        )
        grid2 = self._define_support_grid(
            x2, bw[1], self.cut, clip[1], self.gridsize
        )

        return grid1, grid2

    def define_support(self, x1, x2=None, weights=None, cache=True):
        """Create the evaluation grid for a given data set."""
        if x2 is None:
            support = self._define_support_univariate(x1, weights)
        else:
            support = self._define_support_bivariate(x1, x2, weights)

        if cache:
            self.support = support

        return support

    def _fit(self, fit_data, weights=None):
        """Fit the scipy kde while adding bw_adjust logic and version check."""
        fit_kws = {"bw_method": self.bw_method}
        if weights is not None:
            fit_kws["weights"] = weights

        kde = gaussian_kde(fit_data, **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        return kde

    def _eval_univariate(self, x, weights=None):
        """Fit and evaluate a univariate on univariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x, cache=False)

        kde = self._fit(x, weights)

        if self.cumulative:
            s_0 = support[0]
            density = np.array([
                kde.integrate_box_1d(s_0, s_i) for s_i in support
            ])
        else:
            density = kde(support)

        return density, support

    def _eval_bivariate(self, x1, x2, weights=None):
        """Fit and evaluate a univariate on bivariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x1, x2, cache=False)

        kde = self._fit([x1, x2], weights)

        if self.cumulative:

            grid1, grid2 = support
            density = np.zeros((grid1.size, grid2.size))
            p0 = grid1.min(), grid2.min()
            for i, xi in enumerate(grid1):
                for j, xj in enumerate(grid2):
                    density[i, j] = kde.integrate_box(p0, (xi, xj))

        else:

            xx1, xx2 = np.meshgrid(*support)
            density = kde([xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

        return density, support

    def __call__(self, x1, x2=None, weights=None):
        """Fit and evaluate on univariate or bivariate data."""
        if x2 is None:
            return self._eval_univariate(x1, weights)
        else:
            return self._eval_bivariate(x1, x2, weights)


# Note: we no longer use this for univariate histograms in histplot,
# preferring _stats.Hist. We'll deprecate this once we have a bivariate Stat class.
class Histogram:
    """Univariate and bivariate histogram estimator."""
    def __init__(
        self,
        stat="count",
        bins="auto",
        binwidth=None,
        binrange=None,
        discrete=False,
        cumulative=False,
    ):
        """Initialize the estimator with its parameters.

        Parameters
        ----------
        stat : str
            Aggregate statistic to compute in each bin.

            - `count`: show the number of observations in each bin
            - `frequency`: show the number of observations divided by the bin width
            - `probability` or `proportion`: normalize such that bar heights sum to 1
            - `percent`: normalize such that bar heights sum to 100
            - `density`: normalize such that the total area of the histogram equals 1

        bins : str, number, vector, or a pair of such values
            Generic bin parameter that can be the name of a reference rule,
            the number of bins, or the breaks of the bins.
            Passed to :func:`numpy.histogram_bin_edges`.
        binwidth : number or pair of numbers
            Width of each bin, overrides ``bins`` but can be used with
            ``binrange``.
        binrange : pair of numbers or a pair of pairs
            Lowest and highest value for bin edges; can be used either
            with ``bins`` or ``binwidth``. Defaults to data extremes.
        discrete : bool or pair of bools
            If True, set ``binwidth`` and ``binrange`` such that bin
            edges cover integer values in the dataset.
        cumulative : bool
            If True, return the cumulative statistic.

        """
        stat_choices = [
            "count", "frequency", "density", "probability", "proportion", "percent",
        ]
        _check_argument("stat", stat_choices, stat)

        self.stat = stat
        self.bins = bins
        self.binwidth = binwidth
        self.binrange = binrange
        self.discrete = discrete
        self.cumulative = cumulative

        self.bin_kws = None
        self._diagnostics_collector = BinDiagnosticsCollector(
            stat=stat, cumulative=cumulative
        )

    def _define_bin_edges(self, x, weights, bins, binwidth, binrange, discrete):
        """Inner function that takes bin parameters as arguments.

        Delegates to the shared :func:`define_bin_edges` for consistent
        edge-case handling across all histogram code paths.
        """
        return define_bin_edges(x, weights, bins, binwidth, binrange, discrete)

    def define_bin_params(self, x1, x2=None, weights=None, cache=True):
        """Given data, return numpy.histogram parameters to define bins."""
        if x2 is None:

            bin_edges = self._define_bin_edges(
                x1, weights, self.bins, self.binwidth, self.binrange, self.discrete,
            )
            bin_kws = bin_edges_to_kws(bin_edges, self.bins)

        else:

            bin_edges = define_bin_edges_bivariate(
                x1, x2, weights,
                self.bins, self.binwidth, self.binrange, self.discrete,
            )
            bin_kws = dict(bins=bin_edges)

        if cache:
            self.bin_kws = bin_kws

        return bin_kws

    @property
    def diagnostics_(self) -> dict[tuple, BinDiagnostics]:
        """Collected bin diagnostics (read-only dict view into collector)."""
        return self._diagnostics_collector.diagnostics

    def compute_group(self, x1, x2=None, weights=None, group_key=()):
        """Compute histogram for one group and return the full result object.

        This is the single internal computation entry point used by the
        public :meth:`__call__`, the objects-layer :class:`Hist`, and the
        plotter-level :func:`histplot`/``displot``.  All three layers
        consume the same :class:`HistGroupResult` structure so bin edges,
        effective sample counts, weights, normalization denominators, and
        empty-group reasons are guaranteed to be identical.

        Parameters
        ----------
        x1 : array-like
            First variable values.
        x2 : array-like, optional
            Second variable values for bivariate histograms.
        weights : array-like, optional
            Observation weights.
        group_key : tuple
            Identifier for this group (e.g. ``(("hue", "A"),)``).

        Returns
        -------
        HistGroupResult
            The complete histogram result including bin edges, counts,
            diagnostics info, and normalized hist values.
        """
        bin_kws = self.bin_kws
        if x2 is None:
            if bin_kws is None:
                bin_kws = self.define_bin_params(x1, weights=weights, cache=False)
            return compute_hist_group_univariate(
                x1, weights, bin_kws, self.stat, self.cumulative, group_key,
            )
        else:
            if bin_kws is None:
                bin_kws = self.define_bin_params(x1, x2, cache=False)
            return compute_hist_group_bivariate(
                x1, x2, weights, bin_kws, self.stat, self.cumulative, group_key,
            )

    def __call__(self, x1, x2=None, weights=None, group_key=()):
        """Count the occurrences in each bin, maybe normalize.

        Returns
        -------
        hist : ndarray
            The histogram counts/values (1D for univariate, 2D for bivariate).
        bin_edges : ndarray or tuple of ndarray
            The bin edges (single array for univariate, ``(x_edges, y_edges)``
            for bivariate).
        """
        result = self.compute_group(x1, x2, weights, group_key)
        self._diagnostics_collector.add_group_from_result(result)
        return result.hist, result.bin_edges


class ECDF:
    """Univariate empirical cumulative distribution estimator."""
    def __init__(self, stat="proportion", complementary=False):
        """Initialize the class with its parameters

        Parameters
        ----------
        stat : {{"proportion", "percent", "count"}}
            Distribution statistic to compute.
        complementary : bool
            If True, use the complementary CDF (1 - CDF)

        """
        _check_argument("stat", ["count", "percent", "proportion"], stat)
        self.stat = stat
        self.complementary = complementary

    def _eval_bivariate(self, x1, x2, weights):
        """Inner function for ECDF of two variables."""
        raise NotImplementedError("Bivariate ECDF is not implemented")

    def _eval_univariate(self, x, weights):
        """Inner function for ECDF of one variable."""
        sorter = x.argsort()
        x = x[sorter]
        weights = weights[sorter]
        y = weights.cumsum()

        if self.stat in ["percent", "proportion"]:
            y = y / y.max()
        if self.stat == "percent":
            y = y * 100

        x = np.r_[-np.inf, x]
        y = np.r_[0, y]

        if self.complementary:
            y = y.max() - y

        return y, x

    def __call__(self, x1, x2=None, weights=None):
        """Return proportion or count of observations below each sorted datapoint."""
        x1 = np.asarray(x1)
        if weights is None:
            weights = np.ones_like(x1)
        else:
            weights = np.asarray(weights)

        if x2 is None:
            return self._eval_univariate(x1, weights)
        else:
            return self._eval_bivariate(x1, x2, weights)


class EstimateAggregator:

    def __init__(self, estimator, errorbar=None, **boot_kws):
        """
        Data aggregator that produces an estimate and error bar interval.

        Parameters
        ----------
        estimator : callable or string
            Function (or method name) that maps a vector to a scalar.
        errorbar : string, (string, number) tuple, or callable
            Name of errorbar method (either "ci", "pi", "se", or "sd"), or a tuple
            with a method name and a level parameter, or a function that maps from a
            vector to a (min, max) interval, or None to hide errorbar. See the
            :doc:`errorbar tutorial </tutorial/error_bars>` for more information.
        boot_kws
            Additional keywords are passed to bootstrap when error_method is "ci".

        """
        self.estimator = estimator

        method, level = _validate_errorbar_arg(errorbar)
        self.error_method = method
        self.error_level = level

        self.boot_kws = boot_kws

    def __call__(self, data, var):
        """Aggregate over `var` column of `data` with estimate and error interval."""
        vals = data[var]
        if callable(self.estimator):
            # You would think we could pass to vals.agg, and yet:
            # https://github.com/mwaskom/seaborn/issues/2943
            estimate = self.estimator(vals)
        else:
            estimate = vals.agg(self.estimator)

        # Options that produce no error bars
        if self.error_method is None:
            err_min = err_max = np.nan
        elif len(data) <= 1:
            err_min = err_max = np.nan

        # Generic errorbars from user-supplied function
        elif callable(self.error_method):
            err_min, err_max = self.error_method(vals)

        # Parametric options
        elif self.error_method == "sd":
            half_interval = vals.std() * self.error_level
            err_min, err_max = estimate - half_interval, estimate + half_interval
        elif self.error_method == "se":
            half_interval = vals.sem() * self.error_level
            err_min, err_max = estimate - half_interval, estimate + half_interval

        # Nonparametric options
        elif self.error_method == "pi":
            err_min, err_max = _percentile_interval(vals, self.error_level)
        elif self.error_method == "ci":
            units = data.get("units", None)
            boots = bootstrap(vals, units=units, func=self.estimator, **self.boot_kws)
            err_min, err_max = _percentile_interval(boots, self.error_level)

        return pd.Series({var: estimate, f"{var}min": err_min, f"{var}max": err_max})


class WeightedAggregator:

    def __init__(self, estimator, errorbar=None, **boot_kws):
        """
        Data aggregator that produces a weighted estimate and error bar interval.

        Parameters
        ----------
        estimator : string
            Function (or method name) that maps a vector to a scalar. Currently
            supports only "mean".
        errorbar : string or (string, number) tuple
            Name of errorbar method or a tuple with a method name and a level parameter.
            Currently the only supported method is "ci".
        boot_kws
            Additional keywords are passed to bootstrap when error_method is "ci".

        """
        if estimator != "mean":
            # Note that, while other weighted estimators may make sense (e.g. median),
            # I'm not aware of an implementation in our dependencies. We can add one
            # in seaborn later, if there is sufficient interest. For now, limit to mean.
            raise ValueError(f"Weighted estimator must be 'mean', not {estimator!r}.")
        self.estimator = estimator

        method, level = _validate_errorbar_arg(errorbar)
        if method is not None and method != "ci":
            # As with the estimator, weighted 'sd' or 'pi' error bars may make sense.
            # But we'll keep things simple for now and limit to (bootstrap) CI.
            raise ValueError(f"Error bar method must be 'ci', not {method!r}.")
        self.error_method = method
        self.error_level = level

        self.boot_kws = boot_kws

    def __call__(self, data, var):
        """Aggregate over `var` column of `data` with estimate and error interval."""
        vals = data[var]
        weights = data["weight"]

        estimate = np.average(vals, weights=weights)

        if self.error_method == "ci" and len(data) > 1:

            def error_func(x, w):
                return np.average(x, weights=w)

            boots = bootstrap(vals, weights, func=error_func, **self.boot_kws)
            err_min, err_max = _percentile_interval(boots, self.error_level)

        else:
            err_min = err_max = np.nan

        return pd.Series({var: estimate, f"{var}min": err_min, f"{var}max": err_max})


class LetterValues:

    def __init__(self, k_depth, outlier_prop, trust_alpha):
        """
        Compute percentiles of a distribution using various tail stopping rules.

        Parameters
        ----------
        k_depth: "tukey", "proportion", "trustworthy", or "full"
            Stopping rule for choosing tail percentiled to show:

            - tukey: Show a similar number of outliers as in a conventional boxplot.
            - proportion: Show approximately `outlier_prop` outliers.
            - trust_alpha: Use `trust_alpha` level for most extreme tail percentile.

        outlier_prop: float
            Parameter for `k_depth="proportion"` setting the expected outlier rate.
        trust_alpha: float
            Parameter for `k_depth="trustworthy"` setting the confidence threshold.

        Notes
        -----
        Based on the proposal in this paper:
        https://vita.had.co.nz/papers/letter-value-plot.pdf

        """
        k_options = ["tukey", "proportion", "trustworthy", "full"]
        if isinstance(k_depth, str):
            _check_argument("k_depth", k_options, k_depth)
        elif not isinstance(k_depth, int):
            err = (
                "The `k_depth` parameter must be either an integer or string "
                f"(one of {k_options}), not {k_depth!r}."
            )
            raise TypeError(err)

        self.k_depth = k_depth
        self.outlier_prop = outlier_prop
        self.trust_alpha = trust_alpha

    def _compute_k(self, n):

        # Select the depth, i.e. number of boxes to draw, based on the method
        if self.k_depth == "full":
            # extend boxes to 100% of the data
            k = int(np.log2(n)) + 1
        elif self.k_depth == "tukey":
            # This results with 5-8 points in each tail
            k = int(np.log2(n)) - 3
        elif self.k_depth == "proportion":
            k = int(np.log2(n)) - int(np.log2(n * self.outlier_prop)) + 1
        elif self.k_depth == "trustworthy":
            normal_quantile_func = np.vectorize(NormalDist().inv_cdf)
            point_conf = 2 * normal_quantile_func(1 - self.trust_alpha / 2) ** 2
            k = int(np.log2(n / point_conf)) + 1
        else:
            # Allow having k directly specified as input
            k = int(self.k_depth)

        return max(k, 1)

    def __call__(self, x):
        """Evaluate the letter values."""
        k = self._compute_k(len(x))
        exp = np.arange(k + 1, 1, -1), np.arange(2, k + 2)
        levels = k + 1 - np.concatenate([exp[0], exp[1][1:]])
        percentiles = 100 * np.concatenate([0.5 ** exp[0], 1 - 0.5 ** exp[1]])
        if self.k_depth == "full":
            percentiles[0] = 0
            percentiles[-1] = 100
        values = np.percentile(x, percentiles)
        fliers = np.asarray(x[(x < values.min()) | (x > values.max())])
        median = np.percentile(x, 50)

        return {
            "k": k,
            "levels": levels,
            "percs": percentiles,
            "values": values,
            "fliers": fliers,
            "median": median,
        }


def _percentile_interval(data, width):
    """Return a percentile interval from data of a given width."""
    edge = (100 - width) / 2
    percentiles = edge, 100 - edge
    return np.nanpercentile(data, percentiles)


def _validate_errorbar_arg(arg):
    """Check type and value of errorbar argument and assign default level."""
    DEFAULT_LEVELS = {
        "ci": 95,
        "pi": 95,
        "se": 1,
        "sd": 1,
    }

    usage = "`errorbar` must be a callable, string, or (string, number) tuple"

    if arg is None:
        return None, None
    elif callable(arg):
        return arg, None
    elif isinstance(arg, str):
        method = arg
        level = DEFAULT_LEVELS.get(method, None)
    else:
        try:
            method, level = arg
        except (ValueError, TypeError) as err:
            raise err.__class__(usage) from err

    _check_argument("errorbar", list(DEFAULT_LEVELS), method)
    if level is not None and not isinstance(level, Number):
        raise TypeError(usage)

    return method, level
