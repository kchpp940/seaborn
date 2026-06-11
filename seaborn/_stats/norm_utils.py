"""Unified normalization and sample filtering utilities for distribution stats.

This module provides shared helpers used by both the legacy plotting functions
in ``distributions.py`` and the object-oriented Stat layer in ``_stats/`` to
ensure consistent handling of:

* Filtering out non-finite samples (NaN, +/- Inf) from data and weights
* Detecting "empty" or "effectively empty" groups (no samples, all-zero weights)
* Computing normalization totals for ``common_norm`` scenarios
* Computing weighted / unweighted histogram normalization (count, density,
  probability, percent, frequency) in a way that is identical between the
  legacy code path and the objects API.
* Computing cumulative sums that preserve norm conventions across groups.

All helpers are deliberately stateless and operate on plain NumPy / pandas
containers so they can be called from either code path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


# ---------------------------------------------------------------------------
# Sample filtering
# ---------------------------------------------------------------------------

def filter_valid_samples(
    data: DataFrame,
    orient: str,
    weight_col: str = "weight",
) -> DataFrame:
    """Drop rows where the orient value or weight is non-finite.

    If a ``weight_col`` exists in the input, it is converted to ``float``.
    If it is missing, no default weight column is added – callers that
    require a numeric weight should apply ``assign`` themselves when needed.
    """
    data = data.copy()
    if weight_col in data.columns:
        data[weight_col] = data[weight_col].astype(float, copy=False)
        finite_mask = (
            np.isfinite(data[orient].to_numpy())
            & np.isfinite(data[weight_col].to_numpy())
        )
    else:
        data[orient] = pd.to_numeric(data[orient], errors="coerce")
        finite_mask = np.isfinite(data[orient].to_numpy())

    return data.loc[finite_mask].reset_index(drop=True)


def filter_valid_vectors(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vector-only version of :func:`filter_valid_samples`."""
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    finite_mask = np.isfinite(values) & np.isfinite(weights)
    return values[finite_mask], weights[finite_mask]


# ---------------------------------------------------------------------------
# Effective sample / empty detection
# ---------------------------------------------------------------------------

def effective_weight_total(weights: np.ndarray) -> float:
    """Sum of weights, treating non-finite values as zero.

    This is the canonical total weight used for normalization.
    """
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    return float(w.sum())


def is_effectively_empty(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    min_positive_weight: float = 0.0,
) -> bool:
    """Return True if a group has no usable samples.

    A group is effectively empty when:
      * there are no finite values, OR
      * the total positive weight is ``<= min_positive_weight``.

    Parameters
    ----------
    min_positive_weight : float, default 0
        If > 0, also treat groups whose total weight does not exceed this
        threshold as empty. Useful for downstream estimators (e.g. KDE) that
        require a strictly positive total.
    """
    v, w = filter_valid_vectors(values, weights)
    if v.size == 0:
        return True
    total = effective_weight_total(w)
    return total <= min_positive_weight


# ---------------------------------------------------------------------------
# Histogram normalization
# ---------------------------------------------------------------------------

def normalize_histogram(
    counts: np.ndarray,
    bin_widths: np.ndarray,
    stat: str,
    *,
    cumulative: bool = False,
    total_weight: Optional[float] = None,
) -> np.ndarray:
    """Normalize raw bin counts to the requested ``stat``.

    This is the single source of truth for histogram normalization. Both
    :class:`seaborn._stats.counting.Hist` and the legacy
    :class:`seaborn._statistics.Histogram` /
    :meth:`_DistributionPlotter.plot_univariate_histogram` code paths should
    funnel their normalization through here so output is identical between
    the legacy and objects APIs.

    Parameters
    ----------
    counts : ndarray
        Raw (possibly weighted) bin counts – i.e. the first return value of
        :func:`numpy.histogram` with ``weights`` but ``density=False``.
    bin_widths : ndarray
        Width of each bin (same length as ``counts``).
    stat : {"count", "density", "probability", "proportion", "percent", "frequency"}
        Target statistic.
    cumulative : bool, default False
        If True, return the cumulative version of the statistic.
    total_weight : float, optional
        Total (effective) weight of the *normalization group*. Required for
        ``density``, ``probability``, ``proportion``, ``percent`` and
        ``frequency``; for ``count`` it is ignored.

        When ``common_norm=True`` this should be the weight sum of *all*
        groups sharing the norm. When ``common_norm=False`` it should be the
        weight sum of the current group alone.
    """
    _check_stat(stat)
    counts = np.asarray(counts, dtype=float)
    bin_widths = np.asarray(bin_widths, dtype=float)

    # -- count --------------------------------------------------------------
    if stat == "count":
        out = counts
        if cumulative:
            out = np.nancumsum(out)
        return out

    # -- all other stats require a valid total ------------------------------
    if total_weight is None:
        raise ValueError(
            f"`total_weight` is required for stat={stat!r}. Pass the total "
            f"effective weight of the normalization group."
        )
    total_weight = float(total_weight)

    # Guard against divide-by-zero. When total_weight is 0 the "correct"
    # normalized output is all zeros (no probability mass anywhere).
    safe_total = total_weight if total_weight > 0 else 1.0

    if stat in ("probability", "proportion"):
        out = counts / safe_total
    elif stat == "percent":
        out = 100.0 * counts / safe_total
    elif stat == "density":
        # counts / total_weight normalizes to "probability"; dividing by the
        # bin width converts to a probability density whose integral is 1.
        out = counts / (safe_total * bin_widths)
    elif stat == "frequency":
        # counts / bin width: the expected "observations per unit x". Sum of
        # out * widths equals the raw count total (i.e. total_weight).
        out = counts / bin_widths
    else:  # pragma: no cover - _check_stat should have raised already
        raise ValueError(f"Unknown stat {stat!r}")

    if total_weight <= 0:
        out = np.zeros_like(out)

    if cumulative:
        # For density / frequency the cumulative is the integral (sum of y*dx).
        if stat in ("density", "frequency"):
            out = np.nancumsum(out * bin_widths)
        else:
            out = np.nancumsum(out)

    return out


def _check_stat(stat: str) -> None:
    valid = {"count", "density", "probability", "proportion", "percent", "frequency"}
    if stat not in valid:
        raise ValueError(
            f"`stat` must be one of {sorted(valid)}; got {stat!r}."
        )


# ---------------------------------------------------------------------------
# ECDF helpers
# ---------------------------------------------------------------------------

def compute_ecdf(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    stat: str = "proportion",
    complementary: bool = False,
    norm_total: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ECDF with consistent handling of empty / zero-weight groups.

    Parameters
    ----------
    values : 1D array-like
        Observed values.
    weights : 1D array-like, optional
        Observation weights (default = 1 for each sample).
    stat : {"proportion", "percent", "count"}
        Target statistic.
    complementary : bool, default False
        If True return the complementary CDF (1 - CDF).
    norm_total : float, optional
        Total effective weight of the *normalization group* (all subsets
        that share a ``common_norm`` scope).  When provided it replaces
        the per-group total as the denominator so that the ECDF of each
        subset reflects its share of the whole.  When ``None`` (the
        default) each group is normalised by its own total.

    Returns
    -------
    y : ndarray
        ECDF value at each ``x`` (length ``n_valid + 1``).
    x : ndarray
        Sorted unique abscissa with a leading ``-inf`` (length ``n_valid + 1``).
    """
    if stat not in {"proportion", "percent", "count"}:
        raise ValueError(
            "ECDF `stat` must be one of 'proportion', 'percent', 'count'; "
            f"got {stat!r}."
        )

    x, w = filter_valid_vectors(values, weights)

    n = x.size
    total = effective_weight_total(w)

    if n == 0 or (stat != "count" and total <= 0):
        y_out = np.array([0.0, 0.0])
        x_out = np.array([-np.inf, np.inf])
        return y_out, x_out

    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    w_sorted = w[order]

    y_raw = np.cumsum(w_sorted, dtype=float)

    if stat == "count":
        y = y_raw
    else:
        denom = float(norm_total) if norm_total is not None else total
        safe_denom = denom if denom > 0 else 1.0
        y = y_raw / safe_denom
        if denom <= 0:
            y = np.zeros_like(y)
        if stat == "percent":
            y = y * 100.0

    y_out = np.r_[0.0, y]
    x_out = np.r_[-np.inf, x_sorted]

    if complementary:
        y_out = y_out[-1] - y_out

    return y_out, x_out


# ---------------------------------------------------------------------------
# KDE helpers
# ---------------------------------------------------------------------------

def normalize_kde_density(
    density: np.ndarray,
    support: np.ndarray,
    group_weight_sum: float,
    norm_group_weight_sum: float,
) -> np.ndarray:
    """Scale a raw KDE output so the integral reflects ``common_norm``.

    Parameters
    ----------
    density : ndarray
        Output of a KDE evaluator – its integral over ``support`` is already
        (approximately) 1 because the estimator normalizes internally.
    support : ndarray
        Evaluation grid (only used to sanity check shapes – the scaling itself
        is purely multiplicative because both numerator and denominator are
        weight sums over the same data).
    group_weight_sum : float
        Total effective weight of the group that produced *density*.
    norm_group_weight_sum : float
        Total effective weight of the *normalization group* (i.e. all groups
        sharing a ``common_norm`` scope, or just this group when
        ``common_norm=False``).

    Returns
    -------
    scaled : ndarray
        Density scaled so that the integral over all curves in the norm group
        equals 1 when ``common_norm=True``, or each curve independently
        integrates to 1 when ``common_norm=False``.
    """
    density = np.asarray(density, dtype=float)
    if norm_group_weight_sum <= 0 or group_weight_sum <= 0:
        return np.zeros_like(density)
    # raw KDE already has integral 1 for the subset. To get the correct
    # mixture weight we rescale by (subset weight / norm group weight).
    return density * (group_weight_sum / norm_group_weight_sum)
