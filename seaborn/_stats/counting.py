from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from pandas import DataFrame

from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
from seaborn._stats.norm_utils import (
    effective_weight_total,
    filter_valid_samples,
    normalize_histogram,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass
class Count(Stat):
    """
    Count distinct observations within groups.

    See Also
    --------
    Hist : A more fully-featured transform including binning and/or normalization.

    Examples
    --------
    .. include:: ../docstrings/objects.Count.rst

    """
    group_by_orient: ClassVar[bool] = True

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        var = {"x": "y", "y": "x"}[orient]
        res = (
            groupby
            .agg(data.assign(**{var: data[orient]}), {var: len})
            .dropna(subset=["x", "y"])
            .reset_index(drop=True)
        )
        return res


@dataclass
class Hist(Stat):
    """
    Bin observations, count them, and optionally normalize or cumulate.

    Parameters
    ----------
    stat : str
        Aggregate statistic to compute in each bin:

        - `count`: the number of observations
        - `density`: normalize so that the total area of the histogram equals 1
        - `percent`: normalize so that bar heights sum to 100
        - `probability` or `proportion`: normalize so that bar heights sum to 1
        - `frequency`: divide the number of observations by the bin width

    bins : str, int, or ArrayLike
        Generic parameter that can be the name of a reference rule, the number
        of bins, or the bin breaks. Passed to :func:`numpy.histogram_bin_edges`.
    binwidth : float
        Width of each bin; overrides `bins` but can be used with `binrange`.
        Note that if `binwidth` does not evenly divide the bin range, the actual
        bin width used will be only approximately equal to the parameter value.
    binrange : (min, max)
        Lowest and highest value for bin edges; can be used with either
        `bins` (when a number) or `binwidth`. Defaults to data extremes.
    common_norm : bool or list of variables
        When not `False`, the normalization is applied across groups. Use
        `True` to normalize across all groups, or pass variable name(s) that
        define normalization groups.
    common_bins : bool or list of variables
        When not `False`, the same bins are used for all groups. Use `True` to
        share bins across all groups, or pass variable name(s) to share within.
    cumulative : bool
        If True, cumulate the bin values.
    discrete : bool
        If True, set `binwidth` and `binrange` so that bins have unit width and
        are centered on integer values

    Notes
    -----
    The choice of bins for computing and plotting a histogram can exert
    substantial influence on the insights that one is able to draw from the
    visualization. If the bins are too large, they may erase important features.
    On the other hand, bins that are too small may be dominated by random
    variability, obscuring the shape of the true underlying distribution. The
    default bin size is determined using a reference rule that depends on the
    sample size and variance. This works well in many cases, (i.e., with
    "well-behaved" data) but it fails in others. It is always a good to try
    different bin sizes to be sure that you are not missing something important.
    This function allows you to specify bins in several different ways, such as
    by setting the total number of bins to use, the width of each bin, or the
    specific locations where the bins should break.

    Examples
    --------
    .. include:: ../docstrings/objects.Hist.rst

    """
    stat: str = "count"
    bins: str | int | ArrayLike = "auto"
    binwidth: float | None = None
    binrange: tuple[float, float] | None = None
    common_norm: bool | list[str] = True
    common_bins: bool | list[str] = True
    cumulative: bool = False
    discrete: bool = False

    def __post_init__(self):

        stat_options = [
            "count", "density", "percent", "probability", "proportion", "frequency"
        ]
        self._check_param_one_of("stat", stat_options)

    def _define_bin_edges(self, vals, weight, bins, binwidth, binrange, discrete):
        """Inner function that takes bin parameters as arguments."""
        vals = vals.replace(-np.inf, np.nan).replace(np.inf, np.nan).dropna()

        if binrange is None:
            start, stop = vals.min(), vals.max()
        else:
            start, stop = binrange

        if discrete:
            bin_edges = np.arange(start - .5, stop + 1.5)
        else:
            if binwidth is not None:
                bins = int(round((stop - start) / binwidth))
            bin_edges = np.histogram_bin_edges(vals, bins, binrange, weight)

        # TODO warning or cap on too many bins?

        return bin_edges

    def _define_bin_params(self, data, orient, scale_type):
        """Given data, return numpy.histogram parameters to define bins."""
        vals = data[orient]
        weights = data.get("weight", None)

        # TODO We'll want this for ordinal / discrete scales too
        # (Do we need discrete as a parameter or just infer from scale?)
        discrete = self.discrete or scale_type == "nominal"

        bin_edges = self._define_bin_edges(
            vals, weights, self.bins, self.binwidth, self.binrange, discrete,
        )

        if isinstance(self.bins, (str, int)):
            n_bins = len(bin_edges) - 1
            bin_range = bin_edges.min(), bin_edges.max()
            bin_kws = dict(bins=n_bins, range=bin_range)
        else:
            bin_kws = dict(bins=bin_edges)

        return bin_kws

    def _get_bins_and_eval(self, data, orient, groupby, scale_type):

        bin_kws = self._define_bin_params(data, orient, scale_type)
        return groupby.apply(data, self._eval, orient, bin_kws)

    def _eval(self, data, orient, bin_kws):
        """Compute raw (weighted) counts without density normalization.

        Always asks numpy for counts so that ``common_norm`` has raw totals
        to work with; the normalization to density / probability / etc. is
        handled in :meth:`_normalize`.
        """

        vals = data[orient]
        weights = data.get("weight", None)

        hist, edges = np.histogram(vals, **bin_kws, weights=weights, density=False)

        width = np.diff(edges)
        center = edges[:-1] + width / 2

        return pd.DataFrame({orient: center, "count": hist, "space": width})

    def _normalize(self, data, total_weight_lookup=None):
        """Normalize raw histogram counts using the unified helper.

        Parameters
        ----------
        data : DataFrame
            Must contain the columns produced by :meth:`_eval`.
        total_weight_lookup : dict[tuple, float] | None
            Mapping from group-tuple (matching the order of the semantic
            columns present in ``data``) to the effective total weight of
            the *normalization group* that the row belongs to. When ``None``
            (or a row's key is missing) each hue group uses its own raw
            total count as the denominator, i.e. ``common_norm=False``
            semantics for that particular group.
        """

        orient_col = [c for c in ("x", "y") if c in data.columns][0]
        grouping_cols = [
            c for c in data.columns
            if c not in (orient_col, "count", "space")
        ]

        def _norm_one(part: DataFrame, tw: float | None) -> DataFrame:
            counts = part["count"].to_numpy(dtype=float)
            spaces = part["space"].to_numpy(dtype=float)
            eff_tw = float(tw) if tw is not None else float(counts.sum())
            out = normalize_histogram(
                counts, spaces, self.stat,
                cumulative=self.cumulative, total_weight=eff_tw,
            )
            return part.assign(**{self.stat: out})

        if grouping_cols:
            parts = []
            for key_tuple, part in data.groupby(grouping_cols, sort=False):
                k = (key_tuple,) if not isinstance(key_tuple, tuple) else key_tuple
                tw = (
                    total_weight_lookup.get(k)
                    if total_weight_lookup is not None
                    else None
                )
                parts.append(_norm_one(part, tw))
            data = pd.concat(parts, ignore_index=True)
        else:
            tw = (
                next(iter(total_weight_lookup.values()))
                if total_weight_lookup is not None
                else None
            )
            data = _norm_one(data, tw)

        return data

    @staticmethod
    def _build_weight_lookup(data, grouping_vars, norm_vars) -> dict[tuple, float]:
        """Return a dict: ``tuple(norm_group_values) -> effective total weight``.

        Only rows that actually contributed to the histogram (finite orient +
        finite weight) are counted.
        """

        if "weight" in data.columns:
            w = np.where(
                np.isfinite(data["weight"].to_numpy(dtype=float)),
                data["weight"].to_numpy(dtype=float),
                0.0,
            )
        else:
            w = np.ones(len(data), dtype=float)

        lookup: dict[tuple, float] = {}
        if not norm_vars:
            lookup[()] = float(w.sum())
        else:
            norm_idx = [grouping_vars.index(v) for v in norm_vars]
            for row_idx in range(len(data)):
                all_cols = [data[v].iat[row_idx] for v in grouping_vars]
                k = tuple(all_cols[i] for i in norm_idx)
                lookup[k] = lookup.get(k, 0.0) + float(w[row_idx])
        return lookup

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        # 1. Always filter non-finite samples up-front and ensure weight exists.
        data = filter_valid_samples(data, orient)

        scale_type = scales[orient].__class__.__name__.lower()
        grouping_vars = [str(v) for v in data if v in groupby.order]

        # Validate common_norm / common_bins parameter early so undefined
        # variable names always raise a warning, regardless of stat / etc.
        if isinstance(self.common_norm, list):
            self._check_grouping_vars("common_norm", grouping_vars, stacklevel=3)
        if isinstance(self.common_bins, list):
            self._check_grouping_vars("common_bins", grouping_vars, stacklevel=3)

        # 1a. Compute per-hue-group effective total weight BEFORE binning,
        #     because after binning we lose the per-sample information.
        #     Format: dict[tuple of grouping values, float]
        has_weight_col = "weight" in data.columns
        weight_per_hue: dict[tuple, float] = {}
        if grouping_vars:
            for key_tuple, part in data.groupby(grouping_vars, sort=False):
                k = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                if has_weight_col:
                    w_col = part["weight"].to_numpy(dtype=float)
                    w = np.where(np.isfinite(w_col), w_col, 0.0)
                    weight_per_hue[k] = float(w.sum())
                else:
                    weight_per_hue[k] = float(len(part))
        else:
            if has_weight_col:
                w_col = data["weight"].to_numpy(dtype=float)
                w = np.where(np.isfinite(w_col), w_col, 0.0)
                weight_per_hue[()] = float(w.sum())
            else:
                weight_per_hue[()] = float(len(data))

        # 2. Bin definition + raw count computation (always density=False so
        #    downstream normalization can do the correct thing).
        if not grouping_vars or self.common_bins is True:
            bin_kws = self._define_bin_params(data, orient, scale_type)
            data = groupby.apply(data, self._eval, orient, bin_kws)
        else:
            if self.common_bins is False:
                bin_groupby = GroupBy(grouping_vars)
            else:
                bin_groupby = GroupBy(self.common_bins)

            data = bin_groupby.apply(
                data, self._get_bins_and_eval, orient, groupby, scale_type,
            )

        # 3. Resolve normalization.
        #    - common_norm=True:   every hue group shares one total weight.
        #    - common_norm=False:  each hue group uses its own count total.
        #    - common_norm=[vars]: hue groups nested inside a "norm group"
        #                          share a total weight across that norm group.
        #
        #    Build a lookup: for each original hue group (identified by the
        #    tuple of its grouping_vars values in the order they appear in the
        #    result DataFrame), what is the correct denominator to use when
        #    normalizing?
        if not grouping_vars or self.stat == "count":
            lookup = None
        elif self.common_norm is True:
            total_all = float(sum(weight_per_hue.values()))
            sem_cols = [c for c in data.columns if c in grouping_vars]
            lookup: dict[tuple, float] = {}
            if sem_cols:
                for key_tuple, _ in data.groupby(sem_cols, sort=False):
                    k = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                    lookup[k] = total_all
            else:
                lookup[()] = total_all
        elif self.common_norm is False:
            lookup = None
        else:
            # common_norm = [var names]. Build a mapping from each hue group's
            # tuple to the total weight of the norm-group it belongs to.
            norm_vars = [v for v in self.common_norm if v in grouping_vars]

            sem_cols = [c for c in data.columns if c in grouping_vars]
            if not norm_vars or not sem_cols:
                total_all = float(sum(weight_per_hue.values()))
                lookup = {}
                for key_tuple, _ in data.groupby(sem_cols, sort=False):
                    k = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                    lookup[k] = total_all
            else:
                lookup = {}
                norm_col_idx = [grouping_vars.index(v) for v in norm_vars]

                for k_hue, hue_total in weight_per_hue.items():
                    nk = tuple(k_hue[i] for i in norm_col_idx)
                    # Sum all hue totals in the same norm-group
                    norm_total = 0.0
                    for k2_hue, t2 in weight_per_hue.items():
                        nk2 = tuple(k2_hue[i] for i in norm_col_idx)
                        if nk2 == nk:
                            norm_total += t2
                    lookup[k_hue] = norm_total

        data = self._normalize(data, total_weight_lookup=lookup)

        other = {"x": "y", "y": "x"}[orient]
        return data.assign(**{other: data[self.stat]})
