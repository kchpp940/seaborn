from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from pandas import DataFrame

from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
from seaborn._statistics import (
    BinDiagnostics,
    BinDiagnosticsCollector,
    HistGroupResult,
    define_bin_edges,
    bin_edges_to_kws,
    normalize_hist_array,
    compute_valid_stats,
    compute_normalization_denominator,
    make_group_key,
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

    _diagnostics_collector: BinDiagnosticsCollector = field(
        default_factory=lambda: BinDiagnosticsCollector(stat="count", cumulative=False),
        init=False,
        repr=False,
    )
    _pending_results: list = field(
        default_factory=list, init=False, repr=False,
    )

    def __post_init__(self):

        stat_options = [
            "count", "density", "percent", "probability", "proportion", "frequency"
        ]
        self._check_param_one_of("stat", stat_options)
        self._diagnostics_collector = BinDiagnosticsCollector(
            stat=self.stat, cumulative=self.cumulative
        )

    @property
    def diagnostics_(self) -> dict[tuple, BinDiagnostics]:
        """Collected bin diagnostics (read-only dict view into collector)."""
        return self._diagnostics_collector.diagnostics

    def _define_bin_edges(self, vals, weight, bins, binwidth, binrange, discrete):
        """Inner function that takes bin parameters as arguments.

        Delegates to the shared :func:`define_bin_edges` for consistent
        edge-case handling across all histogram code paths.
        """
        return define_bin_edges(vals, weight, bins, binwidth, binrange, discrete)

    def _define_bin_params(self, data, orient, scale_type):
        """Given data, return numpy.histogram parameters to define bins."""
        vals = data[orient]
        weights = data.get("weight", None)

        discrete = self.discrete or scale_type == "nominal"

        bin_edges = self._define_bin_edges(
            vals, weights, self.bins, self.binwidth, self.binrange, discrete,
        )

        return bin_edges_to_kws(bin_edges, self.bins)

    def _get_bins_and_eval(self, data, orient, groupby, scale_type):

        bin_kws = self._define_bin_params(data, orient, scale_type)
        return groupby.apply(data, self._eval, orient, bin_kws)

    def _eval(self, data, orient, bin_kws):
        """Evaluate histogram for a single group.

        Computes the raw histogram and stores a :class:`HistGroupResult`
        containing per-group diagnostics info.  This avoids a second
        histogram computation pass for diagnostics later.
        """
        vals = data[orient]
        weights = data.get("weight", None)

        density = self.stat == "density"
        hist_raw, edges = np.histogram(
            vals, **bin_kws, weights=weights, density=density,
        )

        hist_normed = normalize_hist_array(
            hist_raw.copy(), edges, self.stat, self.cumulative,
        )
        count, weight_sum, empty_reason = compute_valid_stats(vals, weights)
        norm_denom = compute_normalization_denominator(
            hist_normed, edges, self.stat,
        )
        result = HistGroupResult(
            hist=hist_normed,
            bin_edges=edges,
            group_key=(),
            count=count,
            weight_sum=weight_sum,
            empty_reason=empty_reason,
            normalization_denominator=norm_denom,
        )
        self._pending_results.append(result)

        width = np.diff(edges)
        center = edges[:-1] + width / 2

        return pd.DataFrame({orient: center, "count": hist_raw, "space": width})

    def _collect_diagnostics_from_pending(self, data, grouping_vars, orient):
        """Assign group keys to pending HistGroupResults and flush to collector.

        This replaces the old ``_collect_diagnostics_for_groups`` which
        re-computed histograms from scratch.  Now we reuse the
        :class:`HistGroupResult` objects produced by :meth:`_eval`, avoiding
        the double computation while still ensuring diagnostics are keyed by
        the correct group identifiers.
        """
        if not grouping_vars:
            if self._pending_results:
                self._pending_results[0] = HistGroupResult(
                    hist=self._pending_results[0].hist,
                    bin_edges=self._pending_results[0].bin_edges,
                    group_key=(),
                    count=self._pending_results[0].count,
                    weight_sum=self._pending_results[0].weight_sum,
                    empty_reason=self._pending_results[0].empty_reason,
                    normalization_denominator=self._pending_results[0].normalization_denominator,
                )
            for result in self._pending_results:
                self._diagnostics_collector.add_group_from_result(result)
            self._pending_results.clear()
            return

        grouper = grouping_vars[0] if len(grouping_vars) == 1 else grouping_vars
        group_iter = list(data.groupby(grouper, sort=False, observed=False))

        if len(group_iter) != len(self._pending_results):
            for result in self._pending_results:
                self._diagnostics_collector.add_group_from_result(result)
            self._pending_results.clear()
            return

        for (key, _), result in zip(group_iter, self._pending_results):
            gk = make_group_key(grouping_vars, key)
            keyed_result = HistGroupResult(
                hist=result.hist,
                bin_edges=result.bin_edges,
                group_key=gk,
                count=result.count,
                weight_sum=result.weight_sum,
                empty_reason=result.empty_reason,
                normalization_denominator=result.normalization_denominator,
            )
            self._diagnostics_collector.add_group_from_result(keyed_result)
        self._pending_results.clear()

    def _normalize(self, data):

        hist = data["count"].to_numpy().astype(float)
        space = data["space"].to_numpy()

        if self.stat == "probability" or self.stat == "proportion":
            total = hist.sum()
            hist = hist / total if total > 0 else hist
        elif self.stat == "percent":
            total = hist.sum()
            hist = hist / total * 100 if total > 0 else hist
        elif self.stat == "frequency":
            hist = hist / space

        if self.cumulative:
            if self.stat in ["density", "frequency"]:
                hist = (hist * space).cumsum()
            else:
                hist = hist.cumsum()

        return data.assign(**{self.stat: hist})

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        scale_type = scales[orient].__class__.__name__.lower()
        self._last_scale_type = scale_type

        self._diagnostics_collector = BinDiagnosticsCollector(
            stat=self.stat, cumulative=self.cumulative
        )
        self._pending_results = []

        grouping_vars = [str(v) for v in data if v in groupby.order]
        original_data = data

        if not grouping_vars or self.common_bins is True:
            common_bin_kws = self._define_bin_params(data, orient, scale_type)
            data = groupby.apply(data, self._eval, orient, common_bin_kws)
        else:
            if self.common_bins is False:
                bin_groupby = GroupBy(grouping_vars)
            else:
                bin_groupby = GroupBy(self.common_bins)
                self._check_grouping_vars("common_bins", grouping_vars)
            data = bin_groupby.apply(
                data, self._get_bins_and_eval, orient, groupby, scale_type,
            )

        if not grouping_vars or self.common_norm is True:
            data = self._normalize(data)
        else:
            if self.common_norm is False:
                norm_groupby = GroupBy(grouping_vars)
            else:
                norm_groupby = GroupBy(self.common_norm)
                self._check_grouping_vars("common_norm", grouping_vars)
            data = norm_groupby.apply(data, self._normalize)

        self._collect_diagnostics_from_pending(
            data=original_data,
            grouping_vars=grouping_vars,
            orient=orient,
        )

        other = {"x": "y", "y": "x"}[orient]
        return data.assign(**{other: data[self.stat]})
