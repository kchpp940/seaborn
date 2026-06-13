from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from pandas import DataFrame

from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
from seaborn._statistics import BinDiagnostics, BinDiagnosticsCollector

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
        """Evaluate histogram for a single group.

        This method is intentionally minimal - it computes the histogram
        and returns the result. Diagnostics are collected separately in
        __call__ using the shared BinDiagnosticsCollector.
        """
        vals = data[orient]
        weights = data.get("weight", None)

        density = self.stat == "density"
        hist, edges = np.histogram(vals, **bin_kws, weights=weights, density=density)

        width = np.diff(edges)
        center = edges[:-1] + width / 2

        return pd.DataFrame({orient: center, "count": hist, "space": width})

    def _compute_bin_kws_for_groups(
        self, data, groupby_vars, orient, common_bins,
    ):
        """Compute bin parameters for each group and return as a dict.

        This is used by diagnostics collection to ensure bin parameters
        match exactly what was used during the actual histogram computation.

        Parameters
        ----------
        data : DataFrame
            Input data with group columns present.
        groupby_vars : list of str
            Names of the full grouping variables.
        orient : str
            Orientation variable name ('x' or 'y').
        common_bins : bool or list of str
            The common_bins parameter.

        Returns
        -------
        dict
            Mapping from group_key tuple to bin_kws dict.
        """
        bin_kws_map = {}

        if not groupby_vars:
            # Single group case
            bin_kws = self._define_bin_params(data, orient, self._last_scale_type)
            bin_kws_map[()] = bin_kws
            return bin_kws_map

        # Determine which variables define the bin groups
        if common_bins is True:
            # All groups share the same bin parameters
            common_bin_kws = self._define_bin_params(data, orient, self._last_scale_type)
            grouper = groupby_vars[0] if len(groupby_vars) == 1 else groupby_vars
            for key, _ in data.groupby(grouper, sort=False, observed=False):
                if isinstance(grouper, list):
                    group_key = tuple(zip(groupby_vars, key))
                else:
                    group_key = ((groupby_vars[0], key),)
                bin_kws_map[group_key] = common_bin_kws
        elif common_bins is False:
            # Each group has its own bin parameters
            grouper = groupby_vars[0] if len(groupby_vars) == 1 else groupby_vars
            for key, part_df in data.groupby(grouper, sort=False, observed=False):
                if isinstance(grouper, list):
                    group_key = tuple(zip(groupby_vars, key))
                else:
                    group_key = ((groupby_vars[0], key),)
                bin_kws = self._define_bin_params(part_df, orient, self._last_scale_type)
                bin_kws_map[group_key] = bin_kws
        else:
            # common_bins is a list of variable names - share bins within those groups
            bin_group_vars = [v for v in common_bins if v in groupby_vars]
            if not bin_group_vars:
                # No bin-grouping vars in grouping vars -> fall back to per-group
                bin_group_vars = groupby_vars

            # First compute bin params per bin-group
            bin_grouper = bin_group_vars[0] if len(bin_group_vars) == 1 else bin_group_vars
            bin_kws_by_bin_group = {}
            for bin_key, bin_part_df in data.groupby(bin_grouper, sort=False, observed=False):
                bin_kws = self._define_bin_params(bin_part_df, orient, self._last_scale_type)
                if isinstance(bin_grouper, list):
                    bin_key_tuple = tuple(zip(bin_group_vars, bin_key))
                else:
                    bin_key_tuple = ((bin_group_vars[0], bin_key),)
                bin_kws_by_bin_group[bin_key_tuple] = bin_kws

            # Then map each full group to its bin params
            grouper = groupby_vars[0] if len(groupby_vars) == 1 else groupby_vars
            for key, part_df in data.groupby(grouper, sort=False, observed=False):
                if isinstance(grouper, list):
                    group_key = tuple(zip(groupby_vars, key))
                    group_dict = dict(group_key)
                else:
                    group_key = ((groupby_vars[0], key),)
                    group_dict = {groupby_vars[0]: key}

                # Find the matching bin group key
                if isinstance(bin_grouper, list):
                    bin_key_tuple = tuple((v, group_dict[v]) for v in bin_group_vars)
                else:
                    bin_key_tuple = ((bin_group_vars[0], group_dict[bin_group_vars[0]]),)

                bin_kws_map[group_key] = bin_kws_by_bin_group.get(
                    bin_key_tuple,
                    self._define_bin_params(part_df, orient, self._last_scale_type)
                )

        return bin_kws_map

    def _collect_diagnostics_for_groups(
        self, data, groupby_vars, orient, bin_kws_map,
    ):
        """Collect diagnostics for each group using the unified collector.

        This method walks the original data (not the histogram output)
        and computes diagnostics for each group. It runs after the main
        histogram computation to avoid interfering with GroupBy.apply.

        Parameters
        ----------
        data : DataFrame
            Original input data with group columns present.
        groupby_vars : list of str
            Names of the grouping variables.
        orient : str
            Orientation variable name ('x' or 'y').
        bin_kws_map : dict
            Mapping from group_key tuple to bin_kws dict.
        """
        if not groupby_vars:
            # Single group case
            bin_kws = bin_kws_map.get(())
            if bin_kws is None:
                bin_kws = self._define_bin_params(data, orient, self._last_scale_type)

            vals = data[orient]
            weights = data.get("weight", None)
            density = self.stat == "density"
            hist, edges = np.histogram(
                vals, **bin_kws, weights=weights, density=density,
            )
            # Apply normalization to match the final output
            hist = self._normalize_array(hist, edges)

            self._diagnostics_collector.add_group_univariate(
                group_key=(),
                x=vals,
                bin_edges=edges,
                hist=hist,
                weights=weights,
            )
            return

        # Multi-group case
        grouper = groupby_vars[0] if len(groupby_vars) == 1 else groupby_vars
        for key, part_df in data.groupby(grouper, sort=False, observed=False):
            if isinstance(grouper, list):
                group_key = tuple(zip(groupby_vars, key))
            else:
                group_key = ((groupby_vars[0], key),)

            bin_kws = bin_kws_map.get(group_key)
            if bin_kws is None:
                bin_kws = self._define_bin_params(part_df, orient, self._last_scale_type)

            vals = part_df[orient]
            weights = part_df.get("weight", None)
            density = self.stat == "density"
            hist, edges = np.histogram(
                vals, **bin_kws, weights=weights, density=density,
            )
            # Apply normalization to match the final output
            hist = self._normalize_array(hist, edges)

            self._diagnostics_collector.add_group_univariate(
                group_key=group_key,
                x=vals,
                bin_edges=edges,
                hist=hist,
                weights=weights,
            )

    def _normalize_array(self, hist, edges):
        """Apply normalization to a raw histogram count array.

        Mirrors the normalization logic in _normalize but for raw arrays.
        Used for diagnostics to match the final output histogram values.
        """
        width = np.diff(edges) if edges is not None else None

        if self.stat == "probability" or self.stat == "proportion":
            hist = hist.astype(float) / hist.sum() if hist.sum() > 0 else hist.astype(float)
        elif self.stat == "percent":
            hist = hist.astype(float) / hist.sum() * 100 if hist.sum() > 0 else hist.astype(float)
        elif self.stat == "frequency":
            hist = hist.astype(float) / width if width is not None else hist.astype(float)
        # Note: density is already applied by np.histogram(density=True)

        if self.cumulative:
            if self.stat in ["density", "frequency"] and width is not None:
                hist = (hist * width).cumsum()
            else:
                hist = hist.cumsum()

        return hist

    def _normalize(self, data):

        hist = data["count"]
        if self.stat == "probability" or self.stat == "proportion":
            hist = hist.astype(float) / hist.sum()
        elif self.stat == "percent":
            hist = hist.astype(float) / hist.sum() * 100
        elif self.stat == "frequency":
            hist = hist.astype(float) / data["space"]

        if self.cumulative:
            if self.stat in ["density", "frequency"]:
                hist = (hist * data["space"]).cumsum()
            else:
                hist = hist.cumsum()

        return data.assign(**{self.stat: hist})

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        scale_type = scales[orient].__class__.__name__.lower()
        self._last_scale_type = scale_type

        # Reset the diagnostics collector for this call
        # Use clear() instead of recreating to preserve external references
        # (single source of truth across layers)
        self._diagnostics_collector.stat = self.stat
        self._diagnostics_collector.cumulative = self.cumulative
        self._diagnostics_collector.clear()

        grouping_vars = [str(v) for v in data if v in groupby.order]
        # Keep a reference to the original input data for diagnostics
        original_data = data

        # --- Compute bin parameters (and track for diagnostics) ---
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
            common_bin_kws = None  # per-group bins

        # --- Normalize ---
        if not grouping_vars or self.common_norm is True:
            data = self._normalize(data)
        else:
            if self.common_norm is False:
                norm_groupby = GroupBy(grouping_vars)
            else:
                norm_groupby = GroupBy(self.common_norm)
                self._check_grouping_vars("common_norm", grouping_vars)
            data = norm_groupby.apply(data, self._normalize)

        # --- Collect diagnostics using original input data ---
        # First compute bin params for each group (matching actual computation)
        bin_kws_map = self._compute_bin_kws_for_groups(
            data=original_data,
            groupby_vars=grouping_vars,
            orient=orient,
            common_bins=self.common_bins,
        )
        # Then collect diagnostics using those bin params
        self._collect_diagnostics_for_groups(
            data=original_data,
            groupby_vars=grouping_vars,
            orient=orient,
            bin_kws_map=bin_kws_map,
        )

        other = {"x": "y", "y": "x"}[orient]
        return data.assign(**{other: data[self.stat]})
