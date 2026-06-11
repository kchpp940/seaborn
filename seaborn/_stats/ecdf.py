"""Empirical cumulative distribution function (ECDF) Stat class."""

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
    compute_ecdf,
    effective_weight_total,
    filter_valid_samples,
)


@dataclass
class ECDF(Stat):
    """
    Compute an empirical cumulative distribution function (ECDF).

    Parameters
    ----------
    stat : {"proportion", "percent", "count"}
        Statistic to compute at each unique value of ``orient``:

        - ``proportion``: empirical CDF in ``[0, 1]``.
        - ``percent``: empirical CDF scaled to ``[0, 100]``.
        - ``count``: weighted cumulative count (no normalization).

    complementary : bool
        If True, return the *complementary* ECDF (``1 - CDF``).
    common_norm : bool or list of variables
        For the ``proportion`` and ``percent`` statistics, this controls
        how the final value of the ECDF is normalized:

        - ``True`` (default): every hue group is normalized by the total
          weight of *all* groups together. The sum of the final
          cumulative values is 1 / 100.
        - ``False``: each hue group is independently normalized, so
          every ECDF goes from 0 to 1 / 100 on its own.
        - ``[var, ...]``: normalize within the sub-scopes defined by the
          listed variables.

        Ignored when ``stat == "count"``.

    Notes
    -----
    For the ``proportion`` / ``percent`` statistics the output is identical
    to :class:`seaborn._statistics.ECDF` so that the legacy plotting
    functions and the objects API agree on the same curves.

    Examples
    --------
    .. include:: ../docstrings/objects.ECDF.rst
    """
    stat: str = "proportion"
    complementary: bool = False
    common_norm: bool | list[str] = True

    group_by_orient: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self._check_param_one_of("stat", ["proportion", "percent", "count"])

    def _eval(self, data: DataFrame, orient: str) -> DataFrame:
        """Compute the ECDF for a single (hue-)group of data.

        Returns a DataFrame with columns:
          * ``orient``: the sorted sample values (plus a leading ``-inf``).
          * ``_total_weight``: effective total weight of this group.
          * ``ecdf``: the un-normalized weighted cumulative count up to
            the corresponding value. This is then normalized in
            :meth:`__call__` to honour ``common_norm``.
        """
        vals = data[orient].to_numpy(dtype=float)
        weights = (
            data["weight"].to_numpy(dtype=float)
            if "weight" in data.columns
            else np.ones_like(vals, dtype=float)
        )

        finite_mask = np.isfinite(vals) & np.isfinite(weights)
        vals = vals[finite_mask]
        weights = weights[finite_mask]

        total_weight = float(np.where(weights > 0, weights, 0.0).sum())

        if vals.size == 0:
            y_out = np.array([0.0, 0.0])
            x_out = np.array([-np.inf, np.inf])
        else:
            order = np.argsort(vals, kind="mergesort")
            x_sorted = vals[order]
            w_sorted = weights[order]
            y_raw = np.cumsum(w_sorted, dtype=float)
            y_out = np.r_[0.0, y_raw]
            x_out = np.r_[-np.inf, x_sorted]

        if self.complementary and y_out.size > 1:
            y_out = y_out[-1] - y_out

        return DataFrame({
            orient: x_out,
            "_total_weight": float(total_weight),
            "ecdf": y_out,
        })

    def _normalize(self, data: DataFrame, orient: str,
                   total_weight_lookup: dict[tuple, float] | None) -> DataFrame:
        """Normalize weighted cumulative counts to the requested stat.

        ``total_weight_lookup`` maps each hue-group's tuple to the
        effective total weight that should be used as the denominator.
        When the lookup is ``None`` (stat == "count", or common_norm is
        handled differently) the un-normalized cumulative values are
        returned as-is (then scaled for ``percent`` below).
        """
        grouping_cols = [
            c for c in data.columns
            if c not in (orient, "_total_weight", "ecdf")
        ]

        def _norm_one(part: DataFrame, denom: float | None) -> DataFrame:
            ecdf = part["ecdf"].to_numpy(dtype=float)

            if self.stat == "count":
                out = ecdf
            else:
                if denom is None:
                    # Fall back to per-group total.
                    denom = float(part["_total_weight"].iloc[0])
                if denom <= 0:
                    out = np.zeros_like(ecdf)
                else:
                    out = ecdf / float(denom)
                if self.stat == "percent":
                    out = out * 100.0

            return part.assign(**{self.stat: out})

        if grouping_cols and total_weight_lookup is not None:
            parts = []
            for key_tuple, part in data.groupby(grouping_cols, sort=False):
                k = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                denom = total_weight_lookup.get(k)
                parts.append(_norm_one(part, denom))
            return pd.concat(parts, ignore_index=True)
        elif grouping_cols:
            # common_norm=False: each group uses its own total.
            parts = [_norm_one(p, None) for _, p in data.groupby(grouping_cols, sort=False)]
            return pd.concat(parts, ignore_index=True)
        else:
            return _norm_one(data, total_weight_lookup.get(()) if total_weight_lookup else None)

    def __call__(
        self,
        data: DataFrame,
        groupby: GroupBy,
        orient: str,
        scales: dict[str, Scale],
    ) -> DataFrame:

        # 1. Filter non-finite data and weights.
        data = filter_valid_samples(data, orient)

        grouping_vars = [str(v) for v in data if v in groupby.order]

        if isinstance(self.common_norm, list):
            self._check_grouping_vars("common_norm", grouping_vars, stacklevel=3)

        # 2. Compute ECDF per hue group (returns un-normalized cumulative counts).
        if not grouping_vars:
            res = self._eval(data, orient)
        else:
            res = groupby.apply(data, self._eval, orient)

        # 3. Build the normalization lookup if stat needs one.
        lookup: dict[tuple, float] | None = None
        if self.stat != "count":
            sem_cols = [c for c in res.columns if c in grouping_vars]
            if not grouping_vars or self.common_norm is True:
                total_all = float(
                    res["_total_weight"].drop_duplicates().sum()
                    if "_total_weight" in res.columns else 0.0
                )
                lookup = {}
                if sem_cols:
                    for key_tuple, _ in res.groupby(sem_cols, sort=False):
                        k = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                        lookup[k] = total_all
                else:
                    lookup[()] = total_all
            elif self.common_norm is False:
                # Each group normalizes by its own total (the default of _norm_one).
                lookup = None
            else:
                # common_norm = [variable names]
                norm_vars = [v for v in self.common_norm if v in grouping_vars]
                norm_idx = [grouping_vars.index(v) for v in norm_vars]

                lookup = {}
                # First compute total per norm-group
                norm_totals: dict[tuple, float] = {}
                hue_totals: dict[tuple, float] = {}
                for key_tuple, part in res.groupby(sem_cols, sort=False):
                    k = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                    tw = float(part["_total_weight"].iloc[0])
                    hue_totals[k] = tw
                    nk = tuple(k[i] for i in norm_idx)
                    norm_totals[nk] = norm_totals.get(nk, 0.0) + tw

                for k_hue in hue_totals:
                    nk = tuple(k_hue[i] for i in norm_idx)
                    lookup[k_hue] = norm_totals.get(nk, 0.0)

        # 4. Normalize and clean up.
        res = self._normalize(res, orient, lookup)

        if "_total_weight" in res.columns:
            res = res.drop(columns=["_total_weight"])
        if "ecdf" in res.columns and "ecdf" != self.stat:
            res = res.drop(columns=["ecdf"])

        other = {"x": "y", "y": "x"}[orient]
        res[other] = res[self.stat]
        return res
