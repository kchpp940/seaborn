from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
try:
    from scipy.stats import gaussian_kde
    _no_scipy = False
except ImportError:
    from seaborn.external.kde import gaussian_kde
    _no_scipy = True

from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
from seaborn._stats.norm_utils import (
    effective_weight_total,
    filter_valid_samples,
    is_effectively_empty,
    normalize_kde_density,
)


@dataclass
class KDE(Stat):
    """
    Compute a univariate kernel density estimate.

    Parameters
    ----------
    bw_adjust : float
        Factor that multiplicatively scales the value chosen using
        `bw_method`. Increasing will make the curve smoother. See Notes.
    bw_method : string, scalar, or callable
        Method for determining the smoothing bandwidth to use. Passed directly
        to :class:`scipy.stats.gaussian_kde`; see there for options.
    common_norm : bool or list of variables
        If `True`, normalize so that the areas of all curves sums to 1.
        If `False`, normalize each curve independently. If a list, defines
        variable(s) to group by and normalize within.
    common_grid : bool or list of variables
        If `True`, all curves will share the same evaluation grid.
        If `False`, each evaluation grid is independent. If a list, defines
        variable(s) to group by and share a grid within.
    gridsize : int or None
        Number of points in the evaluation grid. If None, the density is
        evaluated at the original datapoints.
    cut : float
        Factor, multiplied by the kernel bandwidth, that determines how far
        the evaluation grid extends past the extreme datapoints. When set to 0,
        the curve is truncated at the data limits.
    cumulative : bool
        If True, estimate a cumulative distribution function. Requires scipy.

    Notes
    -----
    The *bandwidth*, or standard deviation of the smoothing kernel, is an
    important parameter. Much like histogram bin width, using the wrong
    bandwidth can produce a distorted representation. Over-smoothing can erase
    true features, while under-smoothing can create false ones. The default
    uses a rule-of-thumb that works best for distributions that are roughly
    bell-shaped. It is a good idea to check the default by varying `bw_adjust`.

    Because the smoothing is performed with a Gaussian kernel, the estimated
    density curve can extend to values that may not make sense. For example, the
    curve may be drawn over negative values when data that are naturally
    positive. The `cut` parameter can be used to control the evaluation range,
    but datasets that have many observations close to a natural boundary may be
    better served by a different method.

    Similar distortions may arise when a dataset is naturally discrete or "spiky"
    (containing many repeated observations of the same value). KDEs will always
    produce a smooth curve, which could be misleading.

    The units on the density axis are a common source of confusion. While kernel
    density estimation produces a probability distribution, the height of the curve
    at each point gives a density, not a probability. A probability can be obtained
    only by integrating the density across a range. The curve is normalized so
    that the integral over all possible values is 1, meaning that the scale of
    the density axis depends on the data values.

    If scipy is installed, its cython-accelerated implementation will be used.

    Examples
    --------
    .. include:: ../docstrings/objects.KDE.rst

    """
    bw_adjust: float = 1
    bw_method: str | float | Callable[[gaussian_kde], float] = "scott"
    common_norm: bool | list[str] = True
    common_grid: bool | list[str] = True
    gridsize: int | None = 200
    cut: float = 3
    cumulative: bool = False

    def __post_init__(self):

        if self.cumulative and _no_scipy:
            raise RuntimeError("Cumulative KDE evaluation requires scipy")

    def _check_var_list_or_boolean(self, param: str, grouping_vars: Any) -> None:
        """Do input checks on grouping parameters."""
        value = getattr(self, param)
        if not (
            isinstance(value, bool)
            or (isinstance(value, list) and all(isinstance(v, str) for v in value))
        ):
            param_name = f"{self.__class__.__name__}.{param}"
            raise TypeError(f"{param_name} must be a boolean or list of strings.")
        self._check_grouping_vars(param, grouping_vars, stacklevel=3)

    def _fit(self, data: DataFrame, orient: str) -> gaussian_kde:
        """Fit and return a KDE object."""

        vals = data[orient].to_numpy(dtype=float)
        weights = data["weight"].to_numpy(dtype=float) if "weight" in data.columns else None

        # Guard against singular data (all identical values)
        if vals.size < 2 or np.nanstd(vals) == 0:
            raise np.linalg.LinAlgError("Singular KDE input")

        fit_kws: dict[str, Any] = {"bw_method": self.bw_method}
        if weights is not None:
            # Skip zero-weight samples – they confuse scipy's KDE covariance.
            pos_mask = np.isfinite(weights) & (weights > 0)
            if pos_mask.sum() < 2:
                raise np.linalg.LinAlgError("Fewer than 2 positive-weight samples")
            vals = vals[pos_mask]
            weights = weights[pos_mask]
            fit_kws["weights"] = weights

        kde = gaussian_kde(vals, **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        return kde

    def _get_support(self, data: DataFrame, orient: str) -> ndarray:
        """Define the grid that the KDE will be evaluated on."""
        if self.gridsize is None:
            return data[orient].to_numpy()

        kde = self._fit(data, orient)
        bw = np.sqrt(kde.covariance.squeeze())
        gridmin = data[orient].min() - bw * self.cut
        gridmax = data[orient].max() + bw * self.cut
        return np.linspace(gridmin, gridmax, self.gridsize)

    def _fit_and_evaluate(
        self, data: DataFrame, orient: str, support: ndarray
    ) -> DataFrame:
        """Transform single group by fitting a KDE and evaluating on a support grid.

        Returns a DataFrame with columns ``[orient, "_group_weight", "density"]``.
        For singular / empty groups we return an empty DataFrame (0 rows)
        so that callers can decide whether to drop them or re-insert a
        zero-density curve for stacking compatibility.
        """
        group_weight = effective_weight_total(
            data["weight"].to_numpy(dtype=float) if "weight" in data.columns
            else np.ones(len(data), dtype=float)
        )

        empty_out = DataFrame(
            columns=[orient, "_group_weight", "density"], dtype=float,
        )

        # Skip fitting for effectively empty groups.
        vals = data[orient].to_numpy(dtype=float)
        weights = (
            data["weight"].to_numpy(dtype=float) if "weight" in data.columns
            else np.ones_like(vals, dtype=float)
        )
        finite_mask = np.isfinite(vals) & np.isfinite(weights)
        pos_mask = finite_mask & (weights > 0)
        n_pos = int(pos_mask.sum())
        if n_pos < 2 or np.nanstd(vals[finite_mask]) == 0:
            return empty_out

        try:
            kde = self._fit(data, orient)
        except np.linalg.LinAlgError:
            return empty_out
        except ValueError as e:
            # scipy may raise ValueError on pathological inputs
            if "array must not contain" in str(e) or "fin" in str(e).lower():
                return empty_out
            raise

        if self.cumulative:
            s_0 = support[0]
            density = np.array(
                [kde.integrate_box_1d(s_0, s_i) for s_i in support],
                dtype=float,
            )
        else:
            density = np.asarray(kde(support), dtype=float)

        # Replace any NaN/Inf from scipy with zeros.
        density = np.where(np.isfinite(density), density, 0.0)

        return DataFrame({
            orient: support,
            "_group_weight": float(group_weight),
            "density": density,
        })

    def _transform(
        self, data: DataFrame, orient: str, grouping_vars: list[str]
    ) -> DataFrame:
        """Transform multiple groups by fitting KDEs and evaluating."""
        # Filter non-finite samples early.
        data = filter_valid_samples(data, orient)

        empty = DataFrame(
            columns=[*data.columns, "_group_weight", "density"], dtype=float,
        )
        if len(data) < 2:
            return empty

        # Define support on the norm-group level (common_grid).
        try:
            support = self._get_support(data, orient)
        except (np.linalg.LinAlgError, ValueError):
            # If support can't be built, skip this whole norm-group.
            return empty

        grouping_vars_active = [x for x in grouping_vars if data[x].nunique() > 1]
        if not grouping_vars_active:
            return self._fit_and_evaluate(data, orient, support)
        groupby = GroupBy(grouping_vars_active)
        return groupby.apply(data, self._fit_and_evaluate, orient, support)

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        # 1. Ensure weight exists, filter non-finite samples and weights.
        data = filter_valid_samples(data, orient)
        if "weight" not in data.columns:
            data = data.assign(weight=1.0)

        grouping_vars = [str(v) for v in data if v in groupby.order]

        # 2. Transform (fit + evaluate) each grid-group.
        if not grouping_vars or self.common_grid is True:
            res = self._transform(data, orient, grouping_vars)
        else:
            if self.common_grid is False:
                grid_vars = grouping_vars
            else:
                self._check_var_list_or_boolean("common_grid", grouping_vars)
                grid_vars = [v for v in self.common_grid if v in grouping_vars]

            res = (
                GroupBy(grid_vars)
                .apply(data, self._transform, orient, grouping_vars)
            )

        # 3. Normalization.
        #    _fit_and_evaluate already returns a raw KDE whose integral is 1
        #    per *original hue group*. We need to scale it so that:
        #    * common_norm=True:  sum of all hue integrals = 1
        #    * common_norm=False: each hue integral = 1
        #    * common_norm=[vars]: hue groups in the same norm-group share
        #                         the integral 1 across that norm-group.
        if res.empty:
            value = {"x": "y", "y": "x"}[orient]
            res[value] = res.get("density", pd.Series(dtype=float))
            return res.drop(columns=[c for c in ("_group_weight", "density") if c in res.columns], errors="ignore")

        sem_cols = [c for c in res.columns if c in grouping_vars]

        # -- Build norm_total lookup: hue-group tuple -> total weight of its norm group.
        if not grouping_vars or self.common_norm is True:
            # Single norm-group (everything together).
            norm_total_all = float(res["_group_weight"].drop_duplicates().sum())
            def _norm_total(_key):
                return norm_total_all
        elif self.common_norm is False:
            # Each hue group is its own norm-group.
            def _norm_total(key):
                # Find the _group_weight of the (single) hue group matching key.
                q = True
                for i, col in enumerate(sem_cols):
                    q &= (res[col] == key[i])
                match = res.loc[q, "_group_weight"].drop_duplicates()
                return float(match.iloc[0]) if len(match) else 0.0
        else:
            norm_vars = [v for v in self.common_norm if v in grouping_vars]
            self._check_var_list_or_boolean("common_norm", grouping_vars)
            norm_idx = [sem_cols.index(v) for v in norm_vars if v in sem_cols]

            # Pre-compute total per norm-group.
            norm_totals: dict[tuple, float] = {}
            for key_tuple, part in res.groupby(sem_cols, sort=False):
                k = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                nk = tuple(k[i] for i in norm_idx)
                gw = float(part["_group_weight"].iloc[0])
                norm_totals[nk] = norm_totals.get(nk, 0.0) + gw

            def _norm_total(key):
                nk = tuple(key[i] for i in norm_idx)
                return norm_totals.get(nk, 0.0)

        # Apply per-hue-group scaling.
        if not sem_cols:
            gw = float(res["_group_weight"].iloc[0])
            nt = float(_norm_total(()))
            res["density"] = normalize_kde_density(
                res["density"].to_numpy(dtype=float),
                res[orient].to_numpy(dtype=float),
                gw, nt,
            )
        else:
            parts = []
            for key_tuple, part in res.groupby(sem_cols, sort=False):
                k = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                gw = float(part["_group_weight"].iloc[0])
                nt = float(_norm_total(k))
                part = part.copy()
                part["density"] = normalize_kde_density(
                    part["density"].to_numpy(dtype=float),
                    part[orient].to_numpy(dtype=float),
                    gw, nt,
                )
                parts.append(part)
            res = pd.concat(parts, ignore_index=True)

        value = {"x": "y", "y": "x"}[orient]
        res[value] = res["density"]
        return res.drop(columns=["_group_weight", "weight"], errors="ignore")
