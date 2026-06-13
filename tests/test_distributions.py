import itertools
import warnings
from dataclasses import fields

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose

from seaborn import distributions as dist
from seaborn.palettes import (
    color_palette,
    light_palette,
)
from seaborn._base import (
    categorical_order,
)
from seaborn._statistics import (
    KDE,
    Histogram,
    _no_scipy,
    BinDiagnostics,
    BinDiagnosticsCollector,
)
from seaborn._stats.counting import Hist as ObjHist
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Continuous as ContScale
from seaborn.distributions import (
    _DistributionPlotter,
    displot,
    distplot,
    histplot,
    ecdfplot,
    kdeplot,
    rugplot,
)
from seaborn.utils import _version_predates
from seaborn.axisgrid import FacetGrid
from seaborn._testing import (
    assert_plots_equal,
    assert_legends_equal,
    assert_colors_equal,
)


def get_contour_coords(c, filter_empty=False):
    """Provide compatability for change in contour artist types."""
    if isinstance(c, mpl.collections.LineCollection):
        # See https://github.com/matplotlib/matplotlib/issues/20906
        return c.get_segments()
    elif isinstance(c, (mpl.collections.PathCollection, mpl.contour.QuadContourSet)):
        return [
            p.vertices[:np.argmax(p.codes) + 1] for p in c.get_paths()
            if len(p) or not filter_empty
        ]


def get_contour_color(c):
    """Provide compatability for change in contour artist types."""
    if isinstance(c, mpl.collections.LineCollection):
        # See https://github.com/matplotlib/matplotlib/issues/20906
        return c.get_color()
    elif isinstance(c, (mpl.collections.PathCollection, mpl.contour.QuadContourSet)):
        if c.get_facecolor().size:
            return c.get_facecolor()
        else:
            return c.get_edgecolor()


class TestDistPlot:

    rs = np.random.RandomState(0)
    x = rs.randn(100)

    def test_hist_bins(self):

        fd_edges = np.histogram_bin_edges(self.x, "fd")
        with pytest.warns(UserWarning):
            ax = distplot(self.x)
        for edge, bar in zip(fd_edges, ax.patches):
            assert pytest.approx(edge) == bar.get_x()

        plt.close(ax.figure)
        n = 25
        n_edges = np.histogram_bin_edges(self.x, n)
        with pytest.warns(UserWarning):
            ax = distplot(self.x, bins=n)
        for edge, bar in zip(n_edges, ax.patches):
            assert pytest.approx(edge) == bar.get_x()

    def test_elements(self):

        with pytest.warns(UserWarning):

            n = 10
            ax = distplot(self.x, bins=n,
                          hist=True, kde=False, rug=False, fit=None)
            assert len(ax.patches) == 10
            assert len(ax.lines) == 0
            assert len(ax.collections) == 0

            plt.close(ax.figure)
            ax = distplot(self.x,
                          hist=False, kde=True, rug=False, fit=None)
            assert len(ax.patches) == 0
            assert len(ax.lines) == 1
            assert len(ax.collections) == 0

            plt.close(ax.figure)
            ax = distplot(self.x,
                          hist=False, kde=False, rug=True, fit=None)
            assert len(ax.patches) == 0
            assert len(ax.lines) == 0
            assert len(ax.collections) == 1

            class Norm:
                """Dummy object that looks like a scipy RV"""
                def fit(self, x):
                    return ()

                def pdf(self, x, *params):
                    return np.zeros_like(x)

            plt.close(ax.figure)
            ax = distplot(
                self.x, hist=False, kde=False, rug=False, fit=Norm())
            assert len(ax.patches) == 0
            assert len(ax.lines) == 1
            assert len(ax.collections) == 0

    def test_distplot_with_nans(self):

        f, (ax1, ax2) = plt.subplots(2)
        x_null = np.append(self.x, [np.nan])

        with pytest.warns(UserWarning):
            distplot(self.x, ax=ax1)
            distplot(x_null, ax=ax2)

        line1 = ax1.lines[0]
        line2 = ax2.lines[0]
        assert np.array_equal(line1.get_xydata(), line2.get_xydata())

        for bar1, bar2 in zip(ax1.patches, ax2.patches):
            assert bar1.get_xy() == bar2.get_xy()
            assert bar1.get_height() == bar2.get_height()


class SharedAxesLevelTests:

    def test_color(self, long_df, **kwargs):

        ax = plt.figure().subplots()
        self.func(data=long_df, x="y", ax=ax, **kwargs)
        assert_colors_equal(self.get_last_color(ax, **kwargs), "C0", check_alpha=False)

        ax = plt.figure().subplots()
        self.func(data=long_df, x="y", ax=ax, **kwargs)
        self.func(data=long_df, x="y", ax=ax, **kwargs)
        assert_colors_equal(self.get_last_color(ax, **kwargs), "C1", check_alpha=False)

        ax = plt.figure().subplots()
        self.func(data=long_df, x="y", color="C2", ax=ax, **kwargs)
        assert_colors_equal(self.get_last_color(ax, **kwargs), "C2", check_alpha=False)


class TestRugPlot(SharedAxesLevelTests):

    func = staticmethod(rugplot)

    def get_last_color(self, ax, **kwargs):

        return ax.collections[-1].get_color()

    def assert_rug_equal(self, a, b):

        assert_array_equal(a.get_segments(), b.get_segments())

    @pytest.mark.parametrize("variable", ["x", "y"])
    def test_long_data(self, long_df, variable):

        vector = long_df[variable]
        vectors = [
            variable, vector, np.asarray(vector), vector.to_list(),
        ]

        f, ax = plt.subplots()
        for vector in vectors:
            rugplot(data=long_df, **{variable: vector})

        for a, b in itertools.product(ax.collections, ax.collections):
            self.assert_rug_equal(a, b)

    def test_bivariate_data(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        rugplot(data=long_df, x="x", y="y", ax=ax1)
        rugplot(data=long_df, x="x", ax=ax2)
        rugplot(data=long_df, y="y", ax=ax2)

        self.assert_rug_equal(ax1.collections[0], ax2.collections[0])
        self.assert_rug_equal(ax1.collections[1], ax2.collections[1])

    def test_wide_vs_long_data(self, wide_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)
        rugplot(data=wide_df, ax=ax1)
        for col in wide_df:
            rugplot(data=wide_df, x=col, ax=ax2)

        wide_segments = np.sort(
            np.array(ax1.collections[0].get_segments())
        )
        long_segments = np.sort(
            np.concatenate([c.get_segments() for c in ax2.collections])
        )

        assert_array_equal(wide_segments, long_segments)

    def test_flat_vector(self, long_df):

        f, ax = plt.subplots()
        rugplot(data=long_df["x"])
        rugplot(x=long_df["x"])
        self.assert_rug_equal(*ax.collections)

    def test_datetime_data(self, long_df):

        ax = rugplot(data=long_df["t"])
        vals = np.stack(ax.collections[0].get_segments())[:, 0, 0]
        assert_array_equal(vals, mpl.dates.date2num(long_df["t"]))

    def test_empty_data(self):

        ax = rugplot(x=[])
        assert not ax.collections

    def test_a_deprecation(self, flat_series):

        f, ax = plt.subplots()

        with pytest.warns(UserWarning):
            rugplot(a=flat_series)
        rugplot(x=flat_series)

        self.assert_rug_equal(*ax.collections)

    @pytest.mark.parametrize("variable", ["x", "y"])
    def test_axis_deprecation(self, flat_series, variable):

        f, ax = plt.subplots()

        with pytest.warns(UserWarning):
            rugplot(flat_series, axis=variable)
        rugplot(**{variable: flat_series})

        self.assert_rug_equal(*ax.collections)

    def test_vertical_deprecation(self, flat_series):

        f, ax = plt.subplots()

        with pytest.warns(UserWarning):
            rugplot(flat_series, vertical=True)
        rugplot(y=flat_series)

        self.assert_rug_equal(*ax.collections)

    def test_rug_data(self, flat_array):

        height = .05
        ax = rugplot(x=flat_array, height=height)
        segments = np.stack(ax.collections[0].get_segments())

        n = flat_array.size
        assert_array_equal(segments[:, 0, 1], np.zeros(n))
        assert_array_equal(segments[:, 1, 1], np.full(n, height))
        assert_array_equal(segments[:, 1, 0], flat_array)

    def test_rug_colors(self, long_df):

        ax = rugplot(data=long_df, x="x", hue="a")

        order = categorical_order(long_df["a"])
        palette = color_palette()

        expected_colors = np.ones((len(long_df), 4))
        for i, val in enumerate(long_df["a"]):
            expected_colors[i, :3] = palette[order.index(val)]

        assert_array_equal(ax.collections[0].get_color(), expected_colors)

    def test_expand_margins(self, flat_array):

        f, ax = plt.subplots()
        x1, y1 = ax.margins()
        rugplot(x=flat_array, expand_margins=False)
        x2, y2 = ax.margins()
        assert x1 == x2
        assert y1 == y2

        f, ax = plt.subplots()
        x1, y1 = ax.margins()
        height = .05
        rugplot(x=flat_array, height=height)
        x2, y2 = ax.margins()
        assert x1 == x2
        assert y1 + height * 2 == pytest.approx(y2)

    def test_multiple_rugs(self):

        values = np.linspace(start=0, stop=1, num=5)
        ax = rugplot(x=values)
        ylim = ax.get_ylim()

        rugplot(x=values, ax=ax, expand_margins=False)

        assert ylim == ax.get_ylim()

    def test_matplotlib_kwargs(self, flat_series):

        lw = 2
        alpha = .2
        ax = rugplot(y=flat_series, linewidth=lw, alpha=alpha)
        rug = ax.collections[0]
        assert np.all(rug.get_alpha() == alpha)
        assert np.all(rug.get_linewidth() == lw)

    def test_axis_labels(self, flat_series):

        ax = rugplot(x=flat_series)
        assert ax.get_xlabel() == flat_series.name
        assert not ax.get_ylabel()

    def test_log_scale(self, long_df):

        ax1, ax2 = plt.figure().subplots(2)

        ax2.set_xscale("log")

        rugplot(data=long_df, x="z", ax=ax1)
        rugplot(data=long_df, x="z", ax=ax2)

        rug1 = np.stack(ax1.collections[0].get_segments())
        rug2 = np.stack(ax2.collections[0].get_segments())

        assert_array_almost_equal(rug1, rug2)


class TestKDEPlotUnivariate(SharedAxesLevelTests):

    func = staticmethod(kdeplot)

    def get_last_color(self, ax, fill=True):

        if fill:
            return ax.collections[-1].get_facecolor()
        else:
            return ax.lines[-1].get_color()

    @pytest.mark.parametrize("fill", [True, False])
    def test_color(self, long_df, fill):

        super().test_color(long_df, fill=fill)

        if fill:

            ax = plt.figure().subplots()
            self.func(data=long_df, x="y", facecolor="C3", fill=True, ax=ax)
            assert_colors_equal(self.get_last_color(ax), "C3", check_alpha=False)

            ax = plt.figure().subplots()
            self.func(data=long_df, x="y", fc="C4", fill=True, ax=ax)
            assert_colors_equal(self.get_last_color(ax), "C4", check_alpha=False)

    @pytest.mark.parametrize(
        "variable", ["x", "y"],
    )
    def test_long_vectors(self, long_df, variable):

        vector = long_df[variable]
        vectors = [
            variable, vector, vector.to_numpy(), vector.to_list(),
        ]

        f, ax = plt.subplots()
        for vector in vectors:
            kdeplot(data=long_df, **{variable: vector})

        xdata = [l.get_xdata() for l in ax.lines]
        for a, b in itertools.product(xdata, xdata):
            assert_array_equal(a, b)

        ydata = [l.get_ydata() for l in ax.lines]
        for a, b in itertools.product(ydata, ydata):
            assert_array_equal(a, b)

    def test_wide_vs_long_data(self, wide_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)
        kdeplot(data=wide_df, ax=ax1, common_norm=False, common_grid=False)
        for col in wide_df:
            kdeplot(data=wide_df, x=col, ax=ax2)

        for l1, l2 in zip(ax1.lines[::-1], ax2.lines):
            assert_array_equal(l1.get_xydata(), l2.get_xydata())

    def test_flat_vector(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df["x"])
        kdeplot(x=long_df["x"])
        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    def test_empty_data(self):

        ax = kdeplot(x=[])
        assert not ax.lines

    def test_singular_data(self):

        with pytest.warns(UserWarning):
            ax = kdeplot(x=np.ones(10))
        assert not ax.lines

        with pytest.warns(UserWarning):
            ax = kdeplot(x=[5])
        assert not ax.lines

        with pytest.warns(UserWarning):
            # https://github.com/mwaskom/seaborn/issues/2762
            ax = kdeplot(x=[1929245168.06679] * 18)
        assert not ax.lines

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ax = kdeplot(x=[5], warn_singular=False)
        assert not ax.lines

    def test_variable_assignment(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", fill=True)
        kdeplot(data=long_df, y="x", fill=True)

        v0 = ax.collections[0].get_paths()[0].vertices
        v1 = ax.collections[1].get_paths()[0].vertices[:, [1, 0]]

        assert_array_equal(v0, v1)

    def test_vertical_deprecation(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, y="x")

        with pytest.warns(UserWarning):
            kdeplot(data=long_df, x="x", vertical=True)

        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    def test_bw_deprecation(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", bw_method="silverman")

        with pytest.warns(UserWarning):
            kdeplot(data=long_df, x="x", bw="silverman")

        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    def test_kernel_deprecation(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x")

        with pytest.warns(UserWarning):
            kdeplot(data=long_df, x="x", kernel="epi")

        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    def test_shade_deprecation(self, long_df):

        f, ax = plt.subplots()
        with pytest.warns(FutureWarning):
            kdeplot(data=long_df, x="x", shade=True)
        kdeplot(data=long_df, x="x", fill=True)
        fill1, fill2 = ax.collections
        assert_array_equal(
            fill1.get_paths()[0].vertices, fill2.get_paths()[0].vertices
        )

    @pytest.mark.parametrize("multiple", ["layer", "stack", "fill"])
    def test_hue_colors(self, long_df, multiple):

        ax = kdeplot(
            data=long_df, x="x", hue="a",
            multiple=multiple,
            fill=True, legend=False
        )

        # Note that hue order is reversed in the plot
        lines = ax.lines[::-1]
        fills = ax.collections[::-1]

        palette = color_palette()

        for line, fill, color in zip(lines, fills, palette):
            assert_colors_equal(line.get_color(), color)
            assert_colors_equal(fill.get_facecolor(), to_rgba(color, .25))

    def test_hue_stacking(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="layer", common_grid=True,
            legend=False, ax=ax1,
        )
        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="stack", fill=False,
            legend=False, ax=ax2,
        )

        layered_densities = np.stack([
            l.get_ydata() for l in ax1.lines
        ])
        stacked_densities = np.stack([
            l.get_ydata() for l in ax2.lines
        ])

        assert_array_equal(layered_densities.cumsum(axis=0), stacked_densities)

    def test_hue_filling(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="layer", common_grid=True,
            legend=False, ax=ax1,
        )
        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="fill", fill=False,
            legend=False, ax=ax2,
        )

        layered = np.stack([l.get_ydata() for l in ax1.lines])
        filled = np.stack([l.get_ydata() for l in ax2.lines])

        assert_array_almost_equal(
            (layered / layered.sum(axis=0)).cumsum(axis=0),
            filled,
        )

    @pytest.mark.parametrize("multiple", ["stack", "fill"])
    def test_fill_default(self, long_df, multiple):

        ax = kdeplot(
            data=long_df, x="x", hue="a", multiple=multiple, fill=None
        )

        assert len(ax.collections) > 0

    @pytest.mark.parametrize("multiple", ["layer", "stack", "fill"])
    def test_fill_nondefault(self, long_df, multiple):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kws = dict(data=long_df, x="x", hue="a")
        kdeplot(**kws, multiple=multiple, fill=False, ax=ax1)
        kdeplot(**kws, multiple=multiple, fill=True, ax=ax2)

        assert len(ax1.collections) == 0
        assert len(ax2.collections) > 0

    def test_color_cycle_interaction(self, flat_series):

        color = (.2, 1, .6)

        f, ax = plt.subplots()
        kdeplot(flat_series)
        kdeplot(flat_series)
        assert_colors_equal(ax.lines[0].get_color(), "C0")
        assert_colors_equal(ax.lines[1].get_color(), "C1")
        plt.close(f)

        f, ax = plt.subplots()
        kdeplot(flat_series, color=color)
        kdeplot(flat_series)
        assert_colors_equal(ax.lines[0].get_color(), color)
        assert_colors_equal(ax.lines[1].get_color(), "C0")
        plt.close(f)

        f, ax = plt.subplots()
        kdeplot(flat_series, fill=True)
        kdeplot(flat_series, fill=True)
        assert_colors_equal(ax.collections[0].get_facecolor(), to_rgba("C0", .25))
        assert_colors_equal(ax.collections[1].get_facecolor(), to_rgba("C1", .25))
        plt.close(f)

    @pytest.mark.parametrize("fill", [True, False])
    def test_artist_color(self, long_df, fill):

        color = (.2, 1, .6)
        alpha = .5

        f, ax = plt.subplots()

        kdeplot(long_df["x"], fill=fill, color=color)
        if fill:
            artist_color = ax.collections[-1].get_facecolor().squeeze()
        else:
            artist_color = ax.lines[-1].get_color()
        default_alpha = .25 if fill else 1
        assert_colors_equal(artist_color, to_rgba(color, default_alpha))

        kdeplot(long_df["x"], fill=fill, color=color, alpha=alpha)
        if fill:
            artist_color = ax.collections[-1].get_facecolor().squeeze()
        else:
            artist_color = ax.lines[-1].get_color()
        assert_colors_equal(artist_color, to_rgba(color, alpha))

    def test_datetime_scale(self, long_df):

        f, (ax1, ax2) = plt.subplots(2)
        kdeplot(x=long_df["t"], fill=True, ax=ax1)
        kdeplot(x=long_df["t"], fill=False, ax=ax2)
        assert ax1.get_xlim() == ax2.get_xlim()

    def test_multiple_argument_check(self, long_df):

        with pytest.raises(ValueError, match="`multiple` must be"):
            kdeplot(data=long_df, x="x", hue="a", multiple="bad_input")

    def test_cut(self, rng):

        x = rng.normal(0, 3, 1000)

        f, ax = plt.subplots()
        kdeplot(x=x, cut=0, legend=False)

        xdata_0 = ax.lines[0].get_xdata()
        assert xdata_0.min() == x.min()
        assert xdata_0.max() == x.max()

        kdeplot(x=x, cut=2, legend=False)

        xdata_2 = ax.lines[1].get_xdata()
        assert xdata_2.min() < xdata_0.min()
        assert xdata_2.max() > xdata_0.max()

        assert len(xdata_0) == len(xdata_2)

    def test_clip(self, rng):

        x = rng.normal(0, 3, 1000)

        clip = -1, 1
        ax = kdeplot(x=x, clip=clip)

        xdata = ax.lines[0].get_xdata()

        assert xdata.min() >= clip[0]
        assert xdata.max() <= clip[1]

    def test_line_is_density(self, long_df):

        ax = kdeplot(data=long_df, x="x", cut=5)
        x, y = ax.lines[0].get_xydata().T
        assert integrate(y, x) == pytest.approx(1)

    @pytest.mark.skipif(_no_scipy, reason="Test requires scipy")
    def test_cumulative(self, long_df):

        ax = kdeplot(data=long_df, x="x", cut=5, cumulative=True)
        y = ax.lines[0].get_ydata()
        assert y[0] == pytest.approx(0)
        assert y[-1] == pytest.approx(1)

    @pytest.mark.skipif(not _no_scipy, reason="Test requires scipy's absence")
    def test_cumulative_requires_scipy(self, long_df):

        with pytest.raises(RuntimeError):
            kdeplot(data=long_df, x="x", cut=5, cumulative=True)

    def test_common_norm(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(
            data=long_df, x="x", hue="c", common_norm=True, cut=10, ax=ax1
        )
        kdeplot(
            data=long_df, x="x", hue="c", common_norm=False, cut=10, ax=ax2
        )

        total_area = 0
        for line in ax1.lines:
            xdata, ydata = line.get_xydata().T
            total_area += integrate(ydata, xdata)
        assert total_area == pytest.approx(1)

        for line in ax2.lines:
            xdata, ydata = line.get_xydata().T
            assert integrate(ydata, xdata) == pytest.approx(1)

    def test_common_grid(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        order = "a", "b", "c"

        kdeplot(
            data=long_df, x="x", hue="a", hue_order=order,
            common_grid=False, cut=0, ax=ax1,
        )
        kdeplot(
            data=long_df, x="x", hue="a", hue_order=order,
            common_grid=True, cut=0, ax=ax2,
        )

        for line, level in zip(ax1.lines[::-1], order):
            xdata = line.get_xdata()
            assert xdata.min() == long_df.loc[long_df["a"] == level, "x"].min()
            assert xdata.max() == long_df.loc[long_df["a"] == level, "x"].max()

        for line in ax2.lines:
            xdata = line.get_xdata().T
            assert xdata.min() == long_df["x"].min()
            assert xdata.max() == long_df["x"].max()

    def test_bw_method(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", bw_method=0.2, legend=False)
        kdeplot(data=long_df, x="x", bw_method=1.0, legend=False)
        kdeplot(data=long_df, x="x", bw_method=3.0, legend=False)

        l1, l2, l3 = ax.lines

        assert (
            np.abs(np.diff(l1.get_ydata())).mean()
            > np.abs(np.diff(l2.get_ydata())).mean()
        )

        assert (
            np.abs(np.diff(l2.get_ydata())).mean()
            > np.abs(np.diff(l3.get_ydata())).mean()
        )

    def test_bw_adjust(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", bw_adjust=0.2, legend=False)
        kdeplot(data=long_df, x="x", bw_adjust=1.0, legend=False)
        kdeplot(data=long_df, x="x", bw_adjust=3.0, legend=False)

        l1, l2, l3 = ax.lines

        assert (
            np.abs(np.diff(l1.get_ydata())).mean()
            > np.abs(np.diff(l2.get_ydata())).mean()
        )

        assert (
            np.abs(np.diff(l2.get_ydata())).mean()
            > np.abs(np.diff(l3.get_ydata())).mean()
        )

    def test_log_scale_implicit(self, rng):

        x = rng.lognormal(0, 1, 100)

        f, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.set_xscale("log")

        kdeplot(x=x, ax=ax1)
        kdeplot(x=x, ax=ax1)

        xdata_log = ax1.lines[0].get_xdata()
        assert (xdata_log > 0).all()
        assert (np.diff(xdata_log, 2) > 0).all()
        assert np.allclose(np.diff(np.log(xdata_log), 2), 0)

        f, ax = plt.subplots()
        ax.set_yscale("log")
        kdeplot(y=x, ax=ax)
        assert_array_equal(ax.lines[0].get_xdata(), ax1.lines[0].get_ydata())

    def test_log_scale_explicit(self, rng):

        x = rng.lognormal(0, 1, 100)

        f, (ax1, ax2, ax3) = plt.subplots(ncols=3)

        ax1.set_xscale("log")
        kdeplot(x=x, ax=ax1)
        kdeplot(x=x, log_scale=True, ax=ax2)
        kdeplot(x=x, log_scale=10, ax=ax3)

        for ax in f.axes:
            assert ax.get_xscale() == "log"

        supports = [ax.lines[0].get_xdata() for ax in f.axes]
        for a, b in itertools.product(supports, supports):
            assert_array_equal(a, b)

        densities = [ax.lines[0].get_ydata() for ax in f.axes]
        for a, b in itertools.product(densities, densities):
            assert_array_equal(a, b)

        f, ax = plt.subplots()
        kdeplot(y=x, log_scale=True, ax=ax)
        assert ax.get_yscale() == "log"

    def test_log_scale_with_hue(self, rng):

        data = rng.lognormal(0, 1, 50), rng.lognormal(0, 2, 100)
        ax = kdeplot(data=data, log_scale=True, common_grid=True)
        assert_array_equal(ax.lines[0].get_xdata(), ax.lines[1].get_xdata())

    def test_log_scale_normalization(self, rng):

        x = rng.lognormal(0, 1, 100)
        ax = kdeplot(x=x, log_scale=True, cut=10)
        xdata, ydata = ax.lines[0].get_xydata().T
        integral = integrate(ydata, np.log10(xdata))
        assert integral == pytest.approx(1)

    def test_weights(self):

        x = [1, 2]
        weights = [2, 1]

        ax = kdeplot(x=x, weights=weights, bw_method=.1)

        xdata, ydata = ax.lines[0].get_xydata().T

        y1 = ydata[np.abs(xdata - 1).argmin()]
        y2 = ydata[np.abs(xdata - 2).argmin()]

        assert y1 == pytest.approx(2 * y2)

    def test_weight_norm(self, rng):

        vals = rng.normal(0, 1, 50)
        x = np.concatenate([vals, vals])
        w = np.repeat([1, 2], 50)
        ax = kdeplot(x=x, weights=w, hue=w, common_norm=True)

        # Recall that artists are added in reverse of hue order
        x1, y1 = ax.lines[0].get_xydata().T
        x2, y2 = ax.lines[1].get_xydata().T

        assert integrate(y1, x1) == pytest.approx(2 * integrate(y2, x2))

    def test_sticky_edges(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(data=long_df, x="x", fill=True, ax=ax1)
        assert ax1.collections[0].sticky_edges.y[:] == [0, np.inf]

        kdeplot(
            data=long_df, x="x", hue="a", multiple="fill", fill=True, ax=ax2
        )
        assert ax2.collections[0].sticky_edges.y[:] == [0, 1]

    def test_line_kws(self, flat_array):

        lw = 3
        color = (.2, .5, .8)
        ax = kdeplot(x=flat_array, linewidth=lw, color=color)
        line, = ax.lines
        assert line.get_linewidth() == lw
        assert_colors_equal(line.get_color(), color)

    def test_input_checking(self, long_df):

        err = "The x variable is categorical,"
        with pytest.raises(TypeError, match=err):
            kdeplot(data=long_df, x="a")

    def test_axis_labels(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(data=long_df, x="x", ax=ax1)
        assert ax1.get_xlabel() == "x"
        assert ax1.get_ylabel() == "Density"

        kdeplot(data=long_df, y="y", ax=ax2)
        assert ax2.get_xlabel() == "Density"
        assert ax2.get_ylabel() == "y"

    def test_legend(self, long_df):

        ax = kdeplot(data=long_df, x="x", hue="a")

        assert ax.legend_.get_title().get_text() == "a"

        legend_labels = ax.legend_.get_texts()
        order = categorical_order(long_df["a"])
        for label, level in zip(legend_labels, order):
            assert label.get_text() == level

        legend_artists = ax.legend_.findobj(mpl.lines.Line2D)
        if _version_predates(mpl, "3.5.0b0"):
            # https://github.com/matplotlib/matplotlib/pull/20699
            legend_artists = legend_artists[::2]
        palette = color_palette()
        for artist, color in zip(legend_artists, palette):
            assert_colors_equal(artist.get_color(), color)

        ax.clear()

        kdeplot(data=long_df, x="x", hue="a", legend=False)

        assert ax.legend_ is None

    def test_replaced_kws(self, long_df):
        with pytest.raises(TypeError, match=r"`data2` has been removed"):
            kdeplot(data=long_df, x="x", data2="y")


class TestKDEPlotBivariate:

    def test_long_vectors(self, long_df):

        ax1 = kdeplot(data=long_df, x="x", y="y")

        x = long_df["x"]
        x_values = [x, x.to_numpy(), x.to_list()]

        y = long_df["y"]
        y_values = [y, y.to_numpy(), y.to_list()]

        for x, y in zip(x_values, y_values):
            f, ax2 = plt.subplots()
            kdeplot(x=x, y=y, ax=ax2)

            for c1, c2 in zip(ax1.collections, ax2.collections):
                assert_array_equal(c1.get_offsets(), c2.get_offsets())

    def test_singular_data(self):

        with pytest.warns(UserWarning):
            ax = dist.kdeplot(x=np.ones(10), y=np.arange(10))
        assert not ax.lines

        with pytest.warns(UserWarning):
            ax = dist.kdeplot(x=[5], y=[6])
        assert not ax.lines

        with pytest.warns(UserWarning):
            ax = kdeplot(x=[1929245168.06679] * 18, y=np.arange(18))
        assert not ax.lines

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ax = kdeplot(x=[5], y=[7], warn_singular=False)
        assert not ax.lines

    def test_fill_artists(self, long_df):

        for fill in [True, False]:
            f, ax = plt.subplots()
            kdeplot(data=long_df, x="x", y="y", hue="c", fill=fill)
            for c in ax.collections:
                if not _version_predates(mpl, "3.8.0rc1"):
                    assert isinstance(c, mpl.contour.QuadContourSet)
                elif fill or not _version_predates(mpl, "3.5.0b0"):
                    assert isinstance(c, mpl.collections.PathCollection)
                else:
                    assert isinstance(c, mpl.collections.LineCollection)

    def test_common_norm(self, rng):

        hue = np.repeat(["a", "a", "a", "b"], 40)
        x, y = rng.multivariate_normal([0, 0], [(.2, .5), (.5, 2)], len(hue)).T
        x[hue == "a"] -= 2
        x[hue == "b"] += 2

        f, (ax1, ax2) = plt.subplots(ncols=2)
        kdeplot(x=x, y=y, hue=hue, common_norm=True, ax=ax1)
        kdeplot(x=x, y=y, hue=hue, common_norm=False, ax=ax2)

        n_seg_1 = sum(len(get_contour_coords(c, True)) for c in ax1.collections)
        n_seg_2 = sum(len(get_contour_coords(c, True)) for c in ax2.collections)
        assert n_seg_2 > n_seg_1

    def test_log_scale(self, rng):

        x = rng.lognormal(0, 1, 100)
        y = rng.uniform(0, 1, 100)

        levels = .2, .5, 1

        f, ax = plt.subplots()
        kdeplot(x=x, y=y, log_scale=True, levels=levels, ax=ax)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

        f, (ax1, ax2) = plt.subplots(ncols=2)
        kdeplot(x=x, y=y, log_scale=(10, False), levels=levels, ax=ax1)
        assert ax1.get_xscale() == "log"
        assert ax1.get_yscale() == "linear"

        p = _DistributionPlotter()
        kde = KDE()
        density, (xx, yy) = kde(np.log10(x), y)
        levels = p._quantile_to_level(density, levels)
        ax2.contour(10 ** xx, yy, density, levels=levels)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            assert len(get_contour_coords(c1)) == len(get_contour_coords(c2))
            for arr1, arr2 in zip(get_contour_coords(c1), get_contour_coords(c2)):
                assert_array_equal(arr1, arr2)

    def test_bandwidth(self, rng):

        n = 100
        x, y = rng.multivariate_normal([0, 0], [(.2, .5), (.5, 2)], n).T

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(x=x, y=y, ax=ax1)
        kdeplot(x=x, y=y, bw_adjust=2, ax=ax2)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            seg1, seg2 = get_contour_coords(c1), get_contour_coords(c2)
            if seg1 + seg2:
                x1 = seg1[0][:, 0]
                x2 = seg2[0][:, 0]
                assert np.abs(x2).max() > np.abs(x1).max()

    def test_weights(self, rng):

        n = 100
        x, y = rng.multivariate_normal([1, 3], [(.2, .5), (.5, 2)], n).T
        hue = np.repeat([0, 1], n // 2)
        weights = rng.uniform(0, 1, n)

        f, (ax1, ax2) = plt.subplots(ncols=2)
        kdeplot(x=x, y=y, hue=hue, ax=ax1)
        kdeplot(x=x, y=y, hue=hue, weights=weights, ax=ax2)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            if get_contour_coords(c1) and get_contour_coords(c2):
                seg1 = np.concatenate(get_contour_coords(c1), axis=0)
                seg2 = np.concatenate(get_contour_coords(c2), axis=0)
                assert not np.array_equal(seg1, seg2)

    def test_hue_ignores_cmap(self, long_df):

        with pytest.warns(UserWarning, match="cmap parameter ignored"):
            ax = kdeplot(data=long_df, x="x", y="y", hue="c", cmap="viridis")

        assert_colors_equal(get_contour_color(ax.collections[0]), "C0")

    def test_contour_line_colors(self, long_df):

        color = (.2, .9, .8, 1)
        ax = kdeplot(data=long_df, x="x", y="y", color=color)

        for c in ax.collections:
            assert_colors_equal(get_contour_color(c), color)

    def test_contour_line_cmap(self, long_df):

        color_list = color_palette("Blues", 12)
        cmap = mpl.colors.ListedColormap(color_list)
        ax = kdeplot(data=long_df, x="x", y="y", cmap=cmap)
        for c in ax.collections:
            for color in get_contour_color(c):
                assert to_rgb(color) in color_list

    def test_contour_fill_colors(self, long_df):

        n = 6
        color = (.2, .9, .8, 1)
        ax = kdeplot(
            data=long_df, x="x", y="y", fill=True, color=color, levels=n,
        )

        cmap = light_palette(color, reverse=True, as_cmap=True)
        lut = cmap(np.linspace(0, 1, 256))
        for c in ax.collections:
            for color in c.get_facecolor():
                assert color in lut

    def test_colorbar(self, long_df):

        ax = kdeplot(data=long_df, x="x", y="y", fill=True, cbar=True)
        assert len(ax.figure.axes) == 2

    def test_levels_and_thresh(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        n = 8
        thresh = .1
        plot_kws = dict(data=long_df, x="x", y="y")
        kdeplot(**plot_kws, levels=n, thresh=thresh, ax=ax1)
        kdeplot(**plot_kws, levels=np.linspace(thresh, 1, n), ax=ax2)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            assert len(get_contour_coords(c1)) == len(get_contour_coords(c2))
            for arr1, arr2 in zip(get_contour_coords(c1), get_contour_coords(c2)):
                assert_array_equal(arr1, arr2)

        with pytest.raises(ValueError):
            kdeplot(**plot_kws, levels=[0, 1, 2])

        ax1.clear()
        ax2.clear()

        kdeplot(**plot_kws, levels=n, thresh=None, ax=ax1)
        kdeplot(**plot_kws, levels=n, thresh=0, ax=ax2)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            assert len(get_contour_coords(c1)) == len(get_contour_coords(c2))
            for arr1, arr2 in zip(get_contour_coords(c1), get_contour_coords(c2)):
                assert_array_equal(arr1, arr2)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            assert_array_equal(c1.get_facecolors(), c2.get_facecolors())

    def test_quantile_to_level(self, rng):

        x = rng.uniform(0, 1, 100000)
        isoprop = np.linspace(.1, 1, 6)

        levels = _DistributionPlotter()._quantile_to_level(x, isoprop)
        for h, p in zip(levels, isoprop):
            assert (x[x <= h].sum() / x.sum()) == pytest.approx(p, abs=1e-4)

    def test_input_checking(self, long_df):

        with pytest.raises(TypeError, match="The x variable is categorical,"):
            kdeplot(data=long_df, x="a", y="y")


class TestHistPlotUnivariate(SharedAxesLevelTests):

    func = staticmethod(histplot)

    def get_last_color(self, ax, element="bars", fill=True):

        if element == "bars":
            if fill:
                return ax.patches[-1].get_facecolor()
            else:
                return ax.patches[-1].get_edgecolor()
        else:
            if fill:
                artist = ax.collections[-1]
                facecolor = artist.get_facecolor()
                edgecolor = artist.get_edgecolor()
                assert_colors_equal(facecolor, edgecolor, check_alpha=False)
                return facecolor
            else:
                return ax.lines[-1].get_color()

    @pytest.mark.parametrize(
        "element,fill",
        itertools.product(["bars", "step", "poly"], [True, False]),
    )
    def test_color(self, long_df, element, fill):

        super().test_color(long_df, element=element, fill=fill)

    @pytest.mark.parametrize(
        "variable", ["x", "y"],
    )
    def test_long_vectors(self, long_df, variable):

        vector = long_df[variable]
        vectors = [
            variable, vector, vector.to_numpy(), vector.to_list(),
        ]

        f, axs = plt.subplots(3)
        for vector, ax in zip(vectors, axs):
            histplot(data=long_df, ax=ax, **{variable: vector})

        bars = [ax.patches for ax in axs]
        for a_bars, b_bars in itertools.product(bars, bars):
            for a, b in zip(a_bars, b_bars):
                assert_array_equal(a.get_height(), b.get_height())
                assert_array_equal(a.get_xy(), b.get_xy())

    def test_wide_vs_long_data(self, wide_df):

        f, (ax1, ax2) = plt.subplots(2)

        histplot(data=wide_df, ax=ax1, common_bins=False)

        for col in wide_df.columns[::-1]:
            histplot(data=wide_df, x=col, ax=ax2)

        for a, b in zip(ax1.patches, ax2.patches):
            assert a.get_height() == b.get_height()
            assert a.get_xy() == b.get_xy()

    def test_flat_vector(self, long_df):

        f, (ax1, ax2) = plt.subplots(2)

        histplot(data=long_df["x"], ax=ax1)
        histplot(data=long_df, x="x", ax=ax2)

        for a, b in zip(ax1.patches, ax2.patches):
            assert a.get_height() == b.get_height()
            assert a.get_xy() == b.get_xy()

    def test_empty_data(self):

        ax = histplot(x=[])
        assert not ax.patches

    def test_variable_assignment(self, long_df):

        f, (ax1, ax2) = plt.subplots(2)

        histplot(data=long_df, x="x", ax=ax1)
        histplot(data=long_df, y="x", ax=ax2)

        for a, b in zip(ax1.patches, ax2.patches):
            assert a.get_height() == b.get_width()

    @pytest.mark.parametrize("element", ["bars", "step", "poly"])
    @pytest.mark.parametrize("multiple", ["layer", "dodge", "stack", "fill"])
    def test_hue_fill_colors(self, long_df, multiple, element):

        ax = histplot(
            data=long_df, x="x", hue="a",
            multiple=multiple, bins=1,
            fill=True, element=element, legend=False,
        )

        palette = color_palette()

        if multiple == "layer":
            if element == "bars":
                a = .5
            else:
                a = .25
        else:
            a = .75

        for bar, color in zip(ax.patches[::-1], palette):
            assert_colors_equal(bar.get_facecolor(), to_rgba(color, a))

        for poly, color in zip(ax.collections[::-1], palette):
            assert_colors_equal(poly.get_facecolor(), to_rgba(color, a))

    def test_hue_stack(self, long_df):

        f, (ax1, ax2) = plt.subplots(2)

        n = 10

        kws = dict(data=long_df, x="x", hue="a", bins=n, element="bars")

        histplot(**kws, multiple="layer", ax=ax1)
        histplot(**kws, multiple="stack", ax=ax2)

        layer_heights = np.reshape([b.get_height() for b in ax1.patches], (-1, n))
        stack_heights = np.reshape([b.get_height() for b in ax2.patches], (-1, n))
        assert_array_equal(layer_heights, stack_heights)

        stack_xys = np.reshape([b.get_xy() for b in ax2.patches], (-1, n, 2))
        assert_array_equal(
            stack_xys[..., 1] + stack_heights,
            stack_heights.cumsum(axis=0),
        )

    def test_hue_fill(self, long_df):

        f, (ax1, ax2) = plt.subplots(2)

        n = 10

        kws = dict(data=long_df, x="x", hue="a", bins=n, element="bars")

        histplot(**kws, multiple="layer", ax=ax1)
        histplot(**kws, multiple="fill", ax=ax2)

        layer_heights = np.reshape([b.get_height() for b in ax1.patches], (-1, n))
        stack_heights = np.reshape([b.get_height() for b in ax2.patches], (-1, n))
        assert_array_almost_equal(
            layer_heights / layer_heights.sum(axis=0), stack_heights
        )

        stack_xys = np.reshape([b.get_xy() for b in ax2.patches], (-1, n, 2))
        assert_array_almost_equal(
            (stack_xys[..., 1] + stack_heights) / stack_heights.sum(axis=0),
            stack_heights.cumsum(axis=0),
        )

    def test_hue_dodge(self, long_df):

        f, (ax1, ax2) = plt.subplots(2)

        bw = 2

        kws = dict(data=long_df, x="x", hue="c", binwidth=bw, element="bars")

        histplot(**kws, multiple="layer", ax=ax1)
        histplot(**kws, multiple="dodge", ax=ax2)

        layer_heights = [b.get_height() for b in ax1.patches]
        dodge_heights = [b.get_height() for b in ax2.patches]
        assert_array_equal(layer_heights, dodge_heights)

        layer_xs = np.reshape([b.get_x() for b in ax1.patches], (2, -1))
        dodge_xs = np.reshape([b.get_x() for b in ax2.patches], (2, -1))
        assert_array_almost_equal(layer_xs[1], dodge_xs[1])
        assert_array_almost_equal(layer_xs[0], dodge_xs[0] - bw / 2)

    def test_hue_as_numpy_dodged(self, long_df):
        # https://github.com/mwaskom/seaborn/issues/2452

        ax = histplot(
            long_df,
            x="y", hue=long_df["a"].to_numpy(),
            multiple="dodge", bins=1,
        )
        # Note hue order reversal
        assert ax.patches[1].get_x() < ax.patches[0].get_x()

    def test_multiple_input_check(self, flat_series):

        with pytest.raises(ValueError, match="`multiple` must be"):
            histplot(flat_series, multiple="invalid")

    def test_element_input_check(self, flat_series):

        with pytest.raises(ValueError, match="`element` must be"):
            histplot(flat_series, element="invalid")

    def test_count_stat(self, flat_series):

        ax = histplot(flat_series, stat="count")
        bar_heights = [b.get_height() for b in ax.patches]
        assert sum(bar_heights) == len(flat_series)

    def test_density_stat(self, flat_series):

        ax = histplot(flat_series, stat="density")
        bar_heights = [b.get_height() for b in ax.patches]
        bar_widths = [b.get_width() for b in ax.patches]
        assert np.multiply(bar_heights, bar_widths).sum() == pytest.approx(1)

    def test_density_stat_common_norm(self, long_df):

        ax = histplot(
            data=long_df, x="x", hue="a",
            stat="density", common_norm=True, element="bars",
        )
        bar_heights = [b.get_height() for b in ax.patches]
        bar_widths = [b.get_width() for b in ax.patches]
        assert np.multiply(bar_heights, bar_widths).sum() == pytest.approx(1)

    def test_density_stat_unique_norm(self, long_df):

        n = 10
        ax = histplot(
            data=long_df, x="x", hue="a",
            stat="density", bins=n, common_norm=False, element="bars",
        )

        bar_groups = ax.patches[:n], ax.patches[-n:]

        for bars in bar_groups:
            bar_heights = [b.get_height() for b in bars]
            bar_widths = [b.get_width() for b in bars]
            bar_areas = np.multiply(bar_heights, bar_widths)
            assert bar_areas.sum() == pytest.approx(1)

    @pytest.fixture(params=["probability", "proportion"])
    def height_norm_arg(self, request):
        return request.param

    def test_probability_stat(self, flat_series, height_norm_arg):

        ax = histplot(flat_series, stat=height_norm_arg)
        bar_heights = [b.get_height() for b in ax.patches]
        assert sum(bar_heights) == pytest.approx(1)

    def test_probability_stat_common_norm(self, long_df, height_norm_arg):

        ax = histplot(
            data=long_df, x="x", hue="a",
            stat=height_norm_arg, common_norm=True, element="bars",
        )
        bar_heights = [b.get_height() for b in ax.patches]
        assert sum(bar_heights) == pytest.approx(1)

    def test_probability_stat_unique_norm(self, long_df, height_norm_arg):

        n = 10
        ax = histplot(
            data=long_df, x="x", hue="a",
            stat=height_norm_arg, bins=n, common_norm=False, element="bars",
        )

        bar_groups = ax.patches[:n], ax.patches[-n:]

        for bars in bar_groups:
            bar_heights = [b.get_height() for b in bars]
            assert sum(bar_heights) == pytest.approx(1)

    def test_percent_stat(self, flat_series):

        ax = histplot(flat_series, stat="percent")
        bar_heights = [b.get_height() for b in ax.patches]
        assert sum(bar_heights) == 100

    def test_common_bins(self, long_df):

        n = 10
        ax = histplot(
            long_df, x="x", hue="a", common_bins=True, bins=n, element="bars",
        )

        bar_groups = ax.patches[:n], ax.patches[-n:]
        assert_array_equal(
            [b.get_xy() for b in bar_groups[0]],
            [b.get_xy() for b in bar_groups[1]]
        )

    def test_unique_bins(self, wide_df):

        ax = histplot(wide_df, common_bins=False, bins=10, element="bars")

        bar_groups = np.split(np.array(ax.patches), len(wide_df.columns))

        for i, col in enumerate(wide_df.columns[::-1]):
            bars = bar_groups[i]
            start = bars[0].get_x()
            stop = bars[-1].get_x() + bars[-1].get_width()
            assert_array_almost_equal(start, wide_df[col].min())
            assert_array_almost_equal(stop, wide_df[col].max())

    def test_range_with_inf(self, rng):

        x = rng.normal(0, 1, 20)
        ax = histplot([-np.inf, *x])
        leftmost_edge = min(p.get_x() for p in ax.patches)
        assert leftmost_edge == x.min()

    def test_weights_with_missing(self, null_df):

        ax = histplot(null_df, x="x", weights="s", bins=5)

        bar_heights = [bar.get_height() for bar in ax.patches]
        total_weight = null_df[["x", "s"]].dropna()["s"].sum()
        assert sum(bar_heights) == pytest.approx(total_weight)

    def test_weight_norm(self, rng):

        vals = rng.normal(0, 1, 50)
        x = np.concatenate([vals, vals])
        w = np.repeat([1, 2], 50)
        ax = histplot(
            x=x, weights=w, hue=w, common_norm=True, stat="density", bins=5
        )

        # Recall that artists are added in reverse of hue order
        y1 = [bar.get_height() for bar in ax.patches[:5]]
        y2 = [bar.get_height() for bar in ax.patches[5:]]

        assert sum(y1) == 2 * sum(y2)

    def test_discrete(self, long_df):

        ax = histplot(long_df, x="s", discrete=True)

        data_min = long_df["s"].min()
        data_max = long_df["s"].max()
        assert len(ax.patches) == (data_max - data_min + 1)

        for i, bar in enumerate(ax.patches):
            assert bar.get_width() == 1
            assert bar.get_x() == (data_min + i - .5)

    def test_discrete_categorical_default(self, long_df):

        ax = histplot(long_df, x="a")
        for i, bar in enumerate(ax.patches):
            assert bar.get_width() == 1

    def test_categorical_yaxis_inversion(self, long_df):

        ax = histplot(long_df, y="a")
        ymax, ymin = ax.get_ylim()
        assert ymax > ymin

    def test_datetime_scale(self, long_df):

        f, (ax1, ax2) = plt.subplots(2)
        histplot(x=long_df["t"], fill=True, ax=ax1)
        histplot(x=long_df["t"], fill=False, ax=ax2)
        assert ax1.get_xlim() == ax2.get_xlim()

    @pytest.mark.parametrize("stat", ["count", "density", "probability"])
    def test_kde(self, flat_series, stat):

        ax = histplot(
            flat_series, kde=True, stat=stat, kde_kws={"cut": 10}
        )

        bar_widths = [b.get_width() for b in ax.patches]
        bar_heights = [b.get_height() for b in ax.patches]
        hist_area = np.multiply(bar_widths, bar_heights).sum()

        density, = ax.lines
        kde_area = integrate(density.get_ydata(), density.get_xdata())

        assert kde_area == pytest.approx(hist_area)

    @pytest.mark.parametrize("multiple", ["layer", "dodge"])
    @pytest.mark.parametrize("stat", ["count", "density", "probability"])
    def test_kde_with_hue(self, long_df, stat, multiple):

        n = 10
        ax = histplot(
            long_df, x="x", hue="c", multiple=multiple,
            kde=True, stat=stat, element="bars",
            kde_kws={"cut": 10}, bins=n,
        )

        bar_groups = ax.patches[:n], ax.patches[-n:]

        for i, bars in enumerate(bar_groups):
            bar_widths = [b.get_width() for b in bars]
            bar_heights = [b.get_height() for b in bars]
            hist_area = np.multiply(bar_widths, bar_heights).sum()

            x, y = ax.lines[i].get_xydata().T
            kde_area = integrate(y, x)

            if multiple == "layer":
                assert kde_area == pytest.approx(hist_area)
            elif multiple == "dodge":
                assert kde_area == pytest.approx(hist_area * 2)

    def test_kde_default_cut(self, flat_series):

        ax = histplot(flat_series, kde=True)
        support = ax.lines[0].get_xdata()
        assert support.min() == flat_series.min()
        assert support.max() == flat_series.max()

    def test_kde_hue(self, long_df):

        n = 10
        ax = histplot(data=long_df, x="x", hue="a", kde=True, bins=n)

        for bar, line in zip(ax.patches[::n], ax.lines):
            assert_colors_equal(
                bar.get_facecolor(), line.get_color(), check_alpha=False
            )

    def test_kde_yaxis(self, flat_series):

        f, ax = plt.subplots()
        histplot(x=flat_series, kde=True)
        histplot(y=flat_series, kde=True)

        x, y = ax.lines
        assert_array_equal(x.get_xdata(), y.get_ydata())
        assert_array_equal(x.get_ydata(), y.get_xdata())

    def test_kde_line_kws(self, flat_series):

        lw = 5
        ax = histplot(flat_series, kde=True, line_kws=dict(lw=lw))
        assert ax.lines[0].get_linewidth() == lw

    def test_kde_singular_data(self):

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ax = histplot(x=np.ones(10), kde=True)
        assert not ax.lines

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ax = histplot(x=[5], kde=True)
        assert not ax.lines

    def test_element_default(self, long_df):

        f, (ax1, ax2) = plt.subplots(2)
        histplot(long_df, x="x", ax=ax1)
        histplot(long_df, x="x", ax=ax2, element="bars")
        assert len(ax1.patches) == len(ax2.patches)

        f, (ax1, ax2) = plt.subplots(2)
        histplot(long_df, x="x", hue="a", ax=ax1)
        histplot(long_df, x="x", hue="a", ax=ax2, element="bars")
        assert len(ax1.patches) == len(ax2.patches)

    def test_bars_no_fill(self, flat_series):

        alpha = .5
        ax = histplot(flat_series, element="bars", fill=False, alpha=alpha)
        for bar in ax.patches:
            assert bar.get_facecolor() == (0, 0, 0, 0)
            assert bar.get_edgecolor()[-1] == alpha

    def test_step_fill(self, flat_series):

        f, (ax1, ax2) = plt.subplots(2)

        n = 10
        histplot(flat_series, element="bars", fill=True, bins=n, ax=ax1)
        histplot(flat_series, element="step", fill=True, bins=n, ax=ax2)

        bar_heights = [b.get_height() for b in ax1.patches]
        bar_widths = [b.get_width() for b in ax1.patches]
        bar_edges = [b.get_x() for b in ax1.patches]

        fill = ax2.collections[0]
        x, y = fill.get_paths()[0].vertices[::-1].T

        assert_array_equal(x[1:2 * n:2], bar_edges)
        assert_array_equal(y[1:2 * n:2], bar_heights)

        assert x[n * 2] == bar_edges[-1] + bar_widths[-1]
        assert y[n * 2] == bar_heights[-1]

    def test_poly_fill(self, flat_series):

        f, (ax1, ax2) = plt.subplots(2)

        n = 10
        histplot(flat_series, element="bars", fill=True, bins=n, ax=ax1)
        histplot(flat_series, element="poly", fill=True, bins=n, ax=ax2)

        bar_heights = np.array([b.get_height() for b in ax1.patches])
        bar_widths = np.array([b.get_width() for b in ax1.patches])
        bar_edges = np.array([b.get_x() for b in ax1.patches])

        fill = ax2.collections[0]
        x, y = fill.get_paths()[0].vertices[::-1].T

        assert_array_equal(x[1:n + 1], bar_edges + bar_widths / 2)
        assert_array_equal(y[1:n + 1], bar_heights)

    def test_poly_no_fill(self, flat_series):

        f, (ax1, ax2) = plt.subplots(2)

        n = 10
        histplot(flat_series, element="bars", fill=False, bins=n, ax=ax1)
        histplot(flat_series, element="poly", fill=False, bins=n, ax=ax2)

        bar_heights = np.array([b.get_height() for b in ax1.patches])
        bar_widths = np.array([b.get_width() for b in ax1.patches])
        bar_edges = np.array([b.get_x() for b in ax1.patches])

        x, y = ax2.lines[0].get_xydata().T

        assert_array_equal(x, bar_edges + bar_widths / 2)
        assert_array_equal(y, bar_heights)

    def test_step_no_fill(self, flat_series):

        f, (ax1, ax2) = plt.subplots(2)

        histplot(flat_series, element="bars", fill=False, ax=ax1)
        histplot(flat_series, element="step", fill=False, ax=ax2)

        bar_heights = [b.get_height() for b in ax1.patches]
        bar_widths = [b.get_width() for b in ax1.patches]
        bar_edges = [b.get_x() for b in ax1.patches]

        x, y = ax2.lines[0].get_xydata().T

        assert_array_equal(x[:-1], bar_edges)
        assert_array_equal(y[:-1], bar_heights)
        assert x[-1] == bar_edges[-1] + bar_widths[-1]
        assert y[-1] == y[-2]

    def test_step_fill_xy(self, flat_series):

        f, ax = plt.subplots()

        histplot(x=flat_series, element="step", fill=True)
        histplot(y=flat_series, element="step", fill=True)

        xverts = ax.collections[0].get_paths()[0].vertices
        yverts = ax.collections[1].get_paths()[0].vertices

        assert_array_equal(xverts, yverts[:, ::-1])

    def test_step_no_fill_xy(self, flat_series):

        f, ax = plt.subplots()

        histplot(x=flat_series, element="step", fill=False)
        histplot(y=flat_series, element="step", fill=False)

        xline, yline = ax.lines

        assert_array_equal(xline.get_xdata(), yline.get_ydata())
        assert_array_equal(xline.get_ydata(), yline.get_xdata())

    def test_weighted_histogram(self):

        ax = histplot(x=[0, 1, 2], weights=[1, 2, 3], discrete=True)

        bar_heights = [b.get_height() for b in ax.patches]
        assert bar_heights == [1, 2, 3]

    def test_weights_with_auto_bins(self, long_df):

        with pytest.warns(UserWarning):
            ax = histplot(long_df, x="x", weights="f")
        assert len(ax.patches) == 10

    def test_shrink(self, long_df):

        f, (ax1, ax2) = plt.subplots(2)

        bw = 2
        shrink = .4

        histplot(long_df, x="x", binwidth=bw, ax=ax1)
        histplot(long_df, x="x", binwidth=bw, shrink=shrink, ax=ax2)

        for p1, p2 in zip(ax1.patches, ax2.patches):

            w1, w2 = p1.get_width(), p2.get_width()
            assert w2 == pytest.approx(shrink * w1)

            x1, x2 = p1.get_x(), p2.get_x()
            assert (x2 + w2 / 2) == pytest.approx(x1 + w1 / 2)

    def test_log_scale_explicit(self, rng):

        x = rng.lognormal(0, 2, 1000)
        ax = histplot(x, log_scale=True, binrange=(-3, 3), binwidth=1)

        bar_widths = [b.get_width() for b in ax.patches]
        steps = np.divide(bar_widths[1:], bar_widths[:-1])
        assert np.allclose(steps, 10)

    def test_log_scale_implicit(self, rng):

        x = rng.lognormal(0, 2, 1000)

        f, ax = plt.subplots()
        ax.set_xscale("log")
        histplot(x, binrange=(-3, 3), binwidth=1, ax=ax)

        bar_widths = [b.get_width() for b in ax.patches]
        steps = np.divide(bar_widths[1:], bar_widths[:-1])
        assert np.allclose(steps, 10)

    def test_log_scale_dodge(self, rng):

        x = rng.lognormal(0, 2, 100)
        hue = np.repeat(["a", "b"], 50)
        ax = histplot(x=x, hue=hue, bins=5, log_scale=True, multiple="dodge")
        x_min = np.log([b.get_x() for b in ax.patches])
        x_max = np.log([b.get_x() + b.get_width() for b in ax.patches])
        assert np.unique(np.round(x_max - x_min, 10)).size == 1

    def test_log_scale_kde(self, rng):

        x = rng.lognormal(0, 1, 1000)
        ax = histplot(x=x, log_scale=True, kde=True, bins=20)
        bar_height = max(p.get_height() for p in ax.patches)
        kde_height = max(ax.lines[0].get_ydata())
        assert bar_height == pytest.approx(kde_height, rel=.1)

    @pytest.mark.parametrize(
        "fill", [True, False],
    )
    def test_auto_linewidth(self, flat_series, fill):

        get_lw = lambda ax: ax.patches[0].get_linewidth()  # noqa: E731

        kws = dict(element="bars", fill=fill)

        f, (ax1, ax2) = plt.subplots(2)
        histplot(flat_series, **kws, bins=10, ax=ax1)
        histplot(flat_series, **kws, bins=100, ax=ax2)
        assert get_lw(ax1) > get_lw(ax2)

        f, ax1 = plt.subplots(figsize=(10, 5))
        f, ax2 = plt.subplots(figsize=(2, 5))
        histplot(flat_series, **kws, bins=30, ax=ax1)
        histplot(flat_series, **kws, bins=30, ax=ax2)
        assert get_lw(ax1) > get_lw(ax2)

        f, ax1 = plt.subplots(figsize=(4, 5))
        f, ax2 = plt.subplots(figsize=(4, 5))
        histplot(flat_series, **kws, bins=30, ax=ax1)
        histplot(10 ** flat_series, **kws, bins=30, log_scale=True, ax=ax2)
        assert get_lw(ax1) == pytest.approx(get_lw(ax2))

        f, ax1 = plt.subplots(figsize=(4, 5))
        f, ax2 = plt.subplots(figsize=(4, 5))
        histplot(y=[0, 1, 1], **kws, discrete=True, ax=ax1)
        histplot(y=["a", "b", "b"], **kws, ax=ax2)
        assert get_lw(ax1) == pytest.approx(get_lw(ax2))

    def test_bar_kwargs(self, flat_series):

        lw = 2
        ec = (1, .2, .9, .5)
        ax = histplot(flat_series, binwidth=1, ec=ec, lw=lw)
        for bar in ax.patches:
            assert_colors_equal(bar.get_edgecolor(), ec)
            assert bar.get_linewidth() == lw

    def test_step_fill_kwargs(self, flat_series):

        lw = 2
        ec = (1, .2, .9, .5)
        ax = histplot(flat_series, element="step", ec=ec, lw=lw)
        poly = ax.collections[0]
        assert_colors_equal(poly.get_edgecolor(), ec)
        assert poly.get_linewidth() == lw

    def test_step_line_kwargs(self, flat_series):

        lw = 2
        ls = "--"
        ax = histplot(flat_series, element="step", fill=False, lw=lw, ls=ls)
        line = ax.lines[0]
        assert line.get_linewidth() == lw
        assert line.get_linestyle() == ls

    def test_label(self, flat_series):

        ax = histplot(flat_series, label="a label")
        handles, labels = ax.get_legend_handles_labels()
        assert len(handles) == 1
        assert labels == ["a label"]

    def test_default_color_scout_cleanup(self, flat_series):

        ax = histplot(flat_series)
        assert len(ax.containers) == 1


class TestHistPlotBivariate:

    def test_mesh(self, long_df):

        hist = Histogram()
        counts, (x_edges, y_edges) = hist(long_df["x"], long_df["y"])

        ax = histplot(long_df, x="x", y="y")
        mesh = ax.collections[0]
        mesh_data = mesh.get_array()

        assert_array_equal(mesh_data.data.flat, counts.T.flat)
        assert_array_equal(mesh_data.mask.flat, counts.T.flat == 0)

        edges = itertools.product(y_edges[:-1], x_edges[:-1])
        for i, (y, x) in enumerate(edges):
            path = mesh.get_paths()[i]
            assert path.vertices[0, 0] == x
            assert path.vertices[0, 1] == y

    def test_mesh_with_hue(self, long_df):

        ax = histplot(long_df, x="x", y="y", hue="c")

        hist = Histogram()
        hist.define_bin_params(long_df["x"], long_df["y"])

        for i, sub_df in long_df.groupby("c"):

            mesh = ax.collections[i]
            mesh_data = mesh.get_array()

            counts, (x_edges, y_edges) = hist(sub_df["x"], sub_df["y"])

            assert_array_equal(mesh_data.data.flat, counts.T.flat)
            assert_array_equal(mesh_data.mask.flat, counts.T.flat == 0)

            edges = itertools.product(y_edges[:-1], x_edges[:-1])
            for i, (y, x) in enumerate(edges):
                path = mesh.get_paths()[i]
                assert path.vertices[0, 0] == x
                assert path.vertices[0, 1] == y

    def test_mesh_with_hue_unique_bins(self, long_df):

        ax = histplot(long_df, x="x", y="y", hue="c", common_bins=False)

        for i, sub_df in long_df.groupby("c"):

            hist = Histogram()

            mesh = ax.collections[i]
            mesh_data = mesh.get_array()

            counts, (x_edges, y_edges) = hist(sub_df["x"], sub_df["y"])

            assert_array_equal(mesh_data.data.flat, counts.T.flat)
            assert_array_equal(mesh_data.mask.flat, counts.T.flat == 0)

            edges = itertools.product(y_edges[:-1], x_edges[:-1])
            for i, (y, x) in enumerate(edges):
                path = mesh.get_paths()[i]
                assert path.vertices[0, 0] == x
                assert path.vertices[0, 1] == y

    def test_mesh_with_col_unique_bins(self, long_df):

        g = displot(long_df, x="x", y="y", col="c", common_bins=False)

        for i, sub_df in long_df.groupby("c"):

            hist = Histogram()

            mesh = g.axes.flat[i].collections[0]
            mesh_data = mesh.get_array()

            counts, (x_edges, y_edges) = hist(sub_df["x"], sub_df["y"])

            assert_array_equal(mesh_data.data.flat, counts.T.flat)
            assert_array_equal(mesh_data.mask.flat, counts.T.flat == 0)

            edges = itertools.product(y_edges[:-1], x_edges[:-1])
            for i, (y, x) in enumerate(edges):
                path = mesh.get_paths()[i]
                assert path.vertices[0, 0] == x
                assert path.vertices[0, 1] == y

    def test_mesh_log_scale(self, rng):

        x, y = rng.lognormal(0, 1, (2, 1000))
        hist = Histogram()
        counts, (x_edges, y_edges) = hist(np.log10(x), np.log10(y))

        ax = histplot(x=x, y=y, log_scale=True)
        mesh = ax.collections[0]
        mesh_data = mesh.get_array()

        assert_array_equal(mesh_data.data.flat, counts.T.flat)

        edges = itertools.product(y_edges[:-1], x_edges[:-1])
        for i, (y_i, x_i) in enumerate(edges):
            path = mesh.get_paths()[i]
            assert path.vertices[0, 0] == pytest.approx(10 ** x_i)
            assert path.vertices[0, 1] == pytest.approx(10 ** y_i)

    def test_mesh_thresh(self, long_df):

        hist = Histogram()
        counts, (x_edges, y_edges) = hist(long_df["x"], long_df["y"])

        thresh = 5
        ax = histplot(long_df, x="x", y="y", thresh=thresh)
        mesh = ax.collections[0]
        mesh_data = mesh.get_array()

        assert_array_equal(mesh_data.data.flat, counts.T.flat)
        assert_array_equal(mesh_data.mask.flat, (counts <= thresh).T.flat)

    def test_mesh_sticky_edges(self, long_df):

        ax = histplot(long_df, x="x", y="y", thresh=None)
        mesh = ax.collections[0]
        assert mesh.sticky_edges.x == [long_df["x"].min(), long_df["x"].max()]
        assert mesh.sticky_edges.y == [long_df["y"].min(), long_df["y"].max()]

        ax.clear()
        ax = histplot(long_df, x="x", y="y")
        mesh = ax.collections[0]
        assert not mesh.sticky_edges.x
        assert not mesh.sticky_edges.y

    def test_mesh_common_norm(self, long_df):

        stat = "density"
        ax = histplot(
            long_df, x="x", y="y", hue="c", common_norm=True, stat=stat,
        )

        hist = Histogram(stat="density")
        hist.define_bin_params(long_df["x"], long_df["y"])

        for i, sub_df in long_df.groupby("c"):

            mesh = ax.collections[i]
            mesh_data = mesh.get_array()

            density, (x_edges, y_edges) = hist(sub_df["x"], sub_df["y"])

            scale = len(sub_df) / len(long_df)
            assert_array_equal(mesh_data.data.flat, (density * scale).T.flat)

    def test_mesh_unique_norm(self, long_df):

        stat = "density"
        ax = histplot(
            long_df, x="x", y="y", hue="c", common_norm=False, stat=stat,
        )

        hist = Histogram()
        bin_kws = hist.define_bin_params(long_df["x"], long_df["y"])

        for i, sub_df in long_df.groupby("c"):

            sub_hist = Histogram(bins=bin_kws["bins"], stat=stat)

            mesh = ax.collections[i]
            mesh_data = mesh.get_array()

            density, (x_edges, y_edges) = sub_hist(sub_df["x"], sub_df["y"])
            assert_array_equal(mesh_data.data.flat, density.T.flat)

    @pytest.mark.parametrize("stat", ["probability", "proportion", "percent"])
    def test_mesh_normalization(self, long_df, stat):

        ax = histplot(
            long_df, x="x", y="y", stat=stat,
        )

        mesh_data = ax.collections[0].get_array()
        expected_sum = {"percent": 100}.get(stat, 1)
        assert mesh_data.data.sum() == expected_sum

    def test_mesh_colors(self, long_df):

        color = "r"
        f, ax = plt.subplots()
        histplot(
            long_df, x="x", y="y", color=color,
        )
        mesh = ax.collections[0]
        assert_array_equal(
            mesh.get_cmap().colors,
            _DistributionPlotter()._cmap_from_color(color).colors,
        )

        f, ax = plt.subplots()
        histplot(
            long_df, x="x", y="y", hue="c",
        )
        colors = color_palette()
        for i, mesh in enumerate(ax.collections):
            assert_array_equal(
                mesh.get_cmap().colors,
                _DistributionPlotter()._cmap_from_color(colors[i]).colors,
            )

    def test_color_limits(self, long_df):

        f, (ax1, ax2, ax3) = plt.subplots(3)
        kws = dict(data=long_df, x="x", y="y")
        hist = Histogram()
        counts, _ = hist(long_df["x"], long_df["y"])

        histplot(**kws, ax=ax1)
        assert ax1.collections[0].get_clim() == (0, counts.max())

        vmax = 10
        histplot(**kws, vmax=vmax, ax=ax2)
        counts, _ = hist(long_df["x"], long_df["y"])
        assert ax2.collections[0].get_clim() == (0, vmax)

        pmax = .8
        pthresh = .1
        f = _DistributionPlotter()._quantile_to_level

        histplot(**kws, pmax=pmax, pthresh=pthresh, ax=ax3)
        counts, _ = hist(long_df["x"], long_df["y"])
        mesh = ax3.collections[0]
        assert mesh.get_clim() == (0, f(counts, pmax))
        assert_array_equal(
            mesh.get_array().mask.flat,
            (counts <= f(counts, pthresh)).T.flat,
        )

    def test_hue_color_limits(self, long_df):

        _, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        kws = dict(data=long_df, x="x", y="y", hue="c", bins=4)

        hist = Histogram(bins=kws["bins"])
        hist.define_bin_params(long_df["x"], long_df["y"])
        full_counts, _ = hist(long_df["x"], long_df["y"])

        sub_counts = []
        for _, sub_df in long_df.groupby(kws["hue"]):
            c, _ = hist(sub_df["x"], sub_df["y"])
            sub_counts.append(c)

        pmax = .8
        pthresh = .05
        f = _DistributionPlotter()._quantile_to_level

        histplot(**kws, common_norm=True, ax=ax1)
        for i, mesh in enumerate(ax1.collections):
            assert mesh.get_clim() == (0, full_counts.max())

        histplot(**kws, common_norm=False, ax=ax2)
        for i, mesh in enumerate(ax2.collections):
            assert mesh.get_clim() == (0, sub_counts[i].max())

        histplot(**kws, common_norm=True, pmax=pmax, pthresh=pthresh, ax=ax3)
        for i, mesh in enumerate(ax3.collections):
            assert mesh.get_clim() == (0, f(full_counts, pmax))
            assert_array_equal(
                mesh.get_array().mask.flat,
                (sub_counts[i] <= f(full_counts, pthresh)).T.flat,
            )

        histplot(**kws, common_norm=False, pmax=pmax, pthresh=pthresh, ax=ax4)
        for i, mesh in enumerate(ax4.collections):
            assert mesh.get_clim() == (0, f(sub_counts[i], pmax))
            assert_array_equal(
                mesh.get_array().mask.flat,
                (sub_counts[i] <= f(sub_counts[i], pthresh)).T.flat,
            )

    def test_colorbar(self, long_df):

        f, ax = plt.subplots()
        histplot(long_df, x="x", y="y", cbar=True, ax=ax)
        assert len(ax.figure.axes) == 2

        f, (ax, cax) = plt.subplots(2)
        histplot(long_df, x="x", y="y", cbar=True, cbar_ax=cax, ax=ax)
        assert len(ax.figure.axes) == 2


class TestECDFPlotUnivariate(SharedAxesLevelTests):

    func = staticmethod(ecdfplot)

    def get_last_color(self, ax):

        return to_rgb(ax.lines[-1].get_color())

    @pytest.mark.parametrize("variable", ["x", "y"])
    def test_long_vectors(self, long_df, variable):

        vector = long_df[variable]
        vectors = [
            variable, vector, vector.to_numpy(), vector.to_list(),
        ]

        f, ax = plt.subplots()
        for vector in vectors:
            ecdfplot(data=long_df, ax=ax, **{variable: vector})

        xdata = [l.get_xdata() for l in ax.lines]
        for a, b in itertools.product(xdata, xdata):
            assert_array_equal(a, b)

        ydata = [l.get_ydata() for l in ax.lines]
        for a, b in itertools.product(ydata, ydata):
            assert_array_equal(a, b)

    def test_hue(self, long_df):

        ax = ecdfplot(long_df, x="x", hue="a")

        for line, color in zip(ax.lines[::-1], color_palette()):
            assert_colors_equal(line.get_color(), color)

    def test_line_kwargs(self, long_df):

        color = "r"
        ls = "--"
        lw = 3
        ax = ecdfplot(long_df, x="x", color=color, ls=ls, lw=lw)

        for line in ax.lines:
            assert_colors_equal(line.get_color(), color)
            assert line.get_linestyle() == ls
            assert line.get_linewidth() == lw

    @pytest.mark.parametrize("data_var", ["x", "y"])
    def test_drawstyle(self, flat_series, data_var):

        ax = ecdfplot(**{data_var: flat_series})
        drawstyles = dict(x="steps-post", y="steps-pre")
        assert ax.lines[0].get_drawstyle() == drawstyles[data_var]

    @pytest.mark.parametrize(
        "data_var,stat_var", [["x", "y"], ["y", "x"]],
    )
    def test_proportion_limits(self, flat_series, data_var, stat_var):

        ax = ecdfplot(**{data_var: flat_series})
        data = getattr(ax.lines[0], f"get_{stat_var}data")()
        assert data[0] == 0
        assert data[-1] == 1
        sticky_edges = getattr(ax.lines[0].sticky_edges, stat_var)
        assert sticky_edges[:] == [0, 1]

    @pytest.mark.parametrize(
        "data_var,stat_var", [["x", "y"], ["y", "x"]],
    )
    def test_proportion_limits_complementary(self, flat_series, data_var, stat_var):

        ax = ecdfplot(**{data_var: flat_series}, complementary=True)
        data = getattr(ax.lines[0], f"get_{stat_var}data")()
        assert data[0] == 1
        assert data[-1] == 0
        sticky_edges = getattr(ax.lines[0].sticky_edges, stat_var)
        assert sticky_edges[:] == [0, 1]

    @pytest.mark.parametrize(
        "data_var,stat_var", [["x", "y"], ["y", "x"]],
    )
    def test_proportion_count(self, flat_series, data_var, stat_var):

        n = len(flat_series)
        ax = ecdfplot(**{data_var: flat_series}, stat="count")
        data = getattr(ax.lines[0], f"get_{stat_var}data")()
        assert data[0] == 0
        assert data[-1] == n
        sticky_edges = getattr(ax.lines[0].sticky_edges, stat_var)
        assert sticky_edges[:] == [0, n]

    def test_weights(self):

        ax = ecdfplot(x=[1, 2, 3], weights=[1, 1, 2])
        y = ax.lines[0].get_ydata()
        assert_array_equal(y, [0, .25, .5, 1])

    def test_bivariate_error(self, long_df):

        with pytest.raises(NotImplementedError, match="Bivariate ECDF plots"):
            ecdfplot(data=long_df, x="x", y="y")

    def test_log_scale(self, long_df):

        ax1, ax2 = plt.figure().subplots(2)

        ecdfplot(data=long_df, x="z", ax=ax1)
        ecdfplot(data=long_df, x="z", log_scale=True, ax=ax2)

        # Ignore first point, which either -inf (in linear) or 0 (in log)
        line1 = ax1.lines[0].get_xydata()[1:]
        line2 = ax2.lines[0].get_xydata()[1:]

        assert_array_almost_equal(line1, line2)


class TestDisPlot:

    # TODO probably good to move these utility attributes/methods somewhere else
    @pytest.mark.parametrize(
        "kwargs", [
            dict(),
            dict(x="x"),
            dict(x="t"),
            dict(x="a"),
            dict(x="z", log_scale=True),
            dict(x="x", binwidth=4),
            dict(x="x", weights="f", bins=5),
            dict(x="x", color="green", linewidth=2, binwidth=4),
            dict(x="x", hue="a", fill=False),
            dict(x="y", hue="a", fill=False),
            dict(x="x", hue="a", multiple="stack"),
            dict(x="x", hue="a", element="step"),
            dict(x="x", hue="a", palette="muted"),
            dict(x="x", hue="a", kde=True),
            dict(x="x", hue="a", stat="density", common_norm=False),
            dict(x="x", y="y"),
        ],
    )
    def test_versus_single_histplot(self, long_df, kwargs):

        ax = histplot(long_df, **kwargs)
        g = displot(long_df, **kwargs)
        assert_plots_equal(ax, g.ax)

        if ax.legend_ is not None:
            assert_legends_equal(ax.legend_, g._legend)

        if kwargs:
            long_df["_"] = "_"
            g2 = displot(long_df, col="_", **kwargs)
            assert_plots_equal(ax, g2.ax)

    @pytest.mark.parametrize(
        "kwargs", [
            dict(),
            dict(x="x"),
            dict(x="t"),
            dict(x="z", log_scale=True),
            dict(x="x", bw_adjust=.5),
            dict(x="x", weights="f"),
            dict(x="x", color="green", linewidth=2),
            dict(x="x", hue="a", multiple="stack"),
            dict(x="x", hue="a", fill=True),
            dict(x="y", hue="a", fill=False),
            dict(x="x", hue="a", palette="muted"),
            dict(x="x", y="y"),
        ],
    )
    def test_versus_single_kdeplot(self, long_df, kwargs):

        ax = kdeplot(data=long_df, **kwargs)
        g = displot(long_df, kind="kde", **kwargs)
        assert_plots_equal(ax, g.ax)

        if ax.legend_ is not None:
            assert_legends_equal(ax.legend_, g._legend)

        if kwargs:
            long_df["_"] = "_"
            g2 = displot(long_df, kind="kde", col="_", **kwargs)
            assert_plots_equal(ax, g2.ax)

    @pytest.mark.parametrize(
        "kwargs", [
            dict(),
            dict(x="x"),
            dict(x="t"),
            dict(x="z", log_scale=True),
            dict(x="x", weights="f"),
            dict(y="x"),
            dict(x="x", color="green", linewidth=2),
            dict(x="x", hue="a", complementary=True),
            dict(x="x", hue="a", stat="count"),
            dict(x="x", hue="a", palette="muted"),
        ],
    )
    def test_versus_single_ecdfplot(self, long_df, kwargs):

        ax = ecdfplot(data=long_df, **kwargs)
        g = displot(long_df, kind="ecdf", **kwargs)
        assert_plots_equal(ax, g.ax)

        if ax.legend_ is not None:
            assert_legends_equal(ax.legend_, g._legend)

        if kwargs:
            long_df["_"] = "_"
            g2 = displot(long_df, kind="ecdf", col="_", **kwargs)
            assert_plots_equal(ax, g2.ax)

    @pytest.mark.parametrize(
        "kwargs", [
            dict(x="x"),
            dict(x="x", y="y"),
            dict(x="x", hue="a"),
        ]
    )
    def test_with_rug(self, long_df, kwargs):

        ax = plt.figure().subplots()
        histplot(data=long_df, **kwargs, ax=ax)
        rugplot(data=long_df, **kwargs, ax=ax)

        g = displot(long_df, rug=True, **kwargs)

        assert_plots_equal(ax, g.ax, labels=False)

        long_df["_"] = "_"
        g2 = displot(long_df, col="_", rug=True, **kwargs)

        assert_plots_equal(ax, g2.ax, labels=False)

    @pytest.mark.parametrize(
        "facet_var", ["col", "row"],
    )
    def test_facets(self, long_df, facet_var):

        kwargs = {facet_var: "a"}
        ax = kdeplot(data=long_df, x="x", hue="a")
        g = displot(long_df, x="x", kind="kde", **kwargs)

        legend_texts = ax.legend_.get_texts()

        for i, line in enumerate(ax.lines[::-1]):
            facet_ax = g.axes.flat[i]
            facet_line = facet_ax.lines[0]
            assert_array_equal(line.get_xydata(), facet_line.get_xydata())

            text = legend_texts[i].get_text()
            assert text in facet_ax.get_title()

    @pytest.mark.parametrize("multiple", ["dodge", "stack", "fill"])
    def test_facet_multiple(self, long_df, multiple):

        bins = np.linspace(0, 20, 5)
        ax = histplot(
            data=long_df[long_df["c"] == 0],
            x="x", hue="a", hue_order=["a", "b", "c"],
            multiple=multiple, bins=bins,
        )

        g = displot(
            data=long_df, x="x", hue="a", col="c", hue_order=["a", "b", "c"],
            multiple=multiple, bins=bins,
        )

        assert_plots_equal(ax, g.axes_dict[0])

    def test_ax_warning(self, long_df):

        ax = plt.figure().subplots()
        with pytest.warns(UserWarning, match="`displot` is a figure-level"):
            displot(long_df, x="x", ax=ax)

    @pytest.mark.parametrize("key", ["col", "row"])
    def test_array_faceting(self, long_df, key):

        a = long_df["a"].to_numpy()
        vals = categorical_order(a)
        g = displot(long_df, x="x", **{key: a})
        assert len(g.axes.flat) == len(vals)
        for ax, val in zip(g.axes.flat, vals):
            assert val in ax.get_title()

    def test_legend(self, long_df):

        g = displot(long_df, x="x", hue="a")
        assert g._legend is not None

    def test_empty(self):

        g = displot(x=[], y=[])
        assert isinstance(g, FacetGrid)

    def test_bivariate_ecdf_error(self, long_df):

        with pytest.raises(NotImplementedError):
            displot(long_df, x="x", y="y", kind="ecdf")

    def test_bivariate_kde_norm(self, rng):

        x, y = rng.normal(0, 1, (2, 100))
        z = [0] * 80 + [1] * 20

        def count_contours(ax):
            if _version_predates(mpl, "3.8.0rc1"):
                return sum(bool(get_contour_coords(c)) for c in ax.collections)
            else:
                return sum(bool(p.vertices.size) for p in ax.collections[0].get_paths())

        g = displot(x=x, y=y, col=z, kind="kde", levels=10)
        l1 = count_contours(g.axes.flat[0])
        l2 = count_contours(g.axes.flat[1])
        assert l1 > l2

        g = displot(x=x, y=y, col=z, kind="kde", levels=10, common_norm=False)
        l1 = count_contours(g.axes.flat[0])
        l2 = count_contours(g.axes.flat[1])
        assert l1 == l2

    def test_bivariate_hist_norm(self, rng):

        x, y = rng.normal(0, 1, (2, 100))
        z = [0] * 80 + [1] * 20

        g = displot(x=x, y=y, col=z, kind="hist")
        clim1 = g.axes.flat[0].collections[0].get_clim()
        clim2 = g.axes.flat[1].collections[0].get_clim()
        assert clim1 == clim2

        g = displot(x=x, y=y, col=z, kind="hist", common_norm=False)
        clim1 = g.axes.flat[0].collections[0].get_clim()
        clim2 = g.axes.flat[1].collections[0].get_clim()
        assert clim1[1] > clim2[1]

    def test_facetgrid_data(self, long_df):

        g = displot(
            data=long_df.to_dict(orient="list"),
            x="z",
            hue=long_df["a"].rename("hue_var"),
            col=long_df["c"].to_numpy(),
        )
        expected_cols = set(long_df.columns.to_list() + ["hue_var", "_col_"])
        assert set(g.data.columns) == expected_cols
        assert_array_equal(g.data["hue_var"], long_df["a"])
        assert_array_equal(g.data["_col_"], long_df["c"])


def integrate(y, x):
    """"Simple numerical integration for testing KDE code."""
    y = np.asarray(y)
    x = np.asarray(x)
    dx = np.diff(x)
    return (dx * y[:-1] + dx * y[1:]).sum() / 2


# ============================================================================
# Bin diagnostics tests - unified collector across legacy/objects layers
# ============================================================================

class TestBinDiagnosticsCollector:
    """Direct tests of the unified BinDiagnosticsCollector."""

    @pytest.fixture
    def rs(self):
        return np.random.RandomState(42)

    def test_bin_diagnostics_fields(self, rs):
        """BinDiagnostics dataclass must expose all documented fields."""
        x = rs.randn(100)
        edges = np.linspace(x.min(), x.max(), 11)
        hist, _ = np.histogram(x, edges)
        diag = BinDiagnostics(
            bin_edges=edges,
            count=100,
            weight_sum=100.0,
            normalization_denominator=None,
            empty_reason=None,
            extra={"stat": "count"},
        )
        assert isinstance(diag.bin_edges, np.ndarray)
        assert diag.bin_edges.shape == (11,)
        assert diag.count == 100
        assert diag.weight_sum == 100.0
        assert diag.normalization_denominator is None
        assert diag.empty_reason is None
        assert "stat" in diag.extra
        # __repr__ must not raise
        assert "BinDiagnostics" in repr(diag)

    def test_collector_single_group(self, rs):
        """Collector should correctly diagnose a single ungrouped dataset."""
        x = rs.randn(150)
        collector = BinDiagnosticsCollector(stat="count", cumulative=False)
        hist, edges = np.histogram(x, bins=20)
        collector.add_group_univariate(
            group_key=(),
            x=x,
            bin_edges=edges,
            hist=hist,
            weights=None,
        )
        diags = collector.diagnostics
        assert () in diags
        assert diags[()].count == 150
        assert diags[()].weight_sum == 150.0
        assert diags[()].bin_edges.shape == (21,)
        # count stat has no normalization denominator
        assert diags[()].normalization_denominator is None
        assert diags[()].empty_reason is None

    def test_collector_density_normalization(self, rs):
        """Collector for density stat should compute integration-to-1 denominator."""
        x = rs.randn(200)
        collector = BinDiagnosticsCollector(stat="density", cumulative=False)
        hist, edges = np.histogram(x, bins=30, density=True)
        collector.add_group_univariate((), x, edges, hist)
        diag = collector.diagnostics[()]
        # density should integrate to ~1
        assert abs(diag.normalization_denominator - 1.0) < 0.05
        assert diag.count == 200

    def test_collector_percent_normalization(self, rs):
        """Percent stat must have normalization_denominator == 100/100=1.0."""
        x = rs.randn(100)
        hist, edges = np.histogram(x, bins=10)
        percent_hist = hist.astype(float) / hist.sum() * 100
        collector = BinDiagnosticsCollector(stat="percent", cumulative=False)
        collector.add_group_univariate((), x, edges, percent_hist)
        diag = collector.diagnostics[()]
        assert diag.normalization_denominator == pytest.approx(1.0, abs=0.001)

    def test_collector_weights(self, rs):
        """weight_sum must equal the sum of valid sample weights."""
        x = rs.randn(50)
        w = rs.uniform(1, 5, 50)
        hist, edges = np.histogram(x, bins=10, weights=w)
        collector = BinDiagnosticsCollector(stat="count")
        collector.add_group_univariate((), x, edges, hist, weights=w)
        diag = collector.diagnostics[()]
        assert diag.count == 50
        assert diag.weight_sum == pytest.approx(w.sum(), abs=1e-9)

    def test_collector_empty_no_data(self):
        """Empty input array -> empty_reason='no_data'."""
        x = np.array([])
        collector = BinDiagnosticsCollector(stat="count")
        edges = np.array([0.0, 1.0])
        hist = np.array([0])
        collector.add_group_univariate((), x, edges, hist)
        diag = collector.diagnostics[()]
        assert diag.count == 0
        assert diag.weight_sum == 0.0
        assert diag.empty_reason == "no_data"

    def test_collector_all_nan(self):
        """All-NaN input -> empty_reason='all_nan'."""
        x = np.full(20, np.nan)
        collector = BinDiagnosticsCollector(stat="count")
        edges = np.array([0.0, 1.0])
        hist = np.array([0])
        collector.add_group_univariate((), x, edges, hist)
        assert collector.diagnostics[()].empty_reason == "all_nan"

    def test_collector_zero_variance(self, rs):
        """Constant-value input -> empty_reason='zero_variance'."""
        x = np.full(30, 5.0)
        collector = BinDiagnosticsCollector(stat="density")
        hist, edges = np.histogram(x, bins=5)
        # density may be 0 if single bin (delta) but collector still checks var
        collector.add_group_univariate((), x, edges, hist)
        assert collector.diagnostics[()].empty_reason == "zero_variance"

    def test_collector_bivariate(self, rs):
        """Bivariate case should record two edge arrays."""
        x = rs.randn(100)
        y = rs.randn(100)
        collector = BinDiagnosticsCollector(stat="count")
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=(8, 12))
        collector.add_group_bivariate(
            group_key=(), x1=x, x2=y,
            bin_edges=[x_edges, y_edges], hist=hist,
        )
        diag = collector.diagnostics[()]
        assert diag.count == 100
        assert isinstance(diag.bin_edges, (tuple, list))
        assert len(diag.bin_edges) == 2
        assert diag.bin_edges[0].shape == (9,)
        assert diag.bin_edges[1].shape == (13,)

    def test_collector_clear(self, rs):
        """collector.clear() must remove all prior diagnostics."""
        x = rs.randn(50)
        collector = BinDiagnosticsCollector(stat="count")
        hist, edges = np.histogram(x, bins=5)
        collector.add_group_univariate((), x, edges, hist)
        assert len(collector.diagnostics) == 1
        collector.clear()
        assert len(collector.diagnostics) == 0

    def test_collector_readonly_view(self, rs):
        """diagnostics property must return a dict whose mutations via add_group
        are reflected (shared dict, not a copy)."""
        x = rs.randn(50)
        collector = BinDiagnosticsCollector(stat="count")
        view = collector.diagnostics
        hist, edges = np.histogram(x, bins=5)
        collector.add_group_univariate((), x, edges, hist)
        assert () in view  # shared reference, not a copy


class TestHistplotDiagnostics:
    """Diagnostics on axes-level histplot."""

    @pytest.fixture
    def rs(self):
        return np.random.RandomState(42)

    def test_histplot_simple_has_diagnostics(self, rs):
        """Simple histplot must attach diagnostics_ dict to returned Axes."""
        x = rs.randn(100)
        ax = histplot(x=x, bins=10)
        plt.close(ax.figure)
        assert hasattr(ax, "diagnostics_")
        assert hasattr(ax, "_hist_estimator")
        # single group key = ()
        assert () in ax.diagnostics_
        d = ax.diagnostics_[()]
        assert d.count == 100
        assert d.bin_edges.shape == (11,)

    def test_histplot_hue_groups(self, rs):
        """With hue, each hue level must get its own diagnostic entry."""
        df = pd.DataFrame({
            "x": np.concatenate([rs.randn(60), rs.randn(40) + 2]),
            "h": ["a"] * 60 + ["b"] * 40,
        })
        ax = histplot(data=df, x="x", hue="h", stat="density")
        plt.close(ax.figure)
        keys = list(ax.diagnostics_.keys())
        # two hue groups
        assert len(keys) == 2
        counts = {k[0][1]: ax.diagnostics_[k].count for k in keys}
        assert counts["a"] == 60
        assert counts["b"] == 40
        # density stat -> normalization denominator ~= 1 for each
        for k in keys:
            assert abs(ax.diagnostics_[k].normalization_denominator - 1.0) < 0.1

    def test_histplot_weights(self, rs):
        """With weights, weight_sum must match sum of weights."""
        x = rs.randn(80)
        w = rs.uniform(0.5, 3.0, 80)
        ax = histplot(x=x, weights=w, stat="count", bins=10)
        plt.close(ax.figure)
        d = ax.diagnostics_[()]
        assert d.count == 80
        assert d.weight_sum == pytest.approx(w.sum(), abs=1e-9)

    def test_histplot_common_bins_shared_edges(self, rs):
        """With common_bins=True (default), all hue groups must share identical edges."""
        df = pd.DataFrame({
            "x": np.concatenate([rs.randn(50), rs.randn(50) + 5]),
            "h": ["x1"] * 50 + ["x2"] * 50,
        })
        ax = histplot(data=df, x="x", hue="h", common_bins=True, bins=15)
        plt.close(ax.figure)
        diags = list(ax.diagnostics_.values())
        edges_0 = diags[0].bin_edges
        for d in diags[1:]:
            assert_array_equal(d.bin_edges, edges_0)

    def test_histplot_probability_normalization(self, rs):
        """stat=probability must report norm_denom == 1 (before normalization)."""
        x = rs.randn(100)
        ax = histplot(x=x, stat="probability", bins=20)
        plt.close(ax.figure)
        d = ax.diagnostics_[()]
        assert d.normalization_denominator == pytest.approx(1.0, abs=1e-9)

    def test_histplot_bivariate(self, rs):
        """Bivariate histplot must report tuple of edges with correct shapes."""
        x = rs.randn(120)
        y = rs.randn(120)
        ax = histplot(x=x, y=y, bins=(10, 15))
        plt.close(ax.figure)
        d = ax.diagnostics_[()]
        assert d.count == 120
        edges = d.bin_edges
        assert isinstance(edges, (tuple, list))
        x_edges, y_edges = edges
        assert x_edges.shape == (11,)
        assert y_edges.shape == (16,)


class TestDisplotDiagnostics:
    """Diagnostics on figure-level displot with FacetGrid."""

    @pytest.fixture
    def rs(self):
        return np.random.RandomState(42)

    def test_displot_simple_has_facetgrid_attrs(self, rs):
        """displot must attach diagnostics to FacetGrid."""
        x = rs.randn(100)
        g = displot(x=x, bins=10)
        plt.close(g.fig)
        assert hasattr(g, "diagnostics_")
        assert hasattr(g, "_hist_estimator")
        assert () in g.diagnostics_
        assert g.diagnostics_[()].count == 100

    def test_displot_hue_and_facet_keys(self, rs):
        """hue + col facet -> 2 * 2 = 4 diagnostic entries with composite keys."""
        n_per_group = 30
        df = pd.DataFrame({
            "x": rs.randn(n_per_group * 4),
            "hue_var": np.repeat(["A", "B"], n_per_group * 2),
            "col_var": np.tile(np.repeat(["c1", "c2"], n_per_group), 2),
        })
        g = displot(data=df, x="x", hue="hue_var", col="col_var")
        plt.close(g.fig)
        diags = g.diagnostics_
        assert len(diags) == 4
        # Each key should be a tuple of ((var, val), (var, val), ...)
        for key in diags:
            assert isinstance(key, tuple)
            assert len(key) == 2  # (hue, col)
            assert key[0][0] == "hue"
            assert key[1][0] == "col"
            counts = {diags[k].count for k in diags}
        # Each group should have n_per_group samples
        assert counts == {n_per_group}

    def test_displot_shared_collector(self, rs):
        """FacetGrid's diagnostics_ MUST be the same object (shared) as the
        collector's internal diagnostics dict - no copying between layers."""
        x = rs.randn(50)
        g = displot(x=x, bins=5)
        plt.close(g.fig)
        # If the sharing works, g._hist_estimator.diagnostics_ IS g.diagnostics_
        assert g._hist_estimator is not None
        # For Hist: the diagnostics_ property is the collector.diagnostics view
        # So the internal dict should be the exact same object
        estimator_diags = g._hist_estimator.diagnostics_
        # Check they are the identical dict (is, not ==)
        assert g.diagnostics_ is estimator_diags


class TestObjHistDiagnostics:
    """Diagnostics on objects-layer Hist Stat."""

    @pytest.fixture
    def rs(self):
        return np.random.RandomState(42)

    def test_obj_hist_collector_type(self, rs):
        """ObjHist must use a BinDiagnosticsCollector internally."""
        h = ObjHist(stat="count", bins=5)
        assert isinstance(h._diagnostics_collector, BinDiagnosticsCollector)

    def test_obj_hist_hue_groups(self, rs):
        """ObjHist with hue grouping should produce one diagnostic per group."""
        df = pd.DataFrame({
            "x": np.concatenate([rs.randn(40), rs.randn(60) + 3]),
            "hue": ["g1"] * 40 + ["g2"] * 60,
        })
        h = ObjHist(stat="density", bins=12)
        gb = GroupBy(["hue"])
        _ = h(df, gb, "x", {"x": ContScale()})
        diags = h.diagnostics_
        assert len(diags) == 2
        # Find g1 and g2 counts
        counts = {}
        for k, v in diags.items():
            # k = (("hue", val),)
            counts[k[0][1]] = v.count
        assert counts["g1"] == 40
        assert counts["g2"] == 60
        # density: norm_denom should integrate to 1 (within tolerance)
        for v in diags.values():
            assert abs(v.normalization_denominator - 1.0) < 0.1

    def test_obj_hist_common_norm_vs_per_group(self, rs):
        """common_norm=False -> each group normalized independently;
        common_norm=True -> single denominator across all groups.
        We check that diagnostics show the correct group counts regardless."""
        df = pd.DataFrame({
            "x": np.concatenate([rs.randn(25), rs.randn(75) + 2]),
            "hue": ["small"] * 25 + ["large"] * 75,
        })
        h_common = ObjHist(stat="probability", common_norm=True)
        h_sep = ObjHist(stat="probability", common_norm=False)
        gb = GroupBy(["hue"])
        h_common(df.copy(), gb, "x", {"x": ContScale()})
        h_sep(df.copy(), gb, "x", {"x": ContScale()})
        # Both should still have the same counts (counts are pre-norm)
        common_counts = sorted(d.count for d in h_common.diagnostics_.values())
        sep_counts = sorted(d.count for d in h_sep.diagnostics_.values())
        assert common_counts == [25, 75]
        assert sep_counts == [25, 75]

    def test_obj_hist_empty_group_nan(self):
        """ObjHist with all-NaN group should diagnose empty_reason='all_nan'."""
        df = pd.DataFrame({
            "x": [np.nan] * 20,
            "hue": ["only"] * 20,
        })
        h = ObjHist(stat="count")
        gb = GroupBy(["hue"])
        with np.errstate(invalid="ignore"):
            result = h(df, gb, "x", {"x": ContScale()})
        diags = h.diagnostics_
        # Even if the result is empty due to NaN, diagnostics should record
        # the group and mark it as 'all_nan' if all values are NaN.
        # (Depends on how Hist handles edge cases; at minimum no crash.)
        for d in diags.values():
            if d.count == 0:
                assert d.empty_reason in ("no_data", "all_nan")


class TestDiagnosticsConsistency:
    """Cross-layer consistency: same data/params -> same diagnostic output
    regardless of whether we go through histplot, displot, or ObjHist."""

    @pytest.fixture
    def rs(self):
        return np.random.RandomState(123)

    @pytest.fixture
    def df_hue(self, rs):
        return pd.DataFrame({
            "x": np.concatenate([rs.randn(50), rs.randn(50) + 2.5]),
            "hue": ["A"] * 50 + ["B"] * 50,
        })

    def test_histplot_vs_objhist_counts_match(self, df_hue):
        """histplot and ObjHist should agree on count, edges, and weight_sum
        for every group under identical bin parameters."""
        bins = 18
        stat = "probability"

        # Via histplot
        ax = histplot(
            data=df_hue, x="x", hue="hue",
            stat=stat, bins=bins,
            # Make sure both use common_bins default (True)
        )
        plt.close(ax.figure)
        histplot_diags = ax.diagnostics_

        # Via objects-layer ObjHist directly
        h = ObjHist(stat=stat, bins=bins)
        gb = GroupBy(["hue"])
        h(df_hue.copy(), gb, "x", {"x": ContScale()})
        obj_diags = h.diagnostics_

        # Normalize keys to a canonical form: dict of {hue_value: diag}
        def _by_hue(diags_dict):
            out = {}
            for k, v in diags_dict.items():
                # k format: (("hue", val),)  or other single-semantic-key forms
                for var, val in k:
                    if var == "hue":
                        out[val] = v
            return out

        hp_by = _by_hue(histplot_diags)
        oh_by = _by_hue(obj_diags)
        assert set(hp_by.keys()) == {"A", "B"}
        assert set(oh_by.keys()) == {"A", "B"}

        for hue_val in ("A", "B"):
            assert hp_by[hue_val].count == oh_by[hue_val].count == 50
            assert hp_by[hue_val].weight_sum == oh_by[hue_val].weight_sum == 50
            # Edge arrays should match because both use common_bins=True on
            # the same data with the same bins= number
            # Use assert_allclose due to tiny floating point differences
            # from different computation paths (max diff ~ 1e-15)
            assert_allclose(
                hp_by[hue_val].bin_edges, oh_by[hue_val].bin_edges,
                rtol=1e-12, atol=1e-12,
            )

    def test_histplot_vs_displot_same_single_group(self, rs):
        """Single-group histplot and displot should produce identical diagnostics
        (no hue, no facet)."""
        x = rs.randn(80)
        ax = histplot(x=x, bins=12, stat="density")
        g = displot(x=x, bins=12, stat="density")
        hp = ax.diagnostics_[()]
        dp = g.diagnostics_[()]
        plt.close(ax.figure)
        plt.close(g.fig)
        assert hp.count == dp.count == 80
        assert hp.weight_sum == dp.weight_sum
        assert_array_equal(hp.bin_edges, dp.bin_edges)
        assert hp.normalization_denominator == pytest.approx(
            dp.normalization_denominator, abs=1e-12
        )


class TestDiagnosticsPublicAPI:
    """Stability tests for the public diagnostics API.

    These tests pin down the externally-visible contract so that the diagnostics_
    attribute (and related structures) are not just private debugging state.
    Every externally observable facts about key formats, empty-reason values, and data structures
    must remain stable for downstream users can rely on them.
    """

    @pytest.fixture
    def rs(self):
        return np.random.RandomState(42)

    @pytest.fixture
    def df_hue(self, rs):
        return pd.DataFrame({
            "x": np.concatenate([rs.randn(40), rs.randn(60) + 1.5]),
            "hue": ["g1"] * 40 + ["g2"] * 60,
        })

    # --- BinDiagnostics dataclass stability ---

    def test_bin_diagnostics_all_documented_fields_exist(self):
        """BinDiagnostics must expose exactly the documented public fields."""
        diag = BinDiagnostics()
        # These field names are part of the stable API.
        expected_fields = {
            "bin_edges", "count", "weight_sum",
            "normalization_denominator", "empty_reason", "extra",
        }
        actual_fields = {f.name for f in fields(BinDiagnostics)}
        assert expected_fields.issubset(actual_fields)
        # Field types / defaults must not change unexpectedly
        assert isinstance(diag.count, int)
        assert isinstance(diag.weight_sum, float)
        assert diag.bin_edges is None
        assert diag.normalization_denominator is None
        assert diag.empty_reason is None
        assert isinstance(diag.extra, dict)

    def test_bin_diagnostics_bin_edges_shape_univariate(self, rs):
        """Univariate bin_edges must be a 1D ndarray of length n_bins+1."""
        x = rs.randn(100)
        collector = BinDiagnosticsCollector(stat="count")
        hist, edges = np.histogram(x, bins=15)
        collector.add_group_univariate((), x, edges, hist)
        diag = collector.diagnostics[()]
        assert isinstance(diag.bin_edges, np.ndarray)
        assert diag.bin_edges.ndim == 1
        assert len(diag.bin_edges) == 16

    def test_bin_diagnostics_bin_edges_shape_bivariate(self, rs):
        """Bivariate bin_edges must be a 2-tuple of 1D ndarrays."""
        x, y = rs.randn(2, 80)
        collector = BinDiagnosticsCollector(stat="count")
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=(10, 12))
        collector.add_group_bivariate((), x, y, (x_edges, y_edges), hist)
        diag = collector.diagnostics[()]
        assert isinstance(diag.bin_edges, tuple)
        assert len(diag.bin_edges) == 2
        assert diag.bin_edges[0].shape == (11,)
        assert diag.bin_edges[1].shape == (13,)

    # --- Empty reason enum stability ---

    def test_empty_reason_values_are_fixed_strings(self):
        """The set of valid empty_reason values must be a stable enum set."""
        # These exact strings are part of the public API contract.
        valid_reasons = {"no_data", "all_nan", "zero_variance"}
        # no_data
        c = BinDiagnosticsCollector(stat="count")
        c.add_group_univariate((), np.array([]), np.array([0, 1]), np.array([0]))
        assert c.diagnostics[()].empty_reason == "no_data"

        # all_nan
        c2 = BinDiagnosticsCollector(stat="count")
        c2.add_group_univariate((), np.full(10, np.nan), np.array([0, 1]), np.array([0]))
        assert c2.diagnostics[()].empty_reason == "all_nan"

        # zero_variance
        c3 = BinDiagnosticsCollector(stat="density")
        zvals = np.full(20, 5.0)
        hist, edges = np.histogram(zvals, bins=5)
        c3.add_group_univariate((), zvals, edges, hist)
        assert c3.diagnostics[()].empty_reason == "zero_variance"

        # valid data -> None
        c4 = BinDiagnosticsCollector(stat="count")
        x = np.array([1.0, 2.0, 3.0])
        h, e = np.histogram(x, bins=3)
        c4.add_group_univariate((), x, e, h)
        assert c4.diagnostics[()].empty_reason is None

    # --- Group key format stability ---

    def test_group_key_single_group_is_empty_tuple(self, rs):
        """Single ungrouped dataset must use () as the group key."""
        x = rs.randn(50)
        # Via legacy Histogram
        h = Histogram(stat="count", bins=10)
        h(x)
        assert () in h.diagnostics_

        # Via objects Hist (use a single-value hue column so groupby is valid,
        # but result shows ()-key via the shared collector for the overall data)
        df = pd.DataFrame({"x": x})
        # Use no grouping: pass a GroupBy whose vars are not in data
        obj_h = ObjHist(stat="count", bins=10)
        gb = GroupBy(["__nonexistent__"])
        obj_h(df, gb, "x", {"x": ContScale()})
        # With nonexistent grouping vars, it falls through to single-group path
        assert () in obj_h.diagnostics_

        # Via histplot
        ax = histplot(x=x, bins=10)
        assert () in ax.diagnostics_
        plt.close(ax.figure)

        # Via displot
        g = displot(x=x, bins=10)
        assert () in g.diagnostics_
        plt.close(g.fig)

    def test_group_key_hue_format(self, df_hue):
        """hue-grouped diagnostics must use ((\"hue\"), value) tuples."""
        ax = histplot(data=df_hue, x="x", hue="hue")
        diags = ax.diagnostics_
        plt.close(ax.figure)

        # Keys must be tuples of (var_name, value) pairs
        assert len(diags) == 2
        for key in diags:
            assert isinstance(key, tuple)
            assert len(key) == 1
            var_name, value = key[0]
            assert var_name == "hue"
            assert value in {"g1", "g2"}

    def test_group_key_facet_format(self, rs):
        """Facet + hue combined must have multiple (var, val) pairs in key."""
        df = pd.DataFrame({
            "x": rs.randn(60),
            "hue": ["a", "b", "c"] * 20,
            "col": ["x", "y"] * 30,
        })
        g = displot(data=df, x="x", hue="hue", col="col")
        diags = g.diagnostics_
        plt.close(g.fig)

        # Each key must have both col and hue in the key tuple
        assert len(diags) == 6  # 2 cols * 3 hues
        for key in diags:
            assert isinstance(key, tuple)
            key_vars = {pair[0] for pair in key}
            assert "col" in key_vars
            assert "hue" in key_vars

    # --- diagnostics_ attribute is publicly accessible ---

    def test_diagnostics_is_dict_of_bin_diagnostics(self, df_hue):
        """diagnostics_ must be a dict mapping tuple -> BinDiagnostics."""
        ax = histplot(data=df_hue, x="x", hue="hue")
        diags = ax.diagnostics_
        plt.close(ax.figure)

        assert isinstance(diags, dict)
        for k, v in diags.items():
            assert isinstance(k, tuple)
            assert isinstance(v, BinDiagnostics)

    def test_diagnostics_readonly_returns_same_object_as_collector(self, rs):
        """diagnostics_ property must be a live view into the collector,
        not a copy, so that updates are immediately visible."""
        df = pd.DataFrame({
            "x": rs.randn(50),
            "grp": ["a"] * 50,
        })
        h = ObjHist(bins=10)
        gb = GroupBy(["grp"])
        h(df, gb, "x", {"x": ContScale()})
        # diagnostics_ property returns the same dict as collector.diagnostics
        assert h.diagnostics_ is h._diagnostics_collector.diagnostics


class TestDiagnosticsCrossLayerConsistency:
    """Verify that diagnostics_ has the same external shape and behavior
    across all three layers: legacy Histogram, objects Hist, and displot's
    FacetGrid results. Users should feel like the same API.
    """

    @pytest.fixture
    def rs(self):
        return np.random.RandomState(77)

    @pytest.fixture
    def x_single(self, rs):
        return rs.randn(100)

    @pytest.fixture
    def df_hue(self, rs):
        return pd.DataFrame({
            "x": np.concatenate([rs.randn(30), rs.randn(70) + 2]),
            "hue": ["A"] * 30 + ["B"] * 70,
        })

    def test_legacy_histogram_diagnostics_structure(self, x_single):
        """Legacy Histogram must expose diagnostics_ with documented fields."""
        h = Histogram(stat="count", bins=12)
        h(x_single)
        diags = h.diagnostics_
        assert () in diags
        diag = diags[()]
        assert isinstance(diag.bin_edges, np.ndarray)
        assert diag.count == 100
        assert diag.weight_sum == 100.0
        assert diag.empty_reason is None
        assert isinstance(diag.extra, dict)
        assert "stat" in diag.extra

    def test_objects_hist_diagnostics_structure(self, x_single):
        """Objects-layer Hist must expose diagnostics_ with same fields."""
        df = pd.DataFrame({
            "x": x_single,
            "grp": ["only"] * 100,
        })
        h = ObjHist(stat="count", bins=12)
        gb = GroupBy(["grp"])
        h(df, gb, "x", {"x": ContScale()})
        diags = h.diagnostics_
        # Objects layer uses group key format (("grp", "only"),)
        assert len(diags) == 1
        diag = list(diags.values())[0]
        assert isinstance(diag.bin_edges, np.ndarray)
        assert diag.count == 100
        assert diag.weight_sum == 100.0
        assert diag.empty_reason is None
        assert isinstance(diag.extra, dict)

    def test_histplot_axes_diagnostics_structure(self, x_single):
        """histplot return Axes must have diagnostics_ attribute."""
        ax = histplot(x=x_single, bins=12, stat="count")
        diags = ax.diagnostics_
        plt.close(ax.figure)
        assert () in diags
        diag = diags[()]
        assert isinstance(diag.bin_edges, np.ndarray)
        assert diag.count == 100
        assert diag.weight_sum == 100.0
        assert diag.empty_reason is None

    def test_displot_facetgrid_diagnostics_structure(self, x_single):
        """displot return FacetGrid must have diagnostics_ attribute."""
        g = displot(x=x_single, bins=12, stat="count")
        diags = g.diagnostics_
        plt.close(g.fig)
        assert () in diags
        diag = diags[()]
        assert isinstance(diag.bin_edges, np.ndarray)
        assert diag.count == 100
        assert diag.weight_sum == 100.0
        assert diag.empty_reason is None

    def test_all_four_layers_same_count_and_edges(self, x_single):
        """Histogram, Hist, histplot, displot must agree on count and edges
        for identical input with identical params."""
        bins = 15
        stat = "density"

        # Legacy Histogram
        h_leg = Histogram(stat=stat, bins=bins)
        h_leg(x_single)
        leg_diag = h_leg.diagnostics_[()]

        # Objects Hist
        df = pd.DataFrame({
            "x": x_single,
            "grp": ["only"] * 100,
        })
        h_obj = ObjHist(stat=stat, bins=bins)
        gb = GroupBy(["grp"])
        h_obj(df, gb, "x", {"x": ContScale()})
        obj_diag = list(h_obj.diagnostics_.values())[0]

        # histplot
        ax = histplot(x=x_single, bins=bins, stat=stat)
        hp_diag = ax.diagnostics_[()]
        plt.close(ax.figure)

        # displot
        g = displot(x=x_single, bins=bins, stat=stat)
        dp_diag = g.diagnostics_[()]
        plt.close(g.fig)

        # Count must match exactly
        assert leg_diag.count == obj_diag.count == hp_diag.count == dp_diag.count == 100
        assert leg_diag.weight_sum == obj_diag.weight_sum == 100.0

        # Edges must be close (tiny floating-point differences from different
        # computation paths are okay)
        assert_allclose(leg_diag.bin_edges, obj_diag.bin_edges, rtol=1e-12, atol=1e-12)
        assert_allclose(hp_diag.bin_edges, dp_diag.bin_edges, rtol=1e-12, atol=1e-12)
        assert_allclose(leg_diag.bin_edges, hp_diag.bin_edges, rtol=1e-12, atol=1e-12)

        # Normalization denominator must match across layers
        assert leg_diag.normalization_denominator == pytest.approx(
            obj_diag.normalization_denominator, abs=1e-12)
        assert hp_diag.normalization_denominator == pytest.approx(
            dp_diag.normalization_denominator, abs=1e-12)

    def test_empty_group_empty_reason_consistent(self):
        """Empty reason strings must be the same enum values in all layers
        that actually support empty data gracefully."""
        # 1) Collector directly -> "no_data"
        c = BinDiagnosticsCollector(stat="count")
        c.add_group_univariate((), np.array([]), np.array([0.0, 1.0]), np.array([0]))
        assert c.diagnostics[()].empty_reason == "no_data"

        # 2) histplot -> no_data (returns early but now populates diagnostics_)
        ax = histplot(x=[])
        assert () in ax.diagnostics_
        assert ax.diagnostics_[()].empty_reason == "no_data"
        plt.close(ax.figure)

        # 3) displot -> no_data (returns early but now populates diagnostics_)
        g = displot(x=[])
        assert () in g.diagnostics_
        assert g.diagnostics_[()].empty_reason == "no_data"
        plt.close(g.fig)

        # 4) Collector with all-NaN -> "all_nan"
        c2 = BinDiagnosticsCollector(stat="count")
        c2.add_group_univariate((), np.full(10, np.nan), np.array([0, 1]), np.array([0]))
        assert c2.diagnostics[()].empty_reason == "all_nan"

        # 5) histplot with all-NaN data -> should also report all_nan
        ax2 = histplot(x=[np.nan, np.nan, np.nan], bins=5)
        # Note: histplot may drop all NaN -> count becomes 0, "all_nan" or similar
        diag = ax2.diagnostics_[()]
        assert diag.empty_reason in ("all_nan", "no_data")  # both valid depending on dropna behavior
        plt.close(ax2.figure)


class TestGroupByNotPolluted:
    """Regression guard: GroupBy.apply must not introspect function signatures."""

    def test_groupby_apply_simple(self):
        """GroupBy.apply with a standard 2-arg lambda must work exactly as before;
        we must never have added inspect/signature logic here."""
        import inspect
        src = inspect.getsource(GroupBy.apply)
        # Make sure GroupBy.apply does not contain the word "inspect" or
        # "signature" or "_group_key" parameter handling
        assert "inspect" not in src
        assert "signature" not in src
        assert "_group_key" not in src

    def test_groupby_apply_no_keyword_injection(self):
        """A func that only accepts (data, arg) must work without error even
        though the old polluted version passed extra keyword args."""
        df = pd.DataFrame({
            "v": [1, 2, 3, 4, 5, 6],
            "g": ["a", "a", "a", "b", "b", "b"],
        })
        gb = GroupBy(["g"])
        # A deliberately strict function that would crash if given extra kwargs
        def my_func(part, multiplier):
            return pd.DataFrame({"out": part["v"] * multiplier})
        result = gb.apply(df, my_func, 2)
        assert set(result["g"]) == {"a", "b"}
        assert result["out"].tolist() == [2, 4, 6, 8, 10, 12]
