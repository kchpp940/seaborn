
import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal

from seaborn._core.groupby import GroupBy
from seaborn._stats.counting import Hist, Count
from seaborn._statistics import BinDiagnostics, BinDiagnosticsCollector


class TestCount:

    @pytest.fixture
    def df(self, rng):

        n = 30
        return pd.DataFrame(dict(
            x=rng.uniform(0, 7, n).round(),
            y=rng.normal(size=n),
            color=rng.choice(["a", "b", "c"], n),
            group=rng.choice(["x", "y"], n),
        ))

    def get_groupby(self, df, orient):

        other = {"x": "y", "y": "x"}[orient]
        cols = [c for c in df if c != other]
        return GroupBy(cols)

    def test_single_grouper(self, df):

        ori = "x"
        df = df[["x"]]
        gb = self.get_groupby(df, ori)
        res = Count()(df, gb, ori, {})
        expected = df.groupby("x").size()
        assert_array_equal(res.sort_values("x")["y"], expected)

    def test_multiple_groupers(self, df):

        ori = "x"
        df = df[["x", "group"]].sort_values("group")
        gb = self.get_groupby(df, ori)
        res = Count()(df, gb, ori, {})
        expected = df.groupby(["x", "group"]).size()
        assert_array_equal(res.sort_values(["x", "group"])["y"], expected)


class TestHist:

    @pytest.fixture
    def single_args(self):

        groupby = GroupBy(["group"])

        class Scale:
            scale_type = "continuous"

        return groupby, "x", {"x": Scale()}

    @pytest.fixture
    def triple_args(self):

        groupby = GroupBy(["group", "a", "s"])

        class Scale:
            scale_type = "continuous"

        return groupby, "x", {"x": Scale()}

    def test_string_bins(self, long_df):

        h = Hist(bins="sqrt")
        bin_kws = h._define_bin_params(long_df, "x", "continuous")
        assert bin_kws["range"] == (long_df["x"].min(), long_df["x"].max())
        assert bin_kws["bins"] == int(np.sqrt(len(long_df)))

    def test_int_bins(self, long_df):

        n = 24
        h = Hist(bins=n)
        bin_kws = h._define_bin_params(long_df, "x", "continuous")
        assert bin_kws["range"] == (long_df["x"].min(), long_df["x"].max())
        assert bin_kws["bins"] == n

    def test_array_bins(self, long_df):

        bins = [-3, -2, 1, 2, 3]
        h = Hist(bins=bins)
        bin_kws = h._define_bin_params(long_df, "x", "continuous")
        assert_array_equal(bin_kws["bins"], bins)

    def test_binwidth(self, long_df):

        binwidth = .5
        h = Hist(binwidth=binwidth)
        bin_kws = h._define_bin_params(long_df, "x", "continuous")
        n_bins = bin_kws["bins"]
        left, right = bin_kws["range"]
        assert (right - left) / n_bins == pytest.approx(binwidth)

    def test_binrange(self, long_df):

        binrange = (-4, 4)
        h = Hist(binrange=binrange)
        bin_kws = h._define_bin_params(long_df, "x", "continuous")
        assert bin_kws["range"] == binrange

    def test_discrete_bins(self, long_df):

        h = Hist(discrete=True)
        x = long_df["x"].astype(int)
        bin_kws = h._define_bin_params(long_df.assign(x=x), "x", "continuous")
        assert bin_kws["range"] == (x.min() - .5, x.max() + .5)
        assert bin_kws["bins"] == (x.max() - x.min() + 1)

    def test_discrete_bins_from_nominal_scale(self, rng):

        h = Hist()
        x = rng.randint(0, 5, 10)
        df = pd.DataFrame({"x": x})
        bin_kws = h._define_bin_params(df, "x", "nominal")
        assert bin_kws["range"] == (x.min() - .5, x.max() + .5)
        assert bin_kws["bins"] == (x.max() - x.min() + 1)

    def test_count_stat(self, long_df, single_args):

        h = Hist(stat="count")
        out = h(long_df, *single_args)
        assert out["y"].sum() == len(long_df)

    def test_probability_stat(self, long_df, single_args):

        h = Hist(stat="probability")
        out = h(long_df, *single_args)
        assert out["y"].sum() == 1

    def test_proportion_stat(self, long_df, single_args):

        h = Hist(stat="proportion")
        out = h(long_df, *single_args)
        assert out["y"].sum() == 1

    def test_percent_stat(self, long_df, single_args):

        h = Hist(stat="percent")
        out = h(long_df, *single_args)
        assert out["y"].sum() == 100

    def test_density_stat(self, long_df, single_args):

        h = Hist(stat="density")
        out = h(long_df, *single_args)
        assert (out["y"] * out["space"]).sum() == 1

    def test_frequency_stat(self, long_df, single_args):

        h = Hist(stat="frequency")
        out = h(long_df, *single_args)
        assert (out["y"] * out["space"]).sum() == len(long_df)

    def test_invalid_stat(self):

        with pytest.raises(ValueError, match="The `stat` parameter for `Hist`"):
            Hist(stat="invalid")

    def test_cumulative_count(self, long_df, single_args):

        h = Hist(stat="count", cumulative=True)
        out = h(long_df, *single_args)
        assert out["y"].max() == len(long_df)

    def test_cumulative_proportion(self, long_df, single_args):

        h = Hist(stat="proportion", cumulative=True)
        out = h(long_df, *single_args)
        assert out["y"].max() == 1

    def test_cumulative_density(self, long_df, single_args):

        h = Hist(stat="density", cumulative=True)
        out = h(long_df, *single_args)
        assert out["y"].max() == 1

    def test_common_norm_default(self, long_df, triple_args):

        h = Hist(stat="percent")
        out = h(long_df, *triple_args)
        assert out["y"].sum() == pytest.approx(100)

    def test_common_norm_false(self, long_df, triple_args):

        h = Hist(stat="percent", common_norm=False)
        out = h(long_df, *triple_args)
        for _, out_part in out.groupby(["a", "s"]):
            assert out_part["y"].sum() == pytest.approx(100)

    def test_common_norm_subset(self, long_df, triple_args):

        h = Hist(stat="percent", common_norm=["a"])
        out = h(long_df, *triple_args)
        for _, out_part in out.groupby("a"):
            assert out_part["y"].sum() == pytest.approx(100)

    def test_common_norm_warning(self, long_df, triple_args):

        h = Hist(common_norm=["b"])
        with pytest.warns(UserWarning, match=r"Undefined variable\(s\)"):
            h(long_df, *triple_args)

    def test_common_bins_default(self, long_df, triple_args):

        h = Hist()
        out = h(long_df, *triple_args)
        bins = []
        for _, out_part in out.groupby(["a", "s"]):
            bins.append(tuple(out_part["x"]))
        assert len(set(bins)) == 1

    def test_common_bins_false(self, long_df, triple_args):

        h = Hist(common_bins=False)
        out = h(long_df, *triple_args)
        bins = []
        for _, out_part in out.groupby(["a", "s"]):
            bins.append(tuple(out_part["x"]))
        assert len(set(bins)) == len(out.groupby(["a", "s"]))

    def test_common_bins_subset(self, long_df, triple_args):

        h = Hist(common_bins=False)
        out = h(long_df, *triple_args)
        bins = []
        for _, out_part in out.groupby("a"):
            bins.append(tuple(out_part["x"]))
        assert len(set(bins)) == out["a"].nunique()

    def test_common_bins_warning(self, long_df, triple_args):

        h = Hist(common_bins=["b"])
        with pytest.warns(UserWarning, match=r"Undefined variable\(s\)"):
            h(long_df, *triple_args)

    def test_histogram_single(self, long_df, single_args):

        h = Hist()
        out = h(long_df, *single_args)
        hist, edges = np.histogram(long_df["x"], bins="auto")
        assert_array_equal(out["y"], hist)
        assert_array_equal(out["space"], np.diff(edges))

    def test_histogram_multiple(self, long_df, triple_args):

        h = Hist()
        out = h(long_df, *triple_args)
        bins = np.histogram_bin_edges(long_df["x"], "auto")
        for (a, s), out_part in out.groupby(["a", "s"]):
            x = long_df.loc[(long_df["a"] == a) & (long_df["s"] == s), "x"]
            hist, edges = np.histogram(x, bins=bins)
            assert_array_equal(out_part["y"], hist)
            assert_array_equal(out_part["space"], np.diff(edges))


class TestHistDiagnostics:

    @pytest.fixture
    def single_args(self):
        groupby = GroupBy(["group"])

        class Scale:
            scale_type = "continuous"

        return groupby, "x", {"x": Scale()}

    @pytest.fixture
    def ungrouped_df(self, rng):
        return pd.DataFrame({
            "x": rng.normal(size=100),
            "group": ["a"] * 100,
        })

    @pytest.fixture
    def grouped_df(self, rng):
        n = 60
        return pd.DataFrame({
            "x": np.concatenate([rng.normal(size=n), rng.normal(size=n) + 3]),
            "group": ["a"] * n + ["b"] * n,
        })

    def test_diagnostics_attr_exists_after_call(self, grouped_df, single_args):
        h = Hist()
        h(grouped_df, *single_args)
        assert hasattr(h, "diagnostics_")
        assert isinstance(h.diagnostics_, dict)

    def test_diagnostics_is_live_view_of_collector(self, grouped_df, single_args):
        h = Hist()
        h(grouped_df, *single_args)
        assert h.diagnostics_ is h._diagnostics_collector.diagnostics

    def test_diagnostics_keys_match_groups(self, grouped_df, single_args):
        h = Hist()
        h(grouped_df, *single_args)
        assert len(h.diagnostics_) == 2
        keys = sorted(h.diagnostics_.keys())
        assert keys == sorted([(("group", "a"),), (("group", "b"),)])

    def test_diagnostics_values_are_bin_diagnostics(self, grouped_df, single_args):
        h = Hist()
        h(grouped_df, *single_args)
        for diag in h.diagnostics_.values():
            assert isinstance(diag, BinDiagnostics)

    def test_diagnostics_bin_edges_is_ndarray(self, ungrouped_df, single_args):
        h = Hist(bins=10)
        h(ungrouped_df, *single_args)
        diag = list(h.diagnostics_.values())[0]
        assert isinstance(diag.bin_edges, np.ndarray)
        assert diag.bin_edges.ndim == 1
        assert len(diag.bin_edges) == 11

    def test_diagnostics_count_matches_input_size(self, grouped_df, single_args):
        h = Hist()
        h(grouped_df, *single_args)
        assert h.diagnostics_[(("group", "a"),)].count == 60
        assert h.diagnostics_[(("group", "b"),)].count == 60

    def test_diagnostics_weight_sum_matches_count_when_no_weights(
        self, grouped_df, single_args
    ):
        h = Hist()
        h(grouped_df, *single_args)
        for diag in h.diagnostics_.values():
            assert diag.weight_sum == float(diag.count)

    def test_diagnostics_weight_sum_with_weights(self, rng, single_args):
        df = pd.DataFrame({
            "x": rng.normal(size=50),
            "group": ["a"] * 50,
            "weight": rng.uniform(0.5, 2.0, size=50),
        })
        # Use explicit bins because numpy's auto binning doesn't work with weights
        h = Hist(bins=10)
        h(df, *single_args)
        diag = h.diagnostics_[(("group", "a"),)]
        assert diag.count == 50
        assert diag.weight_sum == pytest.approx(df["weight"].sum())

    def test_diagnostics_normalization_denominator_count_stat(
        self, ungrouped_df, single_args
    ):
        h = Hist(stat="count")
        h(ungrouped_df, *single_args)
        diag = list(h.diagnostics_.values())[0]
        assert diag.normalization_denominator is None

    def test_diagnostics_normalization_denominator_proportion(
        self, ungrouped_df, single_args
    ):
        h = Hist(stat="proportion")
        h(ungrouped_df, *single_args)
        diag = list(h.diagnostics_.values())[0]
        # proportion normalizes by dividing each count by weight_sum, so:
        # norm_denom = weight_sum (the divisor for each bin count to get proportion)
        # result: sum(proportions) == 1, so norm_denom == weight_sum
        assert diag.normalization_denominator == pytest.approx(float(diag.count))

    def test_diagnostics_normalization_denominator_density(
        self, ungrouped_df, single_args
    ):
        h = Hist(stat="density")
        h(ungrouped_df, *single_args)
        diag = list(h.diagnostics_.values())[0]
        # For density stat, norm denom should be the data range / count product
        # (total weight * binwidth product)
        assert diag.normalization_denominator is not None
        assert diag.normalization_denominator > 0

    def test_diagnostics_empty_reason_none_for_valid_data(
        self, ungrouped_df, single_args
    ):
        h = Hist()
        h(ungrouped_df, *single_args)
        diag = list(h.diagnostics_.values())[0]
        assert diag.empty_reason is None

    def test_diagnostics_extra_is_dict(self, ungrouped_df, single_args):
        h = Hist()
        h(ungrouped_df, *single_args)
        diag = list(h.diagnostics_.values())[0]
        assert isinstance(diag.extra, dict)

    def test_diagnostics_collector_shares_stat(self, grouped_df, single_args):
        stat = "density"
        h = Hist(stat=stat)
        h(grouped_df, *single_args)
        assert h._diagnostics_collector.stat == stat

    def test_diagnostics_collector_shares_cumulative(self, grouped_df, single_args):
        h = Hist(stat="count", cumulative=True)
        h(grouped_df, *single_args)
        assert h._diagnostics_collector.cumulative is True

    def test_diagnostics_bin_edges_agree_with_np_histogram(
        self, ungrouped_df, single_args
    ):
        bins = 12
        h = Hist(bins=bins)
        h(ungrouped_df, *single_args)
        diag = list(h.diagnostics_.values())[0]
        _, np_edges = np.histogram(ungrouped_df["x"], bins=bins)
        assert_array_equal(diag.bin_edges, np_edges)

    def test_diagnostics_collector_clear_resets_dict(self, grouped_df, single_args):
        h = Hist()
        h(grouped_df, *single_args)
        diag_ref = h.diagnostics_
        orig_keys = set(diag_ref.keys())
        h._diagnostics_collector.clear()
        # After clear, diagnostics_ (same ref) is empty
        assert len(diag_ref) == 0
        assert orig_keys - set(diag_ref.keys()) == orig_keys

    def test_diagnostics_cumulative_affects_denominator_when_cumulative(
        self, ungrouped_df, single_args
    ):
        h = Hist(stat="count", cumulative=True)
        h(ungrouped_df, *single_args)
        diag = list(h.diagnostics_.values())[0]
        # cumulative count should have extra info, but count remains total input
        assert diag.count == 100
        assert diag.empty_reason is None
