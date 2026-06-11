import numpy as np
import pandas as pd
import pytest

from seaborn._core.plot import Plot
from seaborn._core.scales import Nominal, Continuous
from seaborn._marks.dot import Dot
from seaborn._marks.line import Line
from seaborn._marks.bar import Bar
from seaborn._stats.aggregation import Agg
from seaborn._stats.counting import Hist


class TestHoverMetadataStructure:
    """Tests for the structure and types of hover metadata output."""

    def test_basic_structure(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()

        assert "subplots" in meta
        assert "facet_spec" in meta
        assert "pair_spec" in meta
        assert "labels" in meta
        assert isinstance(meta["subplots"], list)
        assert len(meta["subplots"]) == 1

        subplot = meta["subplots"][0]
        assert "subplot_index" in subplot
        assert subplot["subplot_index"] == (0, 0)
        assert "layers" in subplot
        assert len(subplot["layers"]) == 1

        layer = subplot["layers"][0]
        assert "mark_type" in layer
        assert layer["mark_type"] == "Dot"
        assert "layer_index" in layer
        assert layer["layer_index"] == 0
        assert "variables" in layer
        assert "x" in layer["variables"]
        assert "y" in layer["variables"]

    def test_variable_metadata_structure(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "c": ["a", "b", "a"]})
        meta = Plot(df, "x", "y").add(Dot(), color="c").hover_metadata()

        var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        assert "variable" in var
        assert "original_values" in var
        assert "scaled_values" in var
        assert "display_labels" in var
        assert "legend_values" in var
        assert "coord_range" in var
        assert "scale_type" in var
        assert "property_type" in var

        assert var["variable"] == "x"
        assert isinstance(var["original_values"], np.ndarray)
        assert isinstance(var["scaled_values"], np.ndarray)
        assert var["original_values"].shape == var["scaled_values"].shape
        assert var["coord_range"] is not None
        assert isinstance(var["coord_range"], tuple)
        assert len(var["coord_range"]) == 2


class TestHoverMetadataCompilationConsistency:
    """Tests that hover metadata is consistent with actual plot compilation."""

    def test_hover_metadata_uses_same_compilation_path(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "c": ["a", "b", "a"]})
        p = Plot(df, "x", "y").add(Dot(), color="c")

        plotter = p.plot()
        assert plotter._hover_metadata is not None

        meta = p.hover_metadata()
        assert meta is not None

        assert len(plotter._hover_metadata["subplots"]) == len(meta["subplots"])
        assert (
            len(plotter._hover_metadata["subplots"][0]["layers"])
            == len(meta["subplots"][0]["layers"])
        )

        color1 = plotter._hover_metadata["subplots"][0]["layers"][0]["variables"]["color"]
        color2 = meta["subplots"][0]["layers"][0]["variables"]["color"]
        assert color1["scale_type"] == color2["scale_type"]
        assert color1["display_labels"] == color2["display_labels"]
        np.testing.assert_array_equal(color1["original_values"], color2["original_values"])
        np.testing.assert_array_equal(color1["scaled_values"], color2["scaled_values"])

    def test_scale_consistency_between_plot_and_metadata(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "c": ["a", "b", "c"]})
        p = (
            Plot(df, "x", "y")
            .add(Dot(), color="c")
            .scale(color=Nominal(["red", "green", "blue"], order=["c", "b", "a"]))
        )

        plotter = p.plot()
        meta = p.hover_metadata()
        color_meta = meta["subplots"][0]["layers"][0]["variables"]["color"]

        assert color_meta["legend_values"] == ["c", "b", "a"]
        assert color_meta["display_labels"] == ["c", "b", "a"]

        actual_scales = plotter._scales["color"]
        assert actual_scales.order == ["c", "b", "a"]


class TestHoverMetadataWithStatTransforms:
    """Tests that stat-transformed data is correctly reflected in metadata."""

    def test_agg_original_values_are_aggregated(self):
        df = pd.DataFrame({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "val": [1, 3, 2, 4, 5, 7],
        })
        meta = (
            Plot(df, "cat", "val")
            .add(Bar(), Agg("mean"))
            .hover_metadata()
        )

        y_var = meta["subplots"][0]["layers"][0]["variables"]["y"]
        expected_means = np.array([2.0, 3.0, 6.0])
        np.testing.assert_array_almost_equal(y_var["original_values"], expected_means)
        np.testing.assert_array_almost_equal(y_var["scaled_values"], expected_means)
        assert y_var["coord_range"] == (2.0, 6.0)

    def test_hist_stat_transform(self):
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.randn(100)})
        meta = (
            Plot(df, "x")
            .add(Bar(), Hist(bins=10))
            .hover_metadata()
        )

        layer = meta["subplots"][0]["layers"][0]
        assert layer["mark_type"] == "Bar"
        assert "x" in layer["variables"]
        assert "y" in layer["variables"]
        y_var = layer["variables"]["y"]
        assert y_var["coord_range"] is not None
        assert y_var["coord_range"][0] >= 0
        assert sum(y_var["original_values"]) == 100


class TestHoverMetadataWithFacet:
    """Tests for facet metadata, including empty subsets."""

    def test_facet_col_structure(self):
        df = pd.DataFrame({
            "x": [1, 2, 1, 2],
            "y": [1, 2, 3, 4],
            "col": ["a", "a", "b", "b"],
        })
        meta = (
            Plot(df, "x", "y")
            .add(Dot())
            .facet(col="col")
            .hover_metadata()
        )

        assert len(meta["subplots"]) == 2
        cols = [s["col"] for s in meta["subplots"]]
        assert set(cols) == {"a", "b"}

        for subplot in meta["subplots"]:
            assert subplot["row"] is None
            assert subplot["subplot_index"][0] == 0
            assert len(subplot["layers"]) == 1
            n_points = len(subplot["layers"][0]["variables"]["x"]["original_values"])
            assert n_points == 2

    def test_facet_row_and_col(self):
        df = pd.DataFrame({
            "x": [1, 2, 1, 2],
            "y": [1, 2, 3, 4],
            "row": ["p", "p", "q", "q"],
            "col": ["a", "b", "a", "b"],
        })
        meta = (
            Plot(df, "x", "y")
            .add(Dot())
            .facet(row="row", col="col")
            .hover_metadata()
        )

        assert len(meta["subplots"]) == 4
        positions = [(s["row"], s["col"]) for s in meta["subplots"]]
        assert set(positions) == {("p", "a"), ("p", "b"), ("q", "a"), ("q", "b")}

        for subplot in meta["subplots"]:
            n_points = len(subplot["layers"][0]["variables"]["x"]["original_values"])
            assert n_points == 1

    def test_facet_empty_subsets(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [1, 2, 3],
            "row": ["p", "p", "q"],
            "col": ["a", "b", "a"],
        })
        meta = (
            Plot(df, "x", "y")
            .add(Dot())
            .facet(row="row", col="col")
            .hover_metadata()
        )

        assert len(meta["subplots"]) == 4

        for subplot in meta["subplots"]:
            if (subplot["row"], subplot["col"]) == ("q", "b"):
                assert len(subplot["layers"]) == 0
            else:
                assert len(subplot["layers"]) == 1
                n_points = len(subplot["layers"][0]["variables"]["x"]["original_values"])
                assert n_points == 1


class TestHoverMetadataWithPair:
    """Tests for pair grid metadata."""

    def test_pair_grid_structure(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        meta = (
            Plot(df)
            .pair(x=["a", "b"], y=["a", "b"])
            .add(Dot())
            .hover_metadata()
        )

        assert len(meta["subplots"]) == 4

        x_vars = [s["x_var"] for s in meta["subplots"]]
        y_vars = [s["y_var"] for s in meta["subplots"]]
        assert set(x_vars) == {"x0", "x1"}
        assert set(y_vars) == {"y0", "y1"}

        for subplot in meta["subplots"]:
            assert len(subplot["layers"]) == 1
            x_vals = subplot["layers"][0]["variables"]["x"]["original_values"]
            y_vals = subplot["layers"][0]["variables"]["y"]["original_values"]
            assert len(x_vals) == 3
            assert len(y_vals) == 3

            if subplot["x_var"] == "x0":
                np.testing.assert_array_equal(x_vals, [1, 2, 3])
            else:
                np.testing.assert_array_equal(x_vals, [4, 5, 6])

            if subplot["y_var"] == "y0":
                np.testing.assert_array_equal(y_vals, [1, 2, 3])
            else:
                np.testing.assert_array_equal(y_vals, [4, 5, 6])

    def test_pair_coord_ranges(self):
        df = pd.DataFrame({"a": [1, 5], "b": [10, 50]})
        meta = (
            Plot(df)
            .pair(x=["a", "b"], y=["a", "b"])
            .add(Dot())
            .hover_metadata()
        )

        for subplot in meta["subplots"]:
            x_range = subplot["x_range"]
            y_range = subplot["y_range"]
            assert x_range is not None
            assert y_range is not None

            if subplot["x_var"] == "x0":
                assert x_range == (1.0, 5.0)
            else:
                assert x_range == (10.0, 50.0)

            if subplot["y_var"] == "y0":
                assert y_range == (1.0, 5.0)
            else:
                assert y_range == (10.0, 50.0)


class TestHoverMetadataWithMultiLayer:
    """Tests for multi-layer plot metadata."""

    def test_multi_layer_structure(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = (
            Plot(df, "x", "y")
            .add(Dot(), color="y")
            .add(Line(color="red"), label="trend")
            .hover_metadata()
        )

        assert len(meta["subplots"][0]["layers"]) == 2

        layer0 = meta["subplots"][0]["layers"][0]
        assert layer0["mark_type"] == "Dot"
        assert layer0["layer_index"] == 0
        assert layer0["layer_label"] is None
        assert "color" in layer0["variables"]

        layer1 = meta["subplots"][0]["layers"][1]
        assert layer1["mark_type"] == "Line"
        assert layer1["layer_index"] == 1
        assert layer1["layer_label"] == "trend"
        assert "color" not in layer1["variables"]

    def test_each_layer_has_own_scales(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y1": [1, 2, 3], "y2": [4, 5, 6]})
        meta = (
            Plot(df, "x")
            .add(Dot(), y="y1", color="y1")
            .add(Line(), y="y2", color="y2")
            .hover_metadata()
        )

        layer0 = meta["subplots"][0]["layers"][0]
        layer1 = meta["subplots"][0]["layers"][1]

        np.testing.assert_array_equal(
            layer0["variables"]["y"]["original_values"], [1, 2, 3]
        )
        np.testing.assert_array_equal(
            layer1["variables"]["y"]["original_values"], [4, 5, 6]
        )

        color0 = layer0["variables"]["color"]["original_values"]
        color1 = layer1["variables"]["color"]["original_values"]
        assert not np.array_equal(color0, color1)


class TestHoverMetadataScaleLabels:
    """Tests for scale label and legend consistency."""

    def test_nominal_legend_labels(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [1, 2, 3, 4],
            "c": ["alpha", "beta", "alpha", "beta"],
        })
        meta = (
            Plot(df, "x", "y")
            .add(Dot(), color="c")
            .scale(color=Nominal(order=["beta", "alpha"]))
            .hover_metadata()
        )

        color_var = meta["subplots"][0]["layers"][0]["variables"]["color"]
        assert color_var["legend_values"] == ["beta", "alpha"]
        assert color_var["display_labels"] == ["beta", "alpha"]

    def test_nominal_values_and_labels(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [1, 2, 3],
            "c": ["a", "b", "c"],
        })
        meta = (
            Plot(df, "x", "y")
            .add(Dot(), color="c")
            .scale(
                color=Nominal(
                    ["#ff0000", "#00ff00", "#0000ff"],
                    order=["a", "b", "c"],
                )
            )
            .hover_metadata()
        )

        color_var = meta["subplots"][0]["layers"][0]["variables"]["color"]
        assert color_var["scale_type"] == "Nominal"
        assert color_var["legend_values"] == ["a", "b", "c"]
        assert color_var["display_labels"] == ["a", "b", "c"]
        assert color_var["original_values"].tolist() == ["a", "b", "c"]
        scaled = color_var["scaled_values"]
        assert scaled.shape == (3, 4)
        assert scaled[0][0] == pytest.approx(1.0)
        assert scaled[0][1] == pytest.approx(0.0)
        assert scaled[0][2] == pytest.approx(0.0)

    def test_continuous_coord_range(self):
        df = pd.DataFrame({"x": [1, 2, 3, 10], "y": [0.5, 2.5, 1.5, 5.5]})
        meta = (
            Plot(df, "x", "y")
            .add(Dot())
            .scale(x=Continuous(trans="log10"))
            .hover_metadata()
        )

        x_var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        y_var = meta["subplots"][0]["layers"][0]["variables"]["y"]
        assert x_var["scale_type"] == "Continuous"
        assert y_var["scale_type"] == "Continuous"
        assert x_var["coord_range"] == (1.0, 10.0)
        assert y_var["coord_range"] == (0.5, 5.5)

    def test_scale_type_in_metadata(self):
        from seaborn._core.scales import Temporal, Boolean

        df = pd.DataFrame({
            "x": pd.date_range("2020-01-01", periods=5),
            "y": [1, 2, 3, 4, 5],
            "b": [True, False, True, False, True],
        })
        meta = (
            Plot(df, "x", "y")
            .add(Dot(), pointstyle="b")
            .scale(x=Temporal(), pointstyle=Boolean())
            .hover_metadata()
        )

        x_var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        b_var = meta["subplots"][0]["layers"][0]["variables"]["pointstyle"]
        assert x_var["scale_type"] == "Temporal"
        assert b_var["scale_type"] == "Boolean"


class TestHoverMetadataEdgeCases:
    """Tests for edge cases in hover metadata collection."""

    def test_single_point_data(self):
        df = pd.DataFrame({"x": [1], "y": [1]})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()
        x_var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        assert x_var["original_values"] == [1]
        assert x_var["coord_range"] == (1.0, 1.0)

    def test_nan_in_data(self):
        df = pd.DataFrame({"x": [1, np.nan, 3], "y": [1, 2, np.nan]})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()
        x_var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        assert x_var["coord_range"] == (1.0, 3.0)

    def test_empty_data(self):
        df = pd.DataFrame({"x": [], "y": []})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()
        assert len(meta["subplots"]) == 1
        assert len(meta["subplots"][0]["layers"]) == 0

    def test_layer_label_passthrough(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = (
            Plot(df, "x", "y")
            .add(Dot(), label="points_layer")
            .hover_metadata()
        )
        assert meta["subplots"][0]["layers"][0]["layer_label"] == "points_layer"
