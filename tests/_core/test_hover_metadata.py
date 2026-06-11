import numpy as np
import pandas as pd
import pytest
import re

from seaborn._core.plot import Plot
from seaborn._core.scales import Nominal, Continuous
from seaborn._marks.dot import Dot
from seaborn._marks.line import Line
from seaborn._marks.bar import Bar, Bars
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

    def test_three_value_layers_present(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "c": ["a", "b", "a"]})
        meta = Plot(df, "x", "y").add(Dot(), color="c").hover_metadata()

        var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        assert "variable" in var
        assert "source_values" in var
        assert "stat_output_values" in var
        assert "scaled_values" in var
        assert "display_labels" in var
        assert "legend_values" in var
        assert "coord_range" in var
        assert "scale_type" in var
        assert "property_type" in var

        assert var["variable"] == "x"
        assert isinstance(var["source_values"], np.ndarray)
        assert isinstance(var["stat_output_values"], np.ndarray)
        assert isinstance(var["scaled_values"], np.ndarray)
        assert var["source_values"].shape == var["stat_output_values"].shape
        assert var["stat_output_values"].shape == var["scaled_values"].shape
        assert var["coord_range"] is not None
        assert isinstance(var["coord_range"], tuple)
        assert len(var["coord_range"]) == 2

    def test_no_stat_all_layers_equal(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()

        x_var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        np.testing.assert_array_equal(x_var["source_values"], x_var["stat_output_values"])

        y_var = meta["subplots"][0]["layers"][0]["variables"]["y"]
        np.testing.assert_array_equal(y_var["source_values"], y_var["stat_output_values"])


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
        np.testing.assert_array_equal(
            color1["stat_output_values"], color2["stat_output_values"]
        )
        np.testing.assert_array_equal(
            color1["scaled_values"], color2["scaled_values"]
        )
        np.testing.assert_array_equal(
            color1["source_values"], color2["source_values"]
        )

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
    """Tests for the three value layers with stat transforms."""

    def test_agg_source_values_are_raw_input(self):
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

        assert len(y_var["source_values"]) == 6
        np.testing.assert_array_equal(
            y_var["source_values"], [1, 3, 2, 4, 5, 7]
        )

    def test_agg_stat_output_values_are_aggregated(self):
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

        assert len(y_var["stat_output_values"]) == 3
        expected_means = np.array([2.0, 3.0, 6.0])
        np.testing.assert_array_almost_equal(
            y_var["stat_output_values"], expected_means
        )
        assert y_var["coord_range"] == (2.0, 6.0)

    def test_agg_source_and_stat_different_lengths(self):
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
        assert len(y_var["source_values"]) == 6
        assert len(y_var["stat_output_values"]) == 3
        assert len(y_var["source_values"]) != len(y_var["stat_output_values"])

    def test_agg_color_source_and_stat_same_for_grouping_var(self):
        df = pd.DataFrame({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "val": [1, 3, 2, 4, 5, 7],
            "grp": ["x", "x", "y", "y", "z", "z"],
        })
        meta = (
            Plot(df, "cat", "val", color="grp")
            .add(Bar(), Agg("mean"))
            .hover_metadata()
        )

        color_var = meta["subplots"][0]["layers"][0]["variables"]["color"]
        assert len(color_var["source_values"]) == 6
        assert len(color_var["stat_output_values"]) == 3
        unique_source = pd.unique(color_var["source_values"])
        unique_stat = pd.unique(color_var["stat_output_values"])
        assert set(unique_source) == set(unique_stat)

    def test_hist_stat_source_values(self):
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.randn(100)})
        meta = (
            Plot(df, "x")
            .add(Bar(), Hist(bins=10))
            .hover_metadata()
        )

        layer = meta["subplots"][0]["layers"][0]
        x_var = layer["variables"]["x"]
        y_var = layer["variables"]["y"]

        assert len(x_var["source_values"]) == 100
        assert len(x_var["stat_output_values"]) == 10
        assert len(y_var["source_values"]) == 0
        assert len(y_var["stat_output_values"]) == 10
        assert y_var["coord_range"] is not None
        assert y_var["coord_range"][0] >= 0
        assert sum(y_var["stat_output_values"]) == 100

    def test_no_stat_layers_identical(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()

        x_var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        np.testing.assert_array_equal(
            x_var["source_values"], x_var["stat_output_values"]
        )
        assert np.array_equal(x_var["source_values"], x_var["stat_output_values"])


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
            n_points = len(
                subplot["layers"][0]["variables"]["x"]["stat_output_values"]
            )
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
            n_points = len(
                subplot["layers"][0]["variables"]["x"]["stat_output_values"]
            )
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
                n_points = len(
                    subplot["layers"][0]["variables"]["x"]["stat_output_values"]
                )
                assert n_points == 1

    def test_facet_source_values_filtered(self):
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

        for subplot in meta["subplots"]:
            x_var = subplot["layers"][0]["variables"]["x"]
            assert len(x_var["source_values"]) == len(x_var["stat_output_values"])
            np.testing.assert_array_equal(
                x_var["source_values"], x_var["stat_output_values"]
            )


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
            x_vals = subplot["layers"][0]["variables"]["x"]["stat_output_values"]
            y_vals = subplot["layers"][0]["variables"]["y"]["stat_output_values"]
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

    def test_pair_with_agg_stat(self):
        df = pd.DataFrame({
            "a": [1, 2, 1, 2],
            "b": [10, 20, 30, 40],
            "cat": ["x", "x", "y", "y"],
        })
        meta = (
            Plot(df, color="cat")
            .pair(x=["a", "b"], y=["a", "b"])
            .add(Bar(), Agg("mean"))
            .hover_metadata()
        )

        assert len(meta["subplots"]) == 4
        for subplot in meta["subplots"]:
            for layer in subplot["layers"]:
                x_var = layer["variables"]["x"]
                assert "source_values" in x_var
                assert "stat_output_values" in x_var
                assert "scaled_values" in x_var


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
            layer0["variables"]["y"]["stat_output_values"], [1, 2, 3]
        )
        np.testing.assert_array_equal(
            layer1["variables"]["y"]["stat_output_values"], [4, 5, 6]
        )

        color0 = layer0["variables"]["color"]["stat_output_values"]
        color1 = layer1["variables"]["color"]["stat_output_values"]
        assert not np.array_equal(color0, color1)

    def test_mixed_stat_and_nonstat_layers(self):
        df = pd.DataFrame({
            "x": ["a", "a", "b", "b"],
            "y": [1, 3, 2, 4],
        })
        meta = (
            Plot(df, "x", "y")
            .add(Dot())
            .add(Bar(), Agg("mean"))
            .hover_metadata()
        )

        assert len(meta["subplots"][0]["layers"]) == 2

        dot_layer = meta["subplots"][0]["layers"][0]
        bar_layer = meta["subplots"][0]["layers"][1]

        assert dot_layer["mark_type"] == "Dot"
        assert len(dot_layer["variables"]["y"]["source_values"]) == 4
        assert len(dot_layer["variables"]["y"]["stat_output_values"]) == 4

        assert bar_layer["mark_type"] == "Bar"
        assert len(bar_layer["variables"]["y"]["source_values"]) == 4
        assert len(bar_layer["variables"]["y"]["stat_output_values"]) == 2


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
        assert color_var["stat_output_values"].tolist() == ["a", "b", "c"]
        scaled = color_var["scaled_values"]
        assert scaled.shape == (3, 3)
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
        assert x_var["coord_range"][0] == pytest.approx(0.0)
        assert x_var["coord_range"][1] == pytest.approx(1.0)
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
            .add(Dot(), marker="b")
            .scale(x=Temporal(), marker=Boolean())
            .hover_metadata()
        )

        x_var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        b_var = meta["subplots"][0]["layers"][0]["variables"]["marker"]
        assert x_var["scale_type"] == "Temporal"
        assert b_var["scale_type"] == "Boolean"


class TestHoverMetadataEdgeCases:
    """Tests for edge cases in hover metadata collection."""

    def test_single_point_data(self):
        df = pd.DataFrame({"x": [1], "y": [1]})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()
        x_var = meta["subplots"][0]["layers"][0]["variables"]["x"]
        assert x_var["source_values"] == [1]
        assert x_var["stat_output_values"] == [1]
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

    def test_scaled_values_match_actual_scale_output(self):
        from seaborn._core.scales import Nominal

        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "c": ["a", "b", "c"]})
        p = Plot(df, "x", "y").add(Dot(), color="c")

        plotter = p.plot()
        meta = p.hover_metadata()

        color_scale = plotter._scales["color"]
        color_var = meta["subplots"][0]["layers"][0]["variables"]["color"]

        expected_scaled = np.asarray(color_scale(df["c"]))
        np.testing.assert_array_almost_equal(
            color_var["scaled_values"], expected_scaled
        )


class TestHoverMetadataElementTracking:
    """Tests for element-level tracking (element_id, indices, draw_order, group_key)."""

    def test_elements_field_present(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]
        assert "elements" in layer
        assert isinstance(layer["elements"], list)
        assert len(layer["elements"]) > 0

    def test_element_structure(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = Plot(df, "x", "y").add(Bar()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]
        elem = layer["elements"][0]
        assert "element_id" in elem
        assert "draw_order" in elem
        assert "stat_index" in elem
        assert "source_index" in elem
        assert "group_key" in elem
        assert isinstance(elem["element_id"], str)
        assert elem["element_id"].startswith("L")
        assert isinstance(elem["draw_order"], int)
        assert isinstance(elem["stat_index"], list)
        assert elem["group_key"] is not None

    def test_dot_single_artist_for_all_points(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]
        assert len(layer["elements"]) == 1
        elem = layer["elements"][0]
        assert len(elem["stat_index"]) == 5
        assert elem["stat_index"] == [0, 1, 2, 3, 4]
        assert elem["source_index"] == [0, 1, 2, 3, 4]
        assert elem["draw_order"] == 0

    def test_bar_one_artist_per_bar(self):
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
        meta = Plot(df, "x", "y").add(Bar()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]
        assert len(layer["elements"]) == 3
        for i, elem in enumerate(layer["elements"]):
            assert elem["draw_order"] == i
            assert elem["stat_index"] == [i]
            assert elem["source_index"] == [i]

    def test_bars_one_artist_per_bar(self):
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
        meta = Plot(df, "x", "y").add(Bars()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]
        assert len(layer["elements"]) == 3
        for i, elem in enumerate(layer["elements"]):
            assert elem["draw_order"] == i
            assert elem["stat_index"] == [i]
            assert elem["source_index"] == [i]

    def test_hist_stat_elements(self):
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.randn(50)})
        meta = Plot(df, "x").add(Bars(), Hist(bins=5)).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]
        assert len(layer["elements"]) == 5
        for i, elem in enumerate(layer["elements"]):
            assert elem["draw_order"] == i
            assert elem["stat_index"] == [i]
            assert elem["source_index"] is None

    def test_agg_stat_elements(self):
        df = pd.DataFrame({
            "x": ["a", "a", "b", "b", "c", "c"],
            "y": [1, 3, 2, 4, 3, 5],
        })
        meta = Plot(df, "x", "y").add(Bar(), Agg("mean")).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]
        assert len(layer["elements"]) == 3
        for i, elem in enumerate(layer["elements"]):
            assert elem["draw_order"] == i
            assert elem["stat_index"] == [i]
            assert elem["source_index"] is None

    def test_facet_elements_separated(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1, 2, 3, 4, 5, 6],
            "g": ["a", "a", "a", "b", "b", "b"],
        })
        meta = Plot(df, "x", "y").facet(col="g").add(Bar()).hover_metadata()
        assert len(meta["subplots"]) == 2

        subplot_a = [s for s in meta["subplots"] if s["col"] == "a"][0]
        subplot_b = [s for s in meta["subplots"] if s["col"] == "b"][0]

        assert len(subplot_a["layers"][0]["elements"]) == 3
        assert len(subplot_b["layers"][0]["elements"]) == 3

        # stat_index contains pandas index values (global row labels)
        # For subplot "a", rows have index 0,1,2
        for i, elem in enumerate(subplot_a["layers"][0]["elements"]):
            assert elem["group_key"]["col"] == "a"
            assert elem["stat_index"] == [i]

        # For subplot "b", rows have index 3,4,5
        for i, elem in enumerate(subplot_b["layers"][0]["elements"]):
            assert elem["group_key"]["col"] == "b"
            assert elem["stat_index"] == [i + 3]
            # source_index maps back to original rows
            assert elem["source_index"] == [i + 3]

    def test_facet_empty_subset_no_elements(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [1, 2, 3, 4],
            "g": ["a", "a", "a", "a"],
        })
        meta = (
            Plot(df, "x", "y")
            .facet(col="g", order=["a", "b"])
            .add(Bar())
            .hover_metadata()
        )
        assert len(meta["subplots"]) == 2

        subplot_a = [s for s in meta["subplots"] if s["col"] == "a"][0]
        subplot_b = [s for s in meta["subplots"] if s["col"] == "b"][0]

        assert len(subplot_a["layers"][0]["elements"]) == 4
        assert len(subplot_b["layers"]) == 0

    def test_pair_elements(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        meta = Plot(df).pair(x=["a", "b"], y=["a", "b"]).add(Dot()).hover_metadata()
        assert len(meta["subplots"]) == 4

        for subplot in meta["subplots"]:
            assert len(subplot["layers"]) == 1
            layer = subplot["layers"][0]
            assert len(layer["elements"]) == 1
            elem = layer["elements"][0]
            assert len(elem["stat_index"]) == 3
            assert elem["source_index"] == [0, 1, 2]

    def test_multi_layer_elements_separated(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = (
            Plot(df, "x", "y")
            .add(Bar(), label="bars")
            .add(Dot(), label="dots")
            .hover_metadata()
        )
        layer = meta["subplots"][0]["layers"]
        assert len(layer) == 2

        bar_layer = [l for l in layer if l["layer_label"] == "bars"][0]
        dot_layer = [l for l in layer if l["layer_label"] == "dots"][0]

        assert len(bar_layer["elements"]) == 3
        assert len(dot_layer["elements"]) == 1

        for elem in bar_layer["elements"]:
            assert isinstance(elem["element_id"], str)
            assert elem["element_id"].startswith("L")

        for elem in dot_layer["elements"]:
            assert isinstance(elem["element_id"], str)
            assert elem["element_id"].startswith("L")

    def test_draw_order_is_sequential(self):
        df = pd.DataFrame({"x": ["a", "b", "c", "d"], "y": [1, 2, 3, 4]})
        meta = Plot(df, "x", "y").add(Bar()).hover_metadata()
        elements = meta["subplots"][0]["layers"][0]["elements"]
        draw_orders = [e["draw_order"] for e in elements]
        assert draw_orders == [0, 1, 2, 3]

    def test_group_key_contains_grouping_vars(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1, 2, 3, 4, 5, 6],
            "g": ["a", "a", "a", "b", "b", "b"],
        })
        from seaborn._marks.line import Line
        meta = (
            Plot(df, "x", "y")
            .add(Line(), color="g")
            .hover_metadata()
        )
        layer = meta["subplots"][0]["layers"][0]
        assert len(layer["elements"]) == 2

        group_keys = [e["group_key"] for e in layer["elements"]]
        color_values = sorted([gk.get("color") for gk in group_keys])
        assert color_values == ["a", "b"]

    def test_element_indices_match_variable_values(self):
        df = pd.DataFrame({"x": [10, 20, 30], "y": [100, 200, 300]})
        meta = Plot(df, "x", "y").add(Bar()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]
        x_var = layer["variables"]["x"]
        y_var = layer["variables"]["y"]

        for i, elem in enumerate(layer["elements"]):
            stat_idx = elem["stat_index"][0]
            assert x_var["stat_output_values"][stat_idx] == pytest.approx([10, 20, 30][i])
            assert y_var["stat_output_values"][stat_idx] == pytest.approx([100, 200, 300][i])
            assert x_var["source_values"][stat_idx] == [10, 20, 30][i]

    def test_line_mark_tracks_elements(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 4, 9, 16]})
        from seaborn._marks.line import Line
        meta = Plot(df, "x", "y").add(Line()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]
        assert len(layer["elements"]) == 1
        elem = layer["elements"][0]
        assert len(elem["stat_index"]) == 4
        assert elem["source_index"] == [0, 1, 2, 3]
        assert elem["draw_order"] == 0

    def test_multiple_group_keys(self):
        df = pd.DataFrame({
            "x": [1, 2, 1, 2],
            "y": [1, 2, 3, 4],
            "g": ["a", "a", "b", "b"],
        })
        from seaborn._marks.line import Line
        meta = (
            Plot(df, "x", "y")
            .add(Line(), color="g", linestyle="g")
            .hover_metadata()
        )
        layer = meta["subplots"][0]["layers"][0]
        assert len(layer["elements"]) == 2
        for elem in layer["elements"]:
            gk = elem["group_key"]
            assert "color" in gk
            assert "linestyle" in gk

    def test_artist_registry_structure(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = Plot(df, "x", "y").add(Dot()).hover_metadata()

        assert "artist_registry" in meta
        assert "element_registry" in meta
        assert isinstance(meta["artist_registry"], dict)
        assert isinstance(meta["element_registry"], dict)

        layer = meta["subplots"][0]["layers"][0]
        elem_id = layer["elements"][0]["element_id"]

        assert elem_id in meta["artist_registry"]
        artist_id = meta["artist_registry"][elem_id]
        assert isinstance(artist_id, int)
        assert artist_id in meta["element_registry"]
        assert meta["element_registry"][artist_id] == elem_id

    def test_element_id_format(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = Plot(df, "x", "y").add(Bar()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]

        for elem in layer["elements"]:
            elem_id = elem["element_id"]
            assert re.match(r"^L\d{3}-P\d{3}-E\d{3}$", elem_id), elem_id

    def test_lookup_artist(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        p = Plot(df, "x", "y").add(Bar())
        plotter = p.plot()
        meta = plotter._hover_metadata

        layer = meta["subplots"][0]["layers"][0]
        elem_id = layer["elements"][0]["element_id"]

        artist = plotter.lookup_artist(elem_id)
        assert artist is not None
        assert id(artist) == meta["artist_registry"][elem_id]

        assert plotter.lookup_artist("nonexistent") is None

    def test_lookup_element(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        p = Plot(df, "x", "y").add(Bar())
        plotter = p.plot()
        meta = plotter._hover_metadata

        layer = meta["subplots"][0]["layers"][0]
        elem_id = layer["elements"][0]["element_id"]
        artist = plotter.lookup_artist(elem_id)

        elem_meta = plotter.lookup_element(artist)
        assert elem_meta is not None
        assert elem_meta["element_id"] == elem_id
        assert elem_meta["stat_index"] == [0]

        class FakeArtist:
            pass
        assert plotter.lookup_element(FakeArtist()) is None

    def test_source_index_after_filtering(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1, 2, 3, 4, 5, 6],
            "g": ["a", "a", "a", "b", "b", "b"],
        })
        meta = Plot(df, "x", "y").facet(col="g").add(Bar()).hover_metadata()

        subplot_a = [s for s in meta["subplots"] if s["col"] == "a"][0]
        subplot_b = [s for s in meta["subplots"] if s["col"] == "b"][0]

        # source_index points to the original (global) row indices
        # For subplot "a", original rows are 0, 1, 2
        a_source_indices = sorted([e["source_index"][0] for e in subplot_a["layers"][0]["elements"]])
        assert a_source_indices == [0, 1, 2]

        # For subplot "b", original rows are 3, 4, 5
        b_source_indices = sorted([e["source_index"][0] for e in subplot_b["layers"][0]["elements"]])
        assert b_source_indices == [3, 4, 5]

        # source_values are per-subplot arrays (filtered), indexed by
        # local position (0-based). Use source_index to map back to
        # the global original data for verification.
        x_var_b = subplot_b["layers"][0]["variables"]["x"]
        b_x_vals = list(x_var_b["source_values"])
        assert b_x_vals == [4, 5, 6]

    def test_element_ids_unique_across_layers(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = (
            Plot(df, "x", "y")
            .add(Bar(), label="bars")
            .add(Dot(), label="dots")
            .hover_metadata()
        )

        all_ids = []
        for subplot in meta["subplots"]:
            for layer in subplot["layers"]:
                for elem in layer["elements"]:
                    all_ids.append(elem["element_id"])

        assert len(all_ids) == len(set(all_ids))

    def test_stable_element_ids(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        p = Plot(df, "x", "y").add(Bar())

        meta1 = p.hover_metadata()
        meta2 = p.hover_metadata()

        ids1 = [elem["element_id"] for elem in meta1["subplots"][0]["layers"][0]["elements"]]
        ids2 = [elem["element_id"] for elem in meta2["subplots"][0]["layers"][0]["elements"]]

        assert ids1 == ids2

    def test_source_index_with_non_contiguous_index(self):
        df = pd.DataFrame({
            "x": [10, 20, 30, 40],
            "y": [1, 2, 3, 4],
        }, index=[5, 10, 15, 20])
        meta = Plot(df, "x", "y").add(Bar()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]

        # stat_index contains pandas index values (5, 10, 15, 20)
        stat_indices = [elem["stat_index"][0] for elem in layer["elements"]]
        assert sorted(stat_indices) == [5, 10, 15, 20]

        # source_index should map to 0-based original row positions
        source_indices = [elem["source_index"][0] for elem in layer["elements"]]
        assert sorted(source_indices) == [0, 1, 2, 3]

    def test_source_index_with_sorted_data(self):
        df = pd.DataFrame({
            "x": [30, 10, 20],
            "y": [3, 1, 2],
        })
        meta = Plot(df, "x", "y").add(Bar()).hover_metadata()
        layer = meta["subplots"][0]["layers"][0]

        # stat_index follows pandas index (0, 1, 2 after default reset)
        # source_index maps to original 0-based positions
        for elem in layer["elements"]:
            assert isinstance(elem["source_index"][0], int)
            assert 0 <= elem["source_index"][0] <= 2

    def test_registry_serializable(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        meta = Plot(df, "x", "y").add(Bar()).hover_metadata()

        import json
        registry = meta["artist_registry"]
        serialized = json.dumps(registry)
        deserialized = json.loads(serialized)
        assert deserialized == registry
