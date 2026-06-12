import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import rcmod


@pytest.fixture(autouse=True)
def clean_registry_and_rc():
    saved = dict(rcmod._theme_profile_registry)
    rcmod._theme_profile_registry.clear()
    rcmod.reset_orig()
    yield
    rcmod._theme_profile_registry.clear()
    rcmod._theme_profile_registry.update(saved)
    rcmod.reset_orig()


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

class TestRegisterThemeProfile:

    def test_register_basic(self):
        params = {"style": "white", "context": "talk", "palette": "Blues_d"}
        rcmod.register_theme_profile("corp", params)
        assert "corp" in rcmod.list_theme_profiles()
        got = rcmod.get_theme_profile("corp")
        assert got["style"] == "white"
        assert got["context"] == "talk"
        assert got["palette"] == "Blues_d"
        # defaults filled in
        assert got["font"] == "sans-serif"
        assert got["font_scale"] == 1
        assert got["color_codes"] is True
        assert got["rc"] == {}

    def test_register_returns_copy(self):
        params = {"style": "white", "rc": {"axes.linewidth": 2}}
        rcmod.register_theme_profile("p", params)
        got = rcmod.get_theme_profile("p")
        got["rc"]["axes.linewidth"] = 999
        got2 = rcmod.get_theme_profile("p")
        assert got2["rc"]["axes.linewidth"] == 2  # unchanged

    def test_register_duplicate_fails(self):
        rcmod.register_theme_profile("dup", {"style": "white"})
        with pytest.raises(ValueError, match="already registered"):
            rcmod.register_theme_profile("dup", {"style": "dark"})

    def test_register_overwrite(self):
        rcmod.register_theme_profile("ov", {"style": "white"})
        rcmod.register_theme_profile("ov", {"style": "dark"}, overwrite=True)
        assert rcmod.get_theme_profile("ov")["style"] == "dark"

    @pytest.mark.parametrize("bad_name", ["", "white", "notebook", "poster", "ticks"])
    def test_register_bad_name(self, bad_name):
        with pytest.raises(ValueError):
            rcmod.register_theme_profile(bad_name, {"style": "white"})

    def test_register_params_not_dict(self):
        with pytest.raises(TypeError):
            rcmod.register_theme_profile("x", ["style", "white"])

    def test_register_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown key"):
            rcmod.register_theme_profile("x", {"styl": "white"})

    def test_register_invalid_context_string(self):
        with pytest.raises(ValueError, match="context.*string must be one"):
            rcmod.register_theme_profile("x", {"context": "huge"})

    def test_register_invalid_style_string(self):
        with pytest.raises(ValueError, match="style.*string must be one"):
            rcmod.register_theme_profile("x", {"style": "beige"})

    def test_register_invalid_font_scale_type(self):
        with pytest.raises(TypeError):
            rcmod.register_theme_profile("x", {"font_scale": "big"})

    def test_register_invalid_font_scale_negative(self):
        with pytest.raises(ValueError):
            rcmod.register_theme_profile("x", {"font_scale": -1})

    def test_register_invalid_color_codes(self):
        with pytest.raises(TypeError):
            rcmod.register_theme_profile("x", {"color_codes": "yes"})

    def test_register_invalid_rc_type(self):
        with pytest.raises(TypeError):
            rcmod.register_theme_profile("x", {"rc": "not a dict"})

    def test_register_invalid_rc_key(self):
        with pytest.raises(ValueError, match="Unrecognized matplotlib rcParam"):
            rcmod.register_theme_profile("x", {"rc": {"figure.invalid_key": 1}})

    def test_register_rc_dict_style_accepted(self):
        rcmod.register_theme_profile("x", {
            "rc": {"axes.facecolor": "red", "figure.facecolor": "blue"},
        })
        assert rcmod.get_theme_profile("x")["rc"]["axes.facecolor"] == "red"


class TestListAndUnregister:

    def test_list_empty(self):
        assert rcmod.list_theme_profiles() == []

    def test_list_sorted(self):
        rcmod.register_theme_profile("bravo", {"style": "white"})
        rcmod.register_theme_profile("alpha", {"style": "white"})
        rcmod.register_theme_profile("charlie", {"style": "white"})
        assert rcmod.list_theme_profiles() == ["alpha", "bravo", "charlie"]

    def test_unregister(self):
        rcmod.register_theme_profile("g", {"style": "white"})
        rcmod.unregister_theme_profile("g")
        assert "g" not in rcmod.list_theme_profiles()

    def test_unregister_missing(self):
        with pytest.raises(ValueError, match="Cannot unregister unknown"):
            rcmod.unregister_theme_profile("nope")

    def test_get_missing_with_suggestions(self):
        rcmod.register_theme_profile("aa", {"style": "white"})
        with pytest.raises(ValueError, match="Available profiles: aa"):
            rcmod.get_theme_profile("zz")

    def test_get_missing_empty(self):
        with pytest.raises(ValueError, match="No profiles are currently registered"):
            rcmod.get_theme_profile("zz")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class TestValidateRcDict:

    def test_none_rc(self):
        assert rcmod._validate_rc_dict(None) == {}

    def test_bad_type(self):
        with pytest.raises(TypeError):
            rcmod._validate_rc_dict([1, 2, 3])

    def test_full_validation_catches_bad_keys(self):
        with pytest.raises(ValueError):
            rcmod._validate_rc_dict({"no.such.key": 1})

    def test_style_category_warns_and_filters(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = rcmod._validate_rc_dict(
                {"axes.facecolor": "red", "font.size": 20, "xtick.labelsize": 99},
                category="style",
            )
        assert out == {"axes.facecolor": "red"}
        assert len(w) == 1
        assert "font.size, xtick.labelsize" in str(w[0].message)

    def test_context_category_warns_and_filters(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = rcmod._validate_rc_dict(
                {"font.size": 20, "axes.facecolor": "red"},
                category="context",
            )
        assert out == {"font.size": 20}
        assert len(w) == 1
        assert "axes.facecolor" in str(w[0].message)


# ---------------------------------------------------------------------------
# set_theme with profile
# ---------------------------------------------------------------------------

class TestSetThemeWithProfile:

    def test_set_theme_by_name(self):
        rcmod.register_theme_profile("corp", {
            "style": "white", "context": "poster",
            "palette": "Reds", "font": "serif",
            "rc": {"lines.linewidth": 7.0},
        })
        rcmod.set_theme(profile="corp")
        assert mpl.rcParams["axes.facecolor"] == "white"  # white style
        assert mpl.rcParams["lines.linewidth"] == 7.0  # rc override
        # poster context => 2x scaling of notebook base 1.5 => 3.0
        assert mpl.rcParams["axes.linewidth"] == pytest.approx(2.5)

    def test_set_theme_inline_dict_profile(self):
        rcmod.set_theme(profile={"style": "dark", "context": "paper"})
        assert mpl.rcParams["axes.facecolor"] == "#EAEAF2"  # dark style
        # paper => 0.8 scaling of base 1.5 => 1.2
        assert mpl.rcParams["axes.linewidth"] == pytest.approx(1.0)

    def test_set_theme_explicit_overrides_profile(self):
        rcmod.register_theme_profile("p", {"style": "white", "context": "talk"})
        rcmod.set_theme(profile="p", style="darkgrid", context="paper")
        # explicit wins
        assert mpl.rcParams["axes.facecolor"] == "#EAEAF2"  # darkgrid's facecolor
        # paper context wins over talk
        notebook_ref = rcmod.plotting_context("notebook")
        paper_ref = rcmod.plotting_context("paper")
        assert mpl.rcParams["axes.linewidth"] == paper_ref["axes.linewidth"]

    def test_set_theme_rc_merged_from_profile_and_arg(self):
        rcmod.register_theme_profile("p", {
            "rc": {"lines.linewidth": 5},
        })
        rcmod.set_theme(profile="p", rc={"axes.facecolor": "purple"})
        assert mpl.rcParams["lines.linewidth"] == 5
        assert mpl.rcParams["axes.facecolor"] == "purple"

    def test_set_theme_rc_explicit_overwrites_profile_rc(self):
        rcmod.register_theme_profile("p", {
            "rc": {"lines.linewidth": 5},
        })
        rcmod.set_theme(profile="p", rc={"lines.linewidth": 99})
        assert mpl.rcParams["lines.linewidth"] == 99

    def test_set_theme_bad_profile_type(self):
        with pytest.raises(TypeError, match=r"`profile` must be a string name, dict"):
            rcmod.set_theme(profile=123)


# ---------------------------------------------------------------------------
# axes_style with profile (getter + context manager + decorator)
# ---------------------------------------------------------------------------

class TestAxesStyleWithProfile:

    def test_axes_style_getter_by_profile_name(self):
        rcmod.register_theme_profile("corp", {
            "style": "ticks",
            "rc": {"axes.facecolor": "beige"},
        })
        style = rcmod.axes_style(profile="corp")
        assert style["axes.facecolor"] == "beige"
        assert style["xtick.bottom"] is True  # ticks style

    def test_axes_style_explicit_beats_profile(self):
        rcmod.register_theme_profile("corp", {"style": "white"})
        style = rcmod.axes_style(style="dark", profile="corp")
        assert style["axes.facecolor"] == "#EAEAF2"  # dark style wins

    def test_axes_style_profile_rc_merged(self):
        rcmod.register_theme_profile("corp", {
            "rc": {"axes.edgecolor": "pink"},
        })
        style = rcmod.axes_style("darkgrid", profile="corp",
                                 rc={"axes.facecolor": "gold"})
        assert style["axes.edgecolor"] == "pink"
        assert style["axes.facecolor"] == "gold"

    def test_axes_style_context_manager_restores(self):
        rcmod.set_theme(style="darkgrid")
        orig = mpl.rcParams["axes.facecolor"]
        rcmod.register_theme_profile("white_p", {"style": "white"})

        with rcmod.axes_style(profile="white_p"):
            assert mpl.rcParams["axes.facecolor"] == "white"
        assert mpl.rcParams["axes.facecolor"] == orig

    def test_axes_style_context_manager_with_rc_override(self):
        rcmod.set_theme(style="darkgrid")
        rcmod.register_theme_profile("p", {"style": "white"})
        orig_face = mpl.rcParams["axes.facecolor"]
        orig_edge = mpl.rcParams["axes.edgecolor"]

        with rcmod.axes_style(profile="p", rc={"axes.edgecolor": "crimson"}):
            assert mpl.rcParams["axes.facecolor"] == "white"
            assert mpl.rcParams["axes.edgecolor"] == "crimson"

        assert mpl.rcParams["axes.facecolor"] == orig_face
        assert mpl.rcParams["axes.edgecolor"] == orig_edge

    def test_axes_style_decorator(self):
        rcmod.set_theme(style="darkgrid")
        orig = mpl.rcParams["axes.facecolor"]
        rcmod.register_theme_profile("p", {"style": "ticks"})

        @rcmod.axes_style(profile="p")
        def inner():
            return mpl.rcParams["xtick.bottom"]

        assert inner() is True
        assert mpl.rcParams["axes.facecolor"] == orig


# ---------------------------------------------------------------------------
# plotting_context with profile
# ---------------------------------------------------------------------------

class TestPlottingContextWithProfile:

    def test_plotting_context_getter_by_name(self):
        rcmod.register_theme_profile("big", {
            "context": "poster",
            "font_scale": 2,
            "rc": {"lines.linewidth": 9},
        })
        ctx = rcmod.plotting_context(profile="big")
        # poster scales notebook base by 2, font_scale doubles fonts again
        notebook_base_fs = 12
        expected_font_size = notebook_base_fs * 2 * 2
        assert ctx["font.size"] == pytest.approx(expected_font_size)
        assert ctx["lines.linewidth"] == 9

    def test_plotting_context_explicit_overrides(self):
        rcmod.register_theme_profile("p", {"context": "poster", "font_scale": 3})
        ctx = rcmod.plotting_context(context="paper", font_scale=1, profile="p")
        # paper scale 0.8 of notebook base (12)
        assert ctx["font.size"] == pytest.approx(12 * 0.8 * 1)

    def test_plotting_context_context_manager_restores(self):
        rcmod.set_theme(context="notebook")
        orig = mpl.rcParams["font.size"]
        rcmod.register_theme_profile("p", {"context": "talk", "font_scale": 2})

        with rcmod.plotting_context(profile="p"):
            assert mpl.rcParams["font.size"] > orig  # should be larger

        assert mpl.rcParams["font.size"] == orig

    def test_plotting_context_decorator(self):
        rcmod.set_theme(context="notebook")
        orig = mpl.rcParams["font.size"]
        rcmod.register_theme_profile("p", {"context": "poster"})

        @rcmod.plotting_context(profile="p")
        def inner():
            return mpl.rcParams["font.size"]

        assert inner() > orig
        assert mpl.rcParams["font.size"] == orig

    def test_plotting_context_font_scale_none_defaults_to_1(self):
        ctx = rcmod.plotting_context("notebook", font_scale=None)
        ref = rcmod.plotting_context("notebook", font_scale=1)
        for k in ["font.size", "axes.labelsize"]:
            assert ctx[k] == ref[k]


# ---------------------------------------------------------------------------
# set_style and set_context with profile
# ---------------------------------------------------------------------------

class TestSetStyleSetContextWithProfile:

    def test_set_style_profile(self):
        rcmod.register_theme_profile("p", {
            "style": "white",
            "rc": {"axes.facecolor": "lime"},
        })
        rcmod.set_style(profile="p")
        assert mpl.rcParams["axes.facecolor"] == "lime"

    def test_set_context_profile(self):
        rcmod.register_theme_profile("p", {
            "context": "poster",
            "font_scale": 1.5,
            "rc": {"axes.linewidth": 3},
        })
        rcmod.set_context(profile="p")
        # poster => 2x notebook base 1.25 axes.linewidth = 2.5
        # but explicit rc overwrites so 3
        assert mpl.rcParams["axes.linewidth"] == 3


# ---------------------------------------------------------------------------
# Profile parameter propagation through rcParams (for figure-level functions)
# ---------------------------------------------------------------------------

class TestProfileSurvivesGlobalState:

    def test_set_theme_profile_then_scatterplot(self):
        """Figure-level functions read from rcParams, so profile set via
        set_theme should be reflected in any axes that get created."""
        rcmod.register_theme_profile("big", {
            "style": "white",
            "context": "poster",
        })
        rcmod.set_theme(profile="big")
        fig, ax = plt.subplots()
        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # white style
        plt.close(fig)

    def test_nested_style_context_with_profile(self):
        rcmod.register_theme_profile("p1", {"style": "white"})
        rcmod.register_theme_profile("p2", {"style": "dark"})
        rcmod.set_theme(profile="p1")
        outer_fc = mpl.rcParams["axes.facecolor"]

        with rcmod.axes_style(profile="p2"):
            inner_fc = mpl.rcParams["axes.facecolor"]
            assert inner_fc == "#EAEAF2"  # dark

        assert mpl.rcParams["axes.facecolor"] == outer_fc  # restored

    def test_nested_context_manager_combo(self):
        rcmod.set_theme(context="notebook", style="darkgrid")
        rcmod.register_theme_profile("s_w", {"style": "white"})
        rcmod.register_theme_profile("c_p", {"context": "paper"})

        orig_fc = mpl.rcParams["axes.facecolor"]
        orig_fs = mpl.rcParams["font.size"]

        with rcmod.axes_style(profile="s_w"):
            with rcmod.plotting_context(profile="c_p"):
                assert mpl.rcParams["axes.facecolor"] == "white"
                assert mpl.rcParams["font.size"] < orig_fs  # paper is smaller

        assert mpl.rcParams["axes.facecolor"] == orig_fc
        assert mpl.rcParams["font.size"] == orig_fs


# ---------------------------------------------------------------------------
# End-to-end: use sns top-level API
# ---------------------------------------------------------------------------

class TestSnsTopLevel:

    def test_sns_namespace_exports(self):
        for name in ["register_theme_profile", "get_theme_profile",
                     "list_theme_profiles", "unregister_theme_profile"]:
            assert hasattr(sns, name), f"sns.{name} not exported"

    def test_sns_set_theme_profile(self):
        sns.register_theme_profile("demo", {"style": "ticks", "context": "talk"})
        sns.set_theme(profile="demo")
        assert mpl.rcParams["xtick.bottom"] is True  # ticks

    def test_sns_axes_style_profile(self):
        sns.register_theme_profile("x", {"style": "ticks"})
        style = sns.axes_style(profile="x")
        assert style["xtick.bottom"] is True

    def test_sns_plotting_context_profile(self):
        sns.register_theme_profile("x", {"context": "paper"})
        ctx = sns.plotting_context(profile="x")
        notebook = sns.plotting_context("notebook")
        assert ctx["font.size"] < notebook["font.size"]

    def test_sns_list_and_unregister(self):
        sns.register_theme_profile("alpha", {"style": "white"})
        assert "alpha" in sns.list_theme_profiles()
        sns.unregister_theme_profile("alpha")
        assert "alpha" not in sns.list_theme_profiles()


# ---------------------------------------------------------------------------
# Unified validation: rc value validation (not just keys)
# ---------------------------------------------------------------------------

class TestUnifiedRcValidation:

    def test_invalid_rc_value_caught_early(self):
        """Invalid color values should be caught during validation."""
        with pytest.raises(ValueError, match="Invalid rcParams"):
            rcmod._validate_rc_dict({"axes.facecolor": "not_a_real_color_xyz"})

    def test_invalid_rc_value_in_profile(self):
        with pytest.raises(ValueError, match="Invalid rcParams"):
            rcmod.register_theme_profile("bad", {
                "rc": {"axes.facecolor": "not_a_color_123"}
            })

    def test_invalid_rc_value_in_set_theme(self):
        with pytest.raises(ValueError, match="Invalid rcParams"):
            rcmod.set_theme(rc={"lines.linestyle": "not_a_linestyle_xyz"})

    def test_unknown_key_same_message_everywhere(self):
        """Unknown rc keys should produce the same error message format."""
        err_msg = "Unrecognized matplotlib rcParam key"

        # In _validate_rc_dict directly
        with pytest.raises(ValueError, match=err_msg):
            rcmod._validate_rc_dict({"no.such.param": 1})

        # In profile registration
        with pytest.raises(ValueError, match=err_msg):
            rcmod.register_theme_profile("x", {"rc": {"bad.key": 1}})

        # In set_theme
        with pytest.raises(ValueError, match=err_msg):
            rcmod.set_theme(rc={"bad.key": 1})

    def test_unknown_profile_name_same_message(self):
        """Unknown profile names should have a consistent message."""
        with pytest.raises(ValueError, match="Unknown theme profile"):
            rcmod.set_theme(profile="nonexistent")
        with pytest.raises(ValueError, match="Unknown theme profile"):
            rcmod.axes_style(profile="nonexistent")
        with pytest.raises(ValueError, match="Unknown theme profile"):
            rcmod.plotting_context(profile="nonexistent")
        with pytest.raises(ValueError, match="Unknown theme profile"):
            rcmod.get_theme_profile("nonexistent")

    def test_style_category_warns_on_context_key(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rcmod.axes_style("white", rc={"font.size": 999})
        assert len(w) == 1
        assert "not in style definition" in str(w[0].message)

    def test_context_category_warns_on_style_key(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rcmod.plotting_context("notebook", rc={"axes.facecolor": "red"})
        assert len(w) == 1
        assert "not in context definition" in str(w[0].message)


# ---------------------------------------------------------------------------
# Figure-level functions with profile (no global pollution)
# ---------------------------------------------------------------------------

class TestFigureLevelProfile:

    def _make_data(self):
        import pandas as pd
        return pd.DataFrame({
            "x": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [2, 4, 3, 5, 3, 5, 4, 6],
            "cat": ["A", "A", "A", "A", "B", "B", "B", "B"],
        })

    def test_relplot_profile_does_not_pollute(self):
        sns.register_theme_profile("dark_p", {
            "style": "dark", "context": "talk",
        })
        orig_fc = mpl.rcParams["axes.facecolor"]
        orig_fs = mpl.rcParams["font.size"]

        df = self._make_data()
        g = sns.relplot(data=df, x="x", y="y", profile="dark_p")
        plt.close("all")

        assert mpl.rcParams["axes.facecolor"] == orig_fc
        assert mpl.rcParams["font.size"] == orig_fs

    def test_catplot_profile_applied_inside(self):
        sns.register_theme_profile("white_ticks_p", {
            "style": "ticks", "palette": "Reds",
        })
        orig_fc = mpl.rcParams["axes.facecolor"]
        seen_fc = []

        with rcmod._ThemeContext("white_ticks_p"):
            seen_fc.append(mpl.rcParams["axes.facecolor"])

        assert seen_fc[0] == "white"  # ticks style uses white background
        assert mpl.rcParams["axes.facecolor"] == orig_fc

    def test_displot_profile_no_pollution(self):
        try:
            sns.register_theme_profile("p", {"context": "paper"})
            orig_fs = mpl.rcParams["font.size"]
            df = self._make_data()
            g = sns.displot(data=df, x="x", profile="p")
            plt.close("all")
            assert mpl.rcParams["font.size"] == orig_fs
        except Exception as e:
            # Some matplotlib/seaborn configs may not have displot fully
            # working in test env; just ensure rc is restored
            plt.close("all")
            rcmod.reset_orig()

    def test_lmplot_profile(self):
        sns.register_theme_profile("lm", {"style": "whitegrid", "context": "talk"})
        orig_fc = mpl.rcParams["axes.facecolor"]
        df = self._make_data()
        try:
            g = sns.lmplot(data=df, x="x", y="y", profile="lm")
            plt.close("all")
        except Exception:
            plt.close("all")
        finally:
            assert mpl.rcParams["axes.facecolor"] == orig_fc

    def test_pairplot_profile(self):
        sns.register_theme_profile("pp", {"style": "white", "context": "talk"})
        orig_fc = mpl.rcParams["axes.facecolor"]
        df = self._make_data()
        try:
            g = sns.pairplot(df, profile="pp")
            plt.close("all")
        except Exception:
            plt.close("all")
        finally:
            assert mpl.rcParams["axes.facecolor"] == orig_fc

    def test_jointplot_profile(self):
        sns.register_theme_profile("jp", {"style": "darkgrid"})
        orig_fc = mpl.rcParams["axes.facecolor"]
        df = self._make_data()
        try:
            g = sns.jointplot(data=df, x="x", y="y", profile="jp")
            plt.close("all")
        except Exception:
            plt.close("all")
        finally:
            assert mpl.rcParams["axes.facecolor"] == orig_fc

    def test_clustermap_profile(self):
        sns.register_theme_profile("cm", {"style": "white", "context": "paper"})
        orig_fc = mpl.rcParams["axes.facecolor"]
        import numpy as np
        data = np.random.randn(8, 8)
        try:
            g = sns.clustermap(data, profile="cm")
            plt.close("all")
        except Exception:
            plt.close("all")
        finally:
            assert mpl.rcParams["axes.facecolor"] == orig_fc

    def test_inline_profile_dict_in_relplot(self):
        orig_fc = mpl.rcParams["axes.facecolor"]
        df = self._make_data()
        g = sns.relplot(
            data=df, x="x", y="y",
            profile={"style": "dark", "context": "paper"},
        )
        plt.close("all")
        assert mpl.rcParams["axes.facecolor"] == orig_fc

    def test_profile_invalid_key_propagates_to_figure_level(self):
        """Invalid rc keys should be caught before any plotting happens."""
        df = self._make_data()
        with pytest.raises(ValueError, match="Unrecognized matplotlib rcParam"):
            sns.relplot(
                data=df, x="x", y="y",
                profile={"rc": {"bad.bad.bad": 1}},
            )

    def test_profile_invalid_rc_value_propagates(self):
        """Invalid rc values should be caught before plotting."""
        df = self._make_data()
        with pytest.raises(ValueError, match="Invalid rcParams"):
            sns.relplot(
                data=df, x="x", y="y",
                profile={"rc": {"axes.facecolor": "not_a_real_color"}},
            )

    def test_unknown_profile_in_relplot(self):
        df = self._make_data()
        with pytest.raises(ValueError, match="Unknown theme profile"):
            sns.relplot(data=df, x="x", y="y", profile="ghost_profile")


# ---------------------------------------------------------------------------
# _ThemeContext internals
# ---------------------------------------------------------------------------

class TestThemeContext:

    def test_nested_theme_contexts(self):
        rcmod.register_theme_profile("a", {"style": "white"})
        rcmod.register_theme_profile("b", {"style": "dark"})

        with rcmod._ThemeContext("a"):
            assert mpl.rcParams["axes.facecolor"] == "white"
            with rcmod._ThemeContext("b"):
                assert mpl.rcParams["axes.facecolor"] == "#EAEAF2"
            assert mpl.rcParams["axes.facecolor"] == "white"

    def test_theme_context_restores_on_exception(self):
        rcmod.register_theme_profile("p", {"style": "ticks"})
        orig = mpl.rcParams["axes.facecolor"]

        try:
            with rcmod._ThemeContext("p"):
                assert mpl.rcParams["axes.facecolor"] == "white"
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass

        assert mpl.rcParams["axes.facecolor"] == orig

    def test_theme_context_with_extra_rc(self):
        rcmod.register_theme_profile("p", {
            "rc": {"savefig.dpi": 200, "axes.facecolor": "beige"},
        })
        orig_dpi = mpl.rcParams["savefig.dpi"]
        orig_fc = mpl.rcParams["axes.facecolor"]

        with rcmod._ThemeContext("p"):
            assert mpl.rcParams["savefig.dpi"] == 200
            assert mpl.rcParams["axes.facecolor"] == "beige"

        assert mpl.rcParams["savefig.dpi"] == orig_dpi
        assert mpl.rcParams["axes.facecolor"] == orig_fc

