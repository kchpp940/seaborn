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
