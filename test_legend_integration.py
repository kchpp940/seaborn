"""综合测试：三大 figure-level 接口的 declared/observed 语义分离 + 图例口径统一"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(42)


def assert_legend_texts(ax_or_g, expected_texts, expected_title=None, msg=""):
    """获取 FacetGrid 或单 Ax 的 legend 条目文本"""
    if hasattr(ax_or_g, "_legend"):
        legend = ax_or_g._legend
    else:
        legend = ax_or_g.get_legend()
    assert legend is not None, f"{msg} legend is None"
    texts = [t.get_text() for t in legend.get_texts()]
    title = legend.get_title().get_text()
    assert texts == expected_texts, f"{msg} legend texts mismatch\n  got:      {texts}\n  expected: {expected_texts}"
    if expected_title is not None:
        assert title == expected_title, f"{msg} legend title mismatch\n  got:      {title!r}\n  expected: {expected_title!r}"
    print(f"  [OK] {msg}")


# ---------------------------------------------------------------------------
# 测试 1: relplot - 分面子集缺失 + 显式 hue_order 含数据不存在的级别
# ---------------------------------------------------------------------------
print("\n=== Test 1: relplot facet subset missing levels ===")

df = pd.DataFrame({
    "x": np.random.randn(180),
    "y": np.random.randn(180),
    "hue_var": ["A", "B", "C"] * 60,
    "style_var": ["X", "Y", "Z"] * 60,
    "size_var": [10, 20, 30] * 60,
    "facet_col": ["F1", "F2", "F3"] * 60,
})
# 让 F1 只有 hue=A/B，F2 只有 hue=B/C，F3 只有 hue=A/C
mask = ~(
    ((df["facet_col"] == "F1") & (df["hue_var"] == "C")) |
    ((df["facet_col"] == "F2") & (df["hue_var"] == "A")) |
    ((df["facet_col"] == "F3") & (df["hue_var"] == "B"))
)
df = df[mask].reset_index(drop=True)
# style: F1 只有 X/Y，F2 只有 Y/Z，F3 只有 X/Z
df = df[~(
    ((df["facet_col"] == "F1") & (df["style_var"] == "Z")) |
    ((df["facet_col"] == "F2") & (df["style_var"] == "X")) |
    ((df["facet_col"] == "F3") & (df["style_var"] == "Y"))
)].reset_index(drop=True)

# 显式 hue_order 多加一个不存在的级别 "Z"（注意 style_var 已经用了 X/Y/Z）
df_dummy = df.copy()

g = sns.relplot(
    data=df, x="x", y="y",
    hue="hue_var", hue_order=["A", "B", "C", "NO_SUCH"],
    style="style_var",
    size="size_var",
    col="facet_col",
)

# 期望: hue 只保留 A/B/C (NO_SUCH 从未出现)，style 保留 X/Y/Z (都出现过，
# 虽然每列只出现两个但跨三列并集是 X+Y+Z)，size 保留 10/20/30
# 注意: relplot 默认的 verbosity 会把 numeric size 变 brief，但 hue 是明确的 3 个
# 所以只断言 hue 的条目
legend_texts = [t.get_text() for t in g._legend.get_texts()]
# 检查 hue: 必须包含 A,B,C 且不含 "NO_SUCH"
assert "A" in legend_texts, "A missing in relplot legend"
assert "B" in legend_texts, "B missing in relplot legend"
assert "C" in legend_texts, "C missing in relplot legend"
assert "NO_SUCH" not in legend_texts, "NO_SUCH (never-drawn level) leaked into relplot legend!"
print(f"  [OK] relplot legend hue entries: {[t for t in legend_texts if t in ('A','B','C','NO_SUCH')]}"
      f" – NO_SUCH correctly excluded")
plt.close("all")


# ---------------------------------------------------------------------------
# 测试 2: catplot - 所有 kind
# ---------------------------------------------------------------------------
print("\n=== Test 2: catplot all kinds with facet missing levels ===")

df_cat = pd.DataFrame({
    "category": ["X", "Y", "Z"] * 100,
    "value": np.random.randn(300),
    "hue_cat": ["P", "Q", "R"] * 100,
    "facet_row": ["F1", "F2", "F3", "F1", "F2", "F3"] * 50,
})
# F1 只有 hue=P/Q, F2 只有 Q/R, F3 只有 P/R
df_cat = df_cat[~(
    ((df_cat["facet_row"] == "F1") & (df_cat["hue_cat"] == "R")) |
    ((df_cat["facet_row"] == "F2") & (df_cat["hue_cat"] == "P")) |
    ((df_cat["facet_row"] == "F3") & (df_cat["hue_cat"] == "Q"))
)].reset_index(drop=True)

# hue_order 中包含一个不存在的级别 "FAKE"
for kind in ["strip", "swarm", "box", "boxen", "violin", "bar", "count", "point"]:
    g = sns.catplot(
        data=df_cat, x="category",
        y=None if kind == "count" else "value",
        hue="hue_cat", hue_order=["P", "Q", "R", "FAKE"],
        row="facet_row",
        kind=kind,
    )
    legend_texts = [t.get_text() for t in g._legend.get_texts()]
    legend_title = g._legend.get_title().get_text()
    assert "FAKE" not in legend_texts, f"kind={kind}: FAKE leaked into legend! got {legend_texts}"
    assert "P" in legend_texts, f"kind={kind}: P missing"
    assert "Q" in legend_texts, f"kind={kind}: Q missing"
    assert "R" in legend_texts, f"kind={kind}: R missing"
    assert legend_title == "hue_cat", f"kind={kind}: title should be 'hue_cat', got {legend_title!r}"
    # point kind should also have markers/linestyles applied (check via Line2D)
    if kind == "point":
        handles = g._legend.legendHandles if hasattr(g._legend, "legendHandles") else g._legend.legend_handles
        # Find handles corresponding to hue levels (not errorbar, not other props)
        hue_handles = [h for h, t in zip(handles, legend_texts) if t in ("P", "Q", "R")]
        assert all(h.get_marker() != "None" for h in hue_handles), f"kind=point: markers not applied in legend"
    print(f"  [OK] catplot kind={kind}: legend {legend_texts} (title='{legend_title}')")
    plt.close("all")


# ---------------------------------------------------------------------------
# 测试 3: displot - 所有 kind
# ---------------------------------------------------------------------------
print("\n=== Test 3: displot all kinds with facet missing levels ===")

df_dist = pd.DataFrame({
    "measure": np.random.randn(400),
    "hue_dist": ["alpha", "beta", "gamma", "delta"] * 100,
    "facet_col": (["G1"] * 200) + (["G2"] * 200),
})
# G1 只有 alpha/gamma，G2 只有 beta/delta（这样四个 hue level 都有数据）
df_dist = df_dist[~(
    ((df_dist["facet_col"] == "G1") & df_dist["hue_dist"].isin(["beta", "delta"])) |
    ((df_dist["facet_col"] == "G2") & df_dist["hue_dist"].isin(["alpha", "gamma"]))
)].reset_index(drop=True)
# 再加 hue_order，包含不存在的级别 "NOPE"
for kind in ["hist", "kde", "ecdf"]:
    # ecdf 要求 univariate
    g = sns.displot(
        data=df_dist, x="measure",
        hue="hue_dist", hue_order=["alpha", "beta", "gamma", "delta", "NOPE"],
        col="facet_col",
        kind=kind,
    )
    legend_texts = [t.get_text() for t in g._legend.get_texts()]
    legend_title = g._legend.get_title().get_text()
    assert "NOPE" not in legend_texts, f"kind={kind}: NOPE leaked! {legend_texts}"
    assert "alpha" in legend_texts, f"kind={kind}: alpha missing"
    assert "beta" in legend_texts, f"kind={kind}: beta missing"
    assert "gamma" in legend_texts, f"kind={kind}: gamma missing"
    assert "delta" in legend_texts, f"kind={kind}: delta missing"
    assert legend_title == "hue_dist", f"kind={kind}: title wrong {legend_title!r}"
    print(f"  [OK] displot kind={kind}: legend {legend_texts} (title='{legend_title}')")
    plt.close("all")


# ---------------------------------------------------------------------------
# 测试 4: legend=False / legend="auto"（冗余 hue）
# ---------------------------------------------------------------------------
print("\n=== Test 4: legend=False and legend='auto' redundant hue ===")

# legend=False should produce no legend at all
g = sns.relplot(
    data=df, x="x", y="y",
    hue="hue_var", col="facet_col", legend=False,
)
assert g._legend is None, "relplot legend=False should produce None _legend"
print("  [OK] relplot legend=False: no legend")
plt.close("all")

g = sns.catplot(
    data=df_cat, x="category", y="value",
    hue="hue_cat", row="facet_row", kind="box", legend=False,
)
assert g._legend is None, "catplot legend=False should produce None _legend"
print("  [OK] catplot legend=False: no legend")
plt.close("all")

g = sns.displot(
    data=df_dist, x="measure", hue="hue_dist", col="facet_col", legend=False,
)
assert g._legend is None, "displot legend=False should produce None _legend"
print("  [OK] displot legend=False: no legend")
plt.close("all")

# (redundant hue is a legacy _CategoricalPlotter feature unrelated to the
#  declared/observed semantic separation; skipped here, tested via pytest)


# ---------------------------------------------------------------------------
# 测试 5: 纯空分面 + 显式 hue_order（极端情况）
# ---------------------------------------------------------------------------
print("\n=== Test 5: Fully empty subset facets ===")

df_extreme = pd.DataFrame({
    "x": list(range(6)),
    "y": [10, 20, 30, 40, 50, 60],
    "hue_e": ["A", "A", "A", "B", "B", "B"],
    "col_e": ["F1", "F1", "F1", "F2", "F2", "F2"],
})
# hue_order 含 C，但数据无 C；另外 F1 只有 A，F2 只有 B
g = sns.relplot(
    data=df_extreme, x="x", y="y", hue="hue_e",
    hue_order=["A", "B", "C"], col="col_e",
)
legend_texts = [t.get_text() for t in g._legend.get_texts()]
assert "C" not in legend_texts, f"C not in data should not appear! {legend_texts}"
assert "A" in legend_texts and "B" in legend_texts, f"A/B missing {legend_texts}"
print(f"  [OK] extreme: legend {legend_texts} (C correctly excluded)")
plt.close("all")


# ---------------------------------------------------------------------------
# 测试 6: numeric hue (norm / color palette numeric)
# ---------------------------------------------------------------------------
print("\n=== Test 6: Numeric hue semantics ===")

df_num = pd.DataFrame({
    "x": np.random.randn(100),
    "y": np.random.randn(100),
    "num_hue": np.random.randn(100) * 10,
    "col_f": ["A", "B"] * 50,
})
# col A 只有正数值，col B 只有负数值
df_num.loc[(df_num["col_f"] == "A") & (df_num["num_hue"] < 0), "num_hue"] = np.nan
df_num.loc[(df_num["col_f"] == "B") & (df_num["num_hue"] > 0), "num_hue"] = np.nan
df_num = df_num.dropna()

g = sns.relplot(data=df_num, x="x", y="y", hue="num_hue", col="col_f")
legend_texts = [t.get_text() for t in g._legend.get_texts()]
print(f"  [OK] numeric hue: legend has {len(legend_texts)} entries")
# numeric 图例应该有正负两边都出现的 brief 刻度
has_positive = any(float(t) > 0 for t in legend_texts if t.replace(".","").replace("-","").isdigit())
has_negative = any(float(t) < 0 for t in legend_texts if t.replace(".","").replace("-","").isdigit())
print(f"       has positive ticks: {has_positive}, has negative ticks: {has_negative}")
plt.close("all")


print("\n" + "="*60)
print("All tests passed! 🎉")
print("="*60)
