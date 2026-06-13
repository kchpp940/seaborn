"""Quick test for refactored Hist diagnostics."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from seaborn._stats.counting import Hist
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Continuous

print("=" * 60)
print("Testing refactored Hist diagnostics")
print("=" * 60)

# Test 1: With hue grouping (common_bins=True, default)
print("\n1. With hue grouping, common_bins=True...")
df2 = pd.DataFrame({
    "x": np.concatenate([np.random.randn(50), np.random.randn(50) + 2]),
    "hue": ["A"] * 50 + ["B"] * 50,
})
hist_stat2 = Hist(stat="density", bins=15)
groupby2 = GroupBy(["hue"])
result2 = hist_stat2(df2, groupby2, "x", {"x": Continuous()})
print(f"   diagnostics keys: {list(hist_stat2.diagnostics_.keys())}")
for k, v in hist_stat2.diagnostics_.items():
    print(f"   {k}: count={v.count}, norm_denom={v.normalization_denominator:.4f}")
assert len(hist_stat2.diagnostics_) == 2
print("   ✓")

# Test 2: histplot (axes-level)
print("\n2. histplot with hue...")
ax = sns.histplot(data=df2, x="x", hue="hue", stat="probability")
print(f"   diagnostics keys: {list(ax.diagnostics_.keys())}")
for k, v in ax.diagnostics_.items():
    print(f"   {k}: count={v.count}, norm_denom={v.normalization_denominator:.4f}")
assert len(ax.diagnostics_) == 2
print("   ✓")

# Test 3: displot (figure-level)
print("\n3. displot with col and hue...")
df3 = pd.DataFrame({
    "x": np.concatenate([
        np.random.randn(25), np.random.randn(25) + 2,
        np.random.randn(25) + 1, np.random.randn(25) + 3,
    ]),
    "hue_var": ["A"] * 25 + ["B"] * 25 + ["A"] * 25 + ["B"] * 25,
    "col_var": ["col1"] * 50 + ["col2"] * 50,
})
g = sns.displot(data=df3, x="x", hue="hue_var", col="col_var", stat="count")
print(f"   diagnostics keys: {list(g.diagnostics_.keys())}")
for k, v in g.diagnostics_.items():
    print(f"   {k}: count={v.count}")
print("   ✓")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
