import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn._statistics import BinDiagnostics, BinDiagnosticsCollector

print("=" * 60)
print("Test 1: BinDiagnostics and BinDiagnosticsCollector basic usage")
print("=" * 60)

collector = BinDiagnosticsCollector(stat="count", cumulative=False)
x = np.random.randn(100)
bins = np.histogram_bin_edges(x, bins=10)
hist, edges = np.histogram(x, bins=bins)
diag = collector.add_group_univariate(
    group_key=(),
    x=x,
    bin_edges=edges,
    hist=hist,
)
print(f"Diagnostics: {diag}")
print(f"bin_edges shape: {diag.bin_edges.shape}")
print(f"count: {diag.count}")
print(f"weight_sum: {diag.weight_sum}")
print(f"normalization_denominator: {diag.normalization_denominator}")
print(f"empty_reason: {diag.empty_reason}")
print()

print("=" * 60)
print("Test 2: histplot (axes-level) univariate diagnostics")
print("=" * 60)

try:
    data = pd.DataFrame({"x": np.random.randn(100), "hue": ["a"] * 50 + ["b"] * 50})
    ax = sns.histplot(data=data, x="x", hue="hue", stat="density")
    print(f"ax has diagnostics_: {hasattr(ax, 'diagnostics_')}")
    if hasattr(ax, 'diagnostics_'):
        print(f"diagnostics_ keys: {list(ax.diagnostics_.keys())}")
        for k, v in ax.diagnostics_.items():
            print(f"  {k}: {v}")
    print(f"ax has _hist_estimator: {hasattr(ax, '_hist_estimator')}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 60)
print("Test 3: displot (figure-level) univariate diagnostics")
print("=" * 60)

try:
    data = pd.DataFrame({
        "x": np.random.randn(100),
        "hue": ["a"] * 50 + ["b"] * 50,
        "col": ["one", "two"] * 50,
    })
    g = sns.displot(data=data, x="x", hue="hue", col="col", stat="density")
    print(f"g has diagnostics_: {hasattr(g, 'diagnostics_')}")
    if hasattr(g, 'diagnostics_'):
        print(f"diagnostics_ keys: {list(g.diagnostics_.keys())}")
        for k, v in g.diagnostics_.items():
            print(f"  {k}: {v}")
    print(f"g has _hist_estimator: {hasattr(g, '_hist_estimator')}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 60)
print("Test 4: Bivariate histplot diagnostics")
print("=" * 60)

try:
    data = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})
    ax = sns.histplot(data=data, x="x", y="y", stat="density")
    print(f"ax has diagnostics_: {hasattr(ax, 'diagnostics_')}")
    if hasattr(ax, 'diagnostics_'):
        print(f"diagnostics_ keys: {list(ax.diagnostics_.keys())}")
        for k, v in ax.diagnostics_.items():
            print(f"  {k}: {v}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 60)
print("Test 5: objects layer Hist diagnostics")
print("=" * 60)

try:
    from seaborn.objects import Plot, Hist, Bars
    p = Plot(x=np.random.randn(100)).add(Hist(), Bars())
    # Plot is lazy, need to trigger compilation
    # Let's just test the Hist stat directly
    from seaborn._stats.counting import Hist
    from seaborn._core.groupby import GroupBy
    
    hist_stat = Hist(stat="density", bins=10)
    data = pd.DataFrame({"x": np.random.randn(100), "y": 0})
    
    # Check if Hist has diagnostics_ property
    print(f"Hist has diagnostics_: {hasattr(hist_stat, 'diagnostics_')}")
    print(f"Hist has _diagnostics_collector: {hasattr(hist_stat, '_diagnostics_collector')}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

plt.close('all')
print("\nDone!")
