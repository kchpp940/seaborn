import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn._statistics import BinDiagnostics, BinDiagnosticsCollector

print("=" * 60)
print("Test 1: Objects layer - Hist stat via Plot")
print("=" * 60)

try:
    from seaborn.objects import Plot, Hist, Bars, Dot
    
    # Test with simple data
    p = Plot(x=np.random.randn(100)).add(Bars(), Hist(stat="density", bins=10))
    fig = p.plot()
    
    # Now try to access diagnostics
    # The question is: where do diagnostics live in the objects interface?
    print("Plot object created successfully")
    print(f"Plot type: {type(p)}")
    
    # Let's check if there's a way to access the stat object
    print("\nExploring Plot object structure...")
    # Check if there are layers
    if hasattr(p, '_layers'):
        print(f"Number of layers: {len(p._layers)}")
        for i, layer in enumerate(p._layers):
            print(f"  Layer {i}: {type(layer)}")
            if hasattr(layer, 'stat'):
                print(f"    stat: {type(layer.stat)}")
                stat = layer.stat
                if hasattr(stat, 'diagnostics_'):
                    print(f"    diagnostics_ keys: {list(stat.diagnostics_.keys())}")
                    for k, v in stat.diagnostics_.items():
                        print(f"      {k}: {v}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 60)
print("Test 2: Objects layer - Hist stat with hue")
print("=" * 60)

try:
    from seaborn.objects import Plot, Hist, Bars
    
    data = pd.DataFrame({
        "x": np.random.randn(200),
        "hue": ["a"] * 100 + ["b"] * 100,
    })
    
    p = Plot(data, x="x", color="hue").add(Bars(), Hist(stat="density", bins=10))
    fig = p.plot()
    
    print("Plot with hue created successfully")
    
    # Check layers
    if hasattr(p, '_layers'):
        for i, layer in enumerate(p._layers):
            if hasattr(layer, 'stat'):
                stat = layer.stat
                print(f"Layer {i} stat type: {type(stat).__name__}")
                if hasattr(stat, 'diagnostics_'):
                    print(f"  diagnostics_ keys: {list(stat.diagnostics_.keys())}")
                    for k, v in stat.diagnostics_.items():
                        print(f"    {k}: {v}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 60)
print("Test 3: Empty group detection")
print("=" * 60)

try:
    collector = BinDiagnosticsCollector(stat="count", cumulative=False)
    
    # Test no_data
    empty_x = np.array([])
    hist, edges = np.histogram(empty_x, bins=10, range=(0, 1))
    diag = collector.add_group_univariate(
        group_key=(("case", "no_data"),),
        x=empty_x,
        bin_edges=edges,
        hist=hist,
    )
    print(f"no_data: empty_reason={diag.empty_reason}, count={diag.count}")
    
    # Test all_nan
    nan_x = np.array([np.nan, np.nan, np.nan])
    hist, edges = np.histogram(nan_x, bins=10, range=(0, 1))
    diag = collector.add_group_univariate(
        group_key=(("case", "all_nan"),),
        x=nan_x,
        bin_edges=edges,
        hist=hist,
    )
    print(f"all_nan: empty_reason={diag.empty_reason}, count={diag.count}")
    
    # Test zero_variance
    zero_var_x = np.array([5.0, 5.0, 5.0, 5.0])
    hist, edges = np.histogram(zero_var_x, bins=10, range=(0, 10))
    diag = collector.add_group_univariate(
        group_key=(("case", "zero_variance"),),
        x=zero_var_x,
        bin_edges=edges,
        hist=hist,
    )
    print(f"zero_variance: empty_reason={diag.empty_reason}, count={diag.count}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

plt.close('all')
print("\nDone!")
