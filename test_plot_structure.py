import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seaborn.objects import Plot, Hist, Bars

print("Exploring Plot and layer structure...")

data = pd.DataFrame({
    "x": np.random.randn(100),
    "hue": ["a"] * 50 + ["b"] * 50,
})

p = Plot(data, x="x", color="hue").add(Bars(), Hist(stat="density", bins=10))

print(f"\nPlot attributes:")
for attr in dir(p):
    if not attr.startswith('_'):
        print(f"  {attr}")

print(f"\nPlot._layers: {type(p._layers)}")
print(f"Number of layers: {len(p._layers)}")
for i, layer in enumerate(p._layers):
    print(f"\nLayer {i}: {type(layer)}")
    if isinstance(layer, dict):
        for k, v in layer.items():
            print(f"  {k}: {type(v)} = {v}")
            if k == 'stat' and hasattr(v, 'diagnostics_'):
                print(f"    diagnostics_: {v.diagnostics_}")

# Try to plot and then check
print("\n" + "=" * 60)
print("After plotting:")
print("=" * 60)
fig = p.plot()

print(f"After plot - p._layers:")
for i, layer in enumerate(p._layers):
    print(f"\nLayer {i}: {type(layer)}")
    if isinstance(layer, dict):
        for k, v in layer.items():
            print(f"  {k}: {type(v)}")
            if k == 'stat':
                stat = v
                print(f"    stat type: {type(stat).__name__}")
                if hasattr(stat, 'diagnostics_'):
                    print(f"    diagnostics_ keys: {list(stat.diagnostics_.keys())}")
                    for k2, v2 in stat.diagnostics_.items():
                        print(f"      {k2}: {v2}")

plt.close('all')
print("\nDone!")
