"""Test distributions.py legacy path still works."""
import numpy as np
import pandas as pd
from seaborn._stats.counting import Hist

rng = np.random.default_rng(42)
n = 60
df = pd.DataFrame({
    'x': rng.normal(0, 1, n),
    'hue': rng.choice(['a', 'b'], n),
    'weights': rng.uniform(0.5, 2.0, n),
})

# Simulate what distributions.py does per-hue group:
estimate_kws = dict(stat='probability', bins='auto', binwidth=None, discrete=False)
estimate_kws['bins'] = 10  # 'auto' with weights not supported

estimator = Hist(**estimate_kws)
orient = 'x'

all_data = df.dropna()
multiple_histograms = {'hue'}
if multiple_histograms:
    bin_kws = estimator._define_bin_params(all_data, orient, None)

print('bin_kws:', bin_kws)

# compute whole weight for common_norm
whole_weight = len(all_data)

histograms = {}
from seaborn._core.groupby import GroupBy
# iter_data hue equivalent
for hue_val, sub_data in df.groupby('hue'):
    sub_data = sub_data.copy()
    sub_data['weight'] = sub_data['weights']
    part_weight = len(sub_data)  # no weights version

    if not (multiple_histograms and True):  # common_bins=True
        bin_kws = estimator._define_bin_params(sub_data, orient, None)
    
    # This is line 471 in distributions.py
    res = estimator._normalize(estimator._eval(sub_data, orient, bin_kws))
    heights = res[estimator.stat].to_numpy()
    widths = res['space'].to_numpy()
    edges = res[orient].to_numpy() - widths / 2
    
    # common_norm scaling
    common_norm = True
    if common_norm:
        heights *= part_weight / whole_weight
    
    print(f'hue={hue_val}: height_sum={heights.sum():.6f}, expected ~{part_weight / whole_weight:.6f}')
    histograms[hue_val] = heights

total = sum(h.sum() for h in histograms.values())
print(f'Total height sum (common_norm=True): {total:.6f}, expected ~1.0')

print()
print('=== test density stat ===')
estimate_kws2 = dict(stat='density', bins=10, binwidth=None, discrete=False)
estimator2 = Hist(**estimate_kws2)
if multiple_histograms:
    bin_kws2 = estimator2._define_bin_params(all_data, orient, None)

for hue_val, sub_data in df.groupby('hue'):
    sub_data = sub_data.copy()
    part_weight = len(sub_data)

    res = estimator2._normalize(estimator2._eval(sub_data, orient, bin_kws2))
    heights = res[estimator2.stat].to_numpy()
    widths = res['space'].to_numpy()
    
    area = (heights * widths).sum()
    print(f'hue={hue_val}: before scaling area={area:.6f}, expected ~1.0')
    
    if common_norm:
        heights *= part_weight / whole_weight
        area2 = (heights * widths).sum()
        print(f'hue={hue_val}: after scaling area={area2:.6f}, expected ~{part_weight / whole_weight:.6f}')
