import numpy as np
import pandas as pd
from seaborn._core.groupby import GroupBy
from seaborn._stats.counting import Hist
from seaborn._stats.density import KDE
from seaborn._statistics import Histogram, ECDF as LegacyECDF, KDE as LegacyKDE

class Scale:
    scale_type = 'continuous'

rng = np.random.default_rng(42)

print('=== Test 1: Hist - compare common_norm=True per-group vs total ===')
n = 60
df = pd.DataFrame({
    'x': rng.normal(0, 1, n),
    'hue': rng.choice(['a', 'b'], n),
})
gb = GroupBy(['hue'])

# With common_norm=True
h_common = Hist(stat='probability', bins=10, common_norm=True)
out_common = h_common(df, gb, 'x', {'x': Scale()})
print('common_norm=True total prob sum:', out_common['y'].sum())
for hue_val, sub in out_common.groupby('hue'):
    print(f'  hue={hue_val}: sum={sub["y"].sum():.6f}')

# With common_norm=False
h_sep = Hist(stat='probability', bins=10, common_norm=False)
out_sep = h_sep(df, gb, 'x', {'x': Scale()})
print('common_norm=False total prob sum:', out_sep['y'].sum())
for hue_val, sub in out_sep.groupby('hue'):
    print(f'  hue={hue_val}: sum={sub["y"].sum():.6f}')

# Compare with expected: common_norm=True means each group is proportion of total
n_a = (df['hue'] == 'a').sum()
n_b = (df['hue'] == 'b').sum()
n_total = len(df)
print(f'Expected (by count): sum_a={n_a/n_total:.6f}, sum_b={n_b/n_total:.6f}')

print()
print('=== Test 2: Hist - weighted common_norm ===')
df_w = pd.DataFrame({
    'x': rng.normal(0, 1, n),
    'hue': rng.choice(['a', 'b'], n),
    'weight': rng.uniform(0.5, 2.0, n),
})
gb_w = GroupBy(['hue'])

h_common_w = Hist(stat='probability', bins=10, common_norm=True)
out_common_w = h_common_w(df_w, gb_w, 'x', {'x': Scale()})
print('common_norm=True weighted total prob sum:', out_common_w['y'].sum())
for hue_val, sub in out_common_w.groupby('hue'):
    print(f'  hue={hue_val}: sum={sub["y"].sum():.6f}')

w_a = df_w.loc[df_w['hue'] == 'a', 'weight'].sum()
w_b = df_w.loc[df_w['hue'] == 'b', 'weight'].sum()
w_total = w_a + w_b
print(f'Expected (by weight): sum_a={w_a/w_total:.6f}, sum_b={w_b/w_total:.6f}')

print()
print('=== Test 3: Hist _normalize per group vs all ===')
# Check what _normalize does when applied to per-group data vs all data concatenated
sub_a = df[df['hue'] == 'a']
sub_b = df[df['hue'] == 'b']
h_test = Hist(stat='probability', bins=10)

bin_kws_all = h_test._define_bin_params(df, 'x', 'continuous')
eval_all = h_test._eval(df, 'x', bin_kws_all)
norm_all = h_test._normalize(eval_all)
print('_normalize on ALL data: prob sum =', norm_all['probability'].sum())

eval_a = h_test._eval(sub_a, 'x', bin_kws_all)  # same bins
norm_a = h_test._normalize(eval_a)
print('_normalize on group A only: prob sum =', norm_a['probability'].sum())

eval_b = h_test._eval(sub_b, 'x', bin_kws_all)
norm_b = h_test._normalize(eval_b)
print('_normalize on group B only: prob sum =', norm_b['probability'].sum())

print()
print('=== Test 4: Legacy plot_univariate_histogram approach simulation ===')
# Legacy does: per-group normalize, then scale by part_weight/whole_weight
whole_weight = len(df)
part_a = len(sub_a)
part_b = len(sub_b)
# Legacy: each group first normalizes to 1, then scales
legacy_a = norm_a['probability'].to_numpy() * (part_a / whole_weight)
legacy_b = norm_b['probability'].to_numpy() * (part_b / whole_weight)
print('Legacy approach (norm then scale):')
print(f'  A sum: {legacy_a.sum():.6f}, B sum: {legacy_b.sum():.6f}, total: {(legacy_a + legacy_b).sum():.6f}')
print(f'  Objects approach (all data norm): total={norm_all["probability"].sum():.6f}')

# Are they the same?
print(f'  Legacy total == objects total? {abs((legacy_a + legacy_b).sum() - norm_all["probability"].sum()) < 1e-10}')
# Check per-bin
print('  Per-bin max diff:', np.abs(legacy_a + legacy_b - norm_all['probability'].to_numpy()).max())

print()
print('=== Test 5: ECDF edge cases ===')
# Empty data
ecdf = LegacyECDF(stat='proportion')
try:
    stat, vals = ecdf(np.array([]))
    print(f'Empty ECDF: stat={stat}, vals={vals}')
except Exception as e:
    print(f'Empty ECDF ERROR: {type(e).__name__}: {e}')

# Single value
try:
    stat, vals = ecdf(np.array([5.0]))
    print(f'Single value ECDF: stat={stat}, vals={vals}')
except Exception as e:
    print(f'Single value ECDF ERROR: {type(e).__name__}: {e}')

# Single value with zero weight
try:
    stat, vals = ecdf(np.array([5.0]), weights=np.array([0.0]))
    print(f'Single value zero-weight ECDF: stat={stat}, vals={vals}, any_nan={np.any(np.isnan(stat))}')
except Exception as e:
    print(f'Single value zero-weight ECDF ERROR: {type(e).__name__}: {e}')

# All zero weights
try:
    stat, vals = ecdf(np.array([1.0, 2.0, 3.0]), weights=np.array([0.0, 0.0, 0.0]))
    print(f'All zero-weight ECDF: stat={stat}, vals={vals}, any_nan={np.any(np.isnan(stat))}')
except Exception as e:
    print(f'All zero-weight ECDF ERROR: {type(e).__name__}: {e}')

print()
print('=== Test 6: Check if Hist drops NaN/Inf correctly ===')
df_bad = pd.DataFrame({
    'x': [1.0, 2.0, np.nan, np.inf, -np.inf, 3.0],
    'hue': ['a', 'a', 'a', 'a', 'a', 'a'],
})
h_bad = Hist(stat='count', bins=5)
gb_bad = GroupBy(['hue'])
try:
    out_bad = h_bad(df_bad, gb_bad, 'x', {'x': Scale()})
    print(f'Hist with NaN/Inf: count sum={out_bad["y"].sum()}, expected=3')
except Exception as e:
    print(f'Hist with NaN/Inf ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== Test 7: Hist - density stat with common_norm ===')
h_dens = Hist(stat='density', bins=10, common_norm=True)
out_dens = h_dens(df, gb, 'x', {'x': Scale()})
print('Density common_norm=True:')
area = (out_dens['y'] * out_dens['space']).sum()
print(f'  Total area: {area:.6f} (should be 1)')
for hue_val, sub in out_dens.groupby('hue'):
    area_g = (sub['y'] * sub['space']).sum()
    print(f'  hue={hue_val}: area={area_g:.6f}')

h_dens_sep = Hist(stat='density', bins=10, common_norm=False)
out_dens_sep = h_dens_sep(df, gb, 'x', {'x': Scale()})
print('Density common_norm=False:')
area = (out_dens_sep['y'] * out_dens_sep['space']).sum()
print(f'  Total area: {area:.6f}')
for hue_val, sub in out_dens_sep.groupby('hue'):
    area_g = (sub['y'] * sub['space']).sum()
    print(f'  hue={hue_val}: area={area_g:.6f} (should be ~1)')

print()
print('=== Test 8: Hist - cumulative with common_norm ===')
h_cum = Hist(stat='probability', bins=10, cumulative=True, common_norm=True)
out_cum = h_cum(df, gb, 'x', {'x': Scale()})
print('Cumulative probability common_norm=True:')
# The max of the cumulative should be 1 (total prob)
print(f'  Max cum value (all groups summed per bin then max): {out_cum.groupby("x")["y"].sum().max():.6f}')
# Each group's max should be their share
for hue_val, sub in out_cum.groupby('hue'):
    print(f'  hue={hue_val}: max cum={sub["y"].max():.6f}')
