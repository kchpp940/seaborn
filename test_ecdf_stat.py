import numpy as np
import pandas as pd
from seaborn._core.groupby import GroupBy
from seaborn._stats.ecdf import ECDF as ECDFStat
from seaborn._statistics import ECDF as LegacyECDF

class Scale:
    scale_type = 'continuous'

rng = np.random.default_rng(42)

print('=== Test 1: ECDF Stat basic (no grouping) ===')
n = 50
df = pd.DataFrame({'x': rng.normal(0, 1, n), 'group': 0})
gb = GroupBy(['group'])
ecdf_stat = ECDFStat(stat='proportion', common_norm=True)
res = ecdf_stat(df, gb, 'x', {'x': Scale()})
print(f'Rows: {len(res)}')
print(f'First few: {res[["x","y"]].head(5).to_dict(orient="records")}')
print(f'Last value: {res["y"].iloc[-1]:.6f}, expected ~1.0')
print(f'Monotonic: {np.all(np.diff(res["y"].to_numpy()) >= -1e-12)}')

print()
print('=== Test 2: ECDF Stat vs LegacyECDF (same input) ===')
vals = rng.normal(0, 1, 30)
legacy = LegacyECDF(stat='proportion')
y_leg, x_leg = legacy(vals)
print(f'Legacy: len x={len(x_leg)}, y_final={y_leg[-1]:.6f}')

df2 = pd.DataFrame({'x': vals, 'group': 0})
gb2 = GroupBy(['group'])
res2 = ecdf_stat(df2, gb2, 'x', {'x': Scale()})
print(f'Stat:   len x={len(res2)}, y_final={res2["y"].iloc[-1]:.6f}')

# Compare values (skip the -inf first row of stat output)
y_stat = res2['y'].to_numpy()
x_stat = res2['x'].to_numpy()
# legacy starts with -inf too
print(f'y match (truncated): {np.allclose(y_leg, y_stat, atol=1e-12)}')
print(f'x match (truncated): {np.allclose(x_leg[~np.isinf(x_leg)], x_stat[~np.isinf(x_stat)], atol=1e-12)}')

print()
print('=== Test 3: ECDF Stat with grouping + common_norm=True ===')
n = 60
df3 = pd.DataFrame({
    'x': rng.normal(0, 1, n),
    'hue': rng.choice(['a', 'b'], n),
})
gb3 = GroupBy(['hue'])
res3 = ecdf_stat(df3, gb3, 'x', {'x': Scale()})
print(f'Total rows: {len(res3)}')
for hue, sub in res3.groupby('hue'):
    print(f'  hue={hue}: rows={len(sub)}, y_final={sub["y"].iloc[-1]:.6f}, monotonic={np.all(np.diff(sub["y"].to_numpy()) >= -1e-12)}')
# Expected: sum of final values = 1
final_vals = [sub['y'].iloc[-1] for _, sub in res3.groupby('hue')]
print(f'  Sum of final values: {sum(final_vals):.6f}, expected ~1.0 (common_norm=True)')

print()
print('=== Test 4: ECDF Stat with grouping + common_norm=False ===')
ecdf_sep = ECDFStat(stat='proportion', common_norm=False)
res4 = ecdf_sep(df3, gb3, 'x', {'x': Scale()})
for hue, sub in res4.groupby('hue'):
    print(f'  hue={hue}: rows={len(sub)}, y_final={sub["y"].iloc[-1]:.6f}, expected ~1.0')
final_vals4 = [sub['y'].iloc[-1] for _, sub in res4.groupby('hue')]
print(f'  Sum of final values: {sum(final_vals4):.6f}, expected ~2.0 (common_norm=False)')

print()
print('=== Test 5: ECDF Stat with all-zero weight group ===')
df5 = pd.DataFrame({
    'x': rng.normal(0, 1, n),
    'hue': rng.choice(['a', 'b'], n),
    'weight': rng.uniform(0.5, 2.0, n),
})
df5.loc[df5['hue'] == 'b', 'weight'] = 0
gb5 = GroupBy(['hue'])
res5 = ecdf_stat(df5, gb5, 'x', {'x': Scale()})
for hue, sub in res5.groupby('hue'):
    y_vals = sub['y'].to_numpy()
    all_zero = (y_vals == 0).all()
    any_nan = np.any(np.isnan(y_vals))
    mono = np.all(np.diff(y_vals) >= -1e-12)
    print(f'  hue={hue}: y_final={sub["y"].iloc[-1]:.6f}, all_zero={all_zero}, any_nan={any_nan}, monotonic={mono}')
final_vals5 = [sub['y'].iloc[-1] for _, sub in res5.groupby('hue')]
print(f'  Sum of final values: {sum(final_vals5):.6f}, expected ~1.0 (only A)')

print()
print('=== Test 6: ECDF Stat empty group ===')
df6 = pd.DataFrame({
    'x': np.array([np.nan, np.inf, 1.0, -np.inf], dtype=float),
    'hue': ['a', 'a', 'b', 'b'],
})
gb6 = GroupBy(['hue'])
try:
    res6 = ecdf_stat(df6, gb6, 'x', {'x': Scale()})
    for hue, sub in res6.groupby('hue'):
        y_vals = sub['y'].to_numpy()
        any_nan = np.any(np.isnan(y_vals))
        print(f'  hue={hue}: rows={len(sub)}, final={y_vals[-1]:.6f}, any_nan={any_nan}')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== Test 7: ECDF Stat percent ===')
ecdf_pct = ECDFStat(stat='percent', common_norm=True)
df7 = pd.DataFrame({'x': vals, 'group': 0})
res7 = ecdf_pct(df7, GroupBy(['group']), 'x', {'x': Scale()})
print(f'percent final={res7["y"].iloc[-1]:.6f}, expected ~100')

print()
print('=== Test 8: ECDF Stat count ===')
ecdf_cnt = ECDFStat(stat='count')
res8 = ecdf_cnt(df7, GroupBy(['group']), 'x', {'x': Scale()})
print(f'count final={res8["y"].iloc[-1]:.6f}, expected {len(vals)}')

print()
print('=== Test 9: ECDF Stat complementary ===')
ecdf_comp = ECDFStat(stat='proportion', complementary=True)
res9 = ecdf_comp(df7, GroupBy(['group']), 'x', {'x': Scale()})
print(f'complementary first={res9["y"].iloc[0]:.6f}, expected ~1.0')
print(f'complementary final={res9["y"].iloc[-1]:.6f}, expected ~0.0')
