import numpy as np
import pandas as pd
from seaborn._core.groupby import GroupBy
from seaborn._stats.counting import Hist
from seaborn._stats.density import KDE
from seaborn._statistics import Histogram, ECDF as LegacyECDF, KDE as LegacyKDE

class Scale:
    scale_type = 'continuous'

rng = np.random.default_rng(42)

def trapz_area(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    dx = np.diff(x)
    return float(np.sum(y[:-1] * dx))

print('=== KDE - basic (no grouping) check area ===')
n = 200
df = pd.DataFrame({'x': rng.normal(0, 1, n), 'group': 0})
gb = GroupBy(['group'])
res = KDE(common_norm=True)(df, gb, 'x', {'x': Scale()})
print(f'Rows: {len(res)}')
print(f'Area (trap): {trapz_area(res["x"], res["y"]):.6f}, expected ~1.0')

print()
print('=== KDE - with two groups, common_norm=True ===')
n = 120
df2 = pd.DataFrame({
    'x': rng.normal(0, 1, n),
    'alpha': rng.choice(['a', 'b'], n),
})

gb_kde = GroupBy(['alpha'])
try:
    res = KDE(common_norm=True)(df2, gb_kde, 'x', {'x': Scale()})
    print(f'Total rows: {len(res)}')
    for alpha_val, sub in res.groupby('alpha'):
        area = trapz_area(sub['x'], sub['y'])
        print(f'  alpha={alpha_val}: rows={len(sub)}, area={area:.6f}, any_nan={sub["y"].isna().any()}')
    # Total area - need to handle potentially different x grids per group
    pivot = res.pivot_table(index='x', columns='alpha', values='y', aggfunc='first').fillna(0)
    x_sorted = np.sort(pivot.index.to_numpy())
    total_area = 0.0
    for col in pivot.columns:
        total_area += trapz_area(x_sorted, pivot[col].to_numpy())
    print(f'  Total area (sum of groups): {total_area:.6f}, expected ~1.0')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== KDE - with two groups, common_norm=False ===')
try:
    res2 = KDE(common_norm=False)(df2, gb_kde, 'x', {'x': Scale()})
    total_area2 = 0.0
    for alpha_val, sub in res2.groupby('alpha'):
        area = trapz_area(sub['x'], sub['y'])
        print(f'  alpha={alpha_val}: area={area:.6f}, expected ~1.0')
        total_area2 += area
    print(f'  Total area: {total_area2:.6f}, expected ~2.0')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== KDE - weighted, one group zero weight ===')
df3 = pd.DataFrame({
    'x': rng.normal(0, 1, n),
    'alpha': rng.choice(['a', 'b'], n),
    'weight': rng.uniform(0.5, 2.0, n),
})
df3.loc[df3['alpha'] == 'b', 'weight'] = 0
try:
    res3 = KDE(common_norm=True)(df3, gb_kde, 'x', {'x': Scale()})
    for alpha_val, sub in res3.groupby('alpha'):
        area = trapz_area(sub['x'], sub['y'])
        print(f'  alpha={alpha_val}: rows={len(sub)}, area={area:.6f}, any_nan={sub["y"].isna().any()}, all_zero={(sub["y"] == 0).all()}')
    pivot3 = res3.pivot_table(index='x', columns='alpha', values='y', aggfunc='first').fillna(0)
    x_sorted3 = np.sort(pivot3.index.to_numpy())
    total_area3 = 0.0
    for col in pivot3.columns:
        total_area3 += trapz_area(x_sorted3, pivot3[col].to_numpy())
    print(f'  Total area: {total_area3:.6f}, expected ~1.0 (only A contributes)')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== Legacy KDE sanity check ===')
sub_a = df2[df2['alpha'] == 'a']
sub_b = df2[df2['alpha'] == 'b']
kde_legacy = LegacyKDE(bw_adjust=1, gridsize=200)
try:
    density_a, grid_a = kde_legacy(sub_a['x'])
    area_a = trapz_area(grid_a, density_a)
    print(f'  Legacy KDE A alone area: {area_a:.6f}, expected ~1.0')
except Exception as e:
    print(f'  Legacy A ERROR: {type(e).__name__}: {e}')
