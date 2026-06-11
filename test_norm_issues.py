import numpy as np
import pandas as pd
from seaborn._core.groupby import GroupBy
from seaborn._stats.counting import Hist
from seaborn._stats.density import KDE
from seaborn._statistics import Histogram, ECDF as LegacyECDF, KDE as LegacyKDE

class Scale:
    scale_type = 'continuous'

print('=== Test 1: Hist - common_norm with hue groups (weighted) ===')
rng = np.random.default_rng(42)
n = 100
df = pd.DataFrame({
    'x': rng.normal(0, 1, n),
    'hue': rng.choice(['a', 'b', 'c'], n),
    'weight': rng.uniform(0.5, 2.0, n),
})
df.loc[df['hue'] == 'b', 'weight'] = 0  # Make one group have all zero weights

gb = GroupBy(['hue'])
h = Hist(stat='probability', common_norm=True)
try:
    out = h(df, gb, 'x', {'x': Scale()})
    print('Hist probability sum (common_norm=True):', out['y'].sum())
    for hue_val, sub in out.groupby('hue'):
        print(f'  hue={hue_val}: sum={sub["y"].sum()}, count_rows={len(sub)}, any_nan={sub["y"].isna().any()}')
except Exception as e:
    print(f'Hist ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== Test 2: Legacy Histogram + common_norm scaling ===')
estimator = Histogram(stat='probability')
for hue_val in ['a', 'b', 'c']:
    sub = df[df['hue'] == hue_val]
    try:
        bin_kws = estimator.define_bin_params(sub['x'], weights=sub['weight'])
        hist, edges = estimator(sub['x'], weights=sub['weight'])
        print(f'  hue={hue_val}: hist_sum={hist.sum()}, any_nan={np.any(np.isnan(hist))}')
    except Exception as e:
        print(f'  hue={hue_val}: ERROR {type(e).__name__}: {e}')

print()
print('=== Test 3: ECDF with zero weights ===')
x = np.array([1.0, 2.0, 3.0, 4.0])
w_zero = np.array([0.0, 0.0, 0.0, 0.0])
ecdf = LegacyECDF(stat='proportion')
try:
    stat, vals = ecdf(x, weights=w_zero)
    print(f'ECDF all-zero weights: stat={stat}, any_nan={np.any(np.isnan(stat))}')
except Exception as e:
    print(f'ECDF all-zero weights ERROR: {type(e).__name__}: {e}')

w_partial = np.array([0.0, 0.0, 1.0, 1.0])
try:
    stat, vals = ecdf(x, weights=w_partial)
    print(f'ECDF partial zero weights: stat={stat}, monotonic={np.all(np.diff(stat) >= 0)}')
except Exception as e:
    print(f'ECDF partial zero weights ERROR: {type(e).__name__}: {e}')

print()
print('=== Test 4: KDE - objects API with empty/zero weight group ===')
df_kde = pd.DataFrame({
    'x': rng.normal(0, 1, 60),
    'alpha': rng.choice(['a', 'b'], 60),
    'weight': rng.uniform(0.5, 2.0, 60),
})
df_kde.loc[df_kde['alpha'] == 'b', 'weight'] = 0

def get_groupby(df, orient):
    cols = [c for c in df if c != orient]
    return GroupBy([*cols, 'group'])

ori = 'x'
df_kde_test = df_kde[[ori, 'alpha', 'weight']]
gb = get_groupby(df_kde_test, ori)
try:
    res = KDE(common_norm=True)(df_kde_test, gb, ori, {})
    print('KDE objects API:')
    print(f'  total rows: {len(res)}')
    for alpha_val, sub in res.groupby('alpha'):
        print(f'  alpha={alpha_val}: rows={len(sub)}, density_sum={sub["density"].sum()}, any_nan={sub["density"].isna().any()}')
except Exception as e:
    print(f'KDE objects ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== Test 5: Hist - discrete variable with common_norm ===')
df_disc = pd.DataFrame({
    'x': rng.choice([1, 2, 3, 4, 5], 50),
    'hue': rng.choice(['a', 'b'], 50),
})
gb_disc = GroupBy(['hue'])
h_disc = Hist(stat='probability', discrete=True, common_norm=True)
try:
    out_disc = h_disc(df_disc, gb_disc, 'x', {'x': Scale()})
    print('Hist discrete probability sum (common_norm=True):', out_disc['y'].sum())
    for hue_val, sub in out_disc.groupby('hue'):
        print(f'  hue={hue_val}: sum={sub["y"].sum()}')
except Exception as e:
    print(f'Hist discrete ERROR: {type(e).__name__}: {e}')

print()
print('=== Test 6: Legacy distributions _resolve_multiple with stack/fill ===')
from seaborn.distributions import _DistributionPlotter

dp = _DistributionPlotter()
# Simulate two normalized probability curves that should stack to 1
curves = {
    (('hue', 'a'),): pd.Series([0.2, 0.3, 0.5], index=[0, 1, 2]),  # sums to 1
    (('hue', 'b'),): pd.Series([0.4, 0.3, 0.3], index=[0, 1, 2]),  # sums to 1
}
dp.variables = {'x': None, 'hue': 'hue'}
dp.var_levels = {'hue': ['a', 'b']}
try:
    stacked, baselines = dp._resolve_multiple(curves.copy(), 'stack')
    print('Stacked result (each row should be cumulative):')
    print(stacked)
    print('Filled result (each row should sum to 1):')
    filled, _ = dp._resolve_multiple(curves.copy(), 'fill')
    print(filled)
    print('Filled row sums:', filled.sum(axis=1).values)
except Exception as e:
    print(f'_resolve_multiple ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
