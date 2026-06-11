import numpy as np
import pandas as pd
from seaborn._core.groupby import GroupBy
from seaborn._stats.counting import Hist
from seaborn._stats.norm_utils import normalize_histogram

class Scale:
    scale_type = 'continuous'

rng = np.random.default_rng(42)
n = 60
df = pd.DataFrame({
    'x': rng.normal(0, 1, n),
    'hue': rng.choice(['a', 'b'], n),
})
print('n_a =', (df['hue'] == 'a').sum())
print('n_b =', (df['hue'] == 'b').sum())
print('n_total =', len(df))

counts_a = np.array([1, 2, 4, 5, 5, 4, 4, 2, 1, 1])
total_weight_norm = 60
bin_width = 0.5

prob_a = normalize_histogram(counts_a, np.full(10, bin_width), 'probability', total_weight=total_weight_norm)
print('prob_a sum =', prob_a.sum(), '(expected 29/60 =', 29/60, ')')

prob_a_sep = normalize_histogram(counts_a, np.full(10, bin_width), 'probability', total_weight=29)
print('prob_a_sep sum =', prob_a_sep.sum(), '(expected 1)')

h = Hist(stat='probability', bins=10, common_norm=True)
gb = GroupBy(['hue'])
scale_type = 'continuous'

bin_kws = h._define_bin_params(df, 'x', scale_type)
print('bin_kws:', bin_kws)

data_parts = []
for hue_val, sub in df.groupby('hue'):
    ev = h._eval(sub, 'x', bin_kws)
    ev['hue'] = hue_val
    data_parts.append(ev)
data_eval = pd.concat(data_parts, ignore_index=True)
print('After _eval:')
print(data_eval[['hue', '_total_weight']].drop_duplicates())
print('Total weight sum:', data_eval['_total_weight'].sum())

tw = float(data_eval['_total_weight'].sum())
print('Calling _normalize with total_weight =', tw)
normed = h._normalize(data_eval, total_weight=tw)
print('After _normalize:')
for hue_val, sub in normed.groupby('hue'):
    print(f'  hue={hue_val}: prob sum =', sub['probability'].sum())
print('  total prob sum =', normed['probability'].sum())

print()
print('=== Now test with norm_utils.filter_valid_samples called first ===')
from seaborn._stats.norm_utils import filter_valid_samples
df_filt = filter_valid_samples(df, 'x')
print(df_filt[['hue', 'weight']].groupby('hue').sum())
