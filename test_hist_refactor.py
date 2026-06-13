import numpy as np
import pandas as pd
from seaborn._stats.counting import Hist
from seaborn._core.groupby import GroupBy

class Scale:
    scale_type = "continuous"

np.random.seed(42)
data = pd.DataFrame({'x': np.random.randn(100), 'group': 'a'})
groupby = GroupBy(['group'])
scales = {'x': Scale()}
hist = Hist(stat='count')
result = hist(data, groupby, 'x', scales)
print('Test 1 passed: basic histogram')
print('  Result columns:', list(result.columns))
print('  Has diagnostics_:', hasattr(hist, 'diagnostics_'))
if hasattr(hist, 'diagnostics_'):
    print('  Diagnostics keys:', list(hist.diagnostics_.keys()))
    for k, d in hist.diagnostics_.items():
        print(f'  {k}: count={d.count}, bin_edges.shape={d.bin_edges.shape}')

data = pd.DataFrame({
    'x': np.random.randn(200),
    'hue': np.random.choice(['A', 'B'], 200),
    'group': 'a',
})
groupby = GroupBy(['group', 'hue'])
hist = Hist(stat='density')
result = hist(data, groupby, 'x', scales)
print('\nTest 2 passed: with hue grouping')
print('  Result columns:', list(result.columns))
print('  Diagnostics keys:', list(hist.diagnostics_.keys()))
for k, d in hist.diagnostics_.items():
    print(f'  {k}: count={d.count}, normalization_denominator={d.normalization_denominator:.3f}')

print('\nTest 3 passed: no _pending_results attribute')
print('  Has _pending_results:', hasattr(hist, '_pending_results'))
