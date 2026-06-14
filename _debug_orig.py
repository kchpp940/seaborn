import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

np.random.seed(42)
tips = pd.DataFrame({
    'total_bill': np.random.uniform(10, 50, 100),
    'tip': np.random.uniform(1, 10, 100),
    'sex': np.random.choice(['Male', 'Female'], 100),
})

# Monkey-patch iter_data to see what's happening
from seaborn._base import VectorPlotter
orig_iter = VectorPlotter.iter_data

def debug_iter(self, grouping_vars=None, **kwargs):
    print(f"\n=== DEBUG iter_data ===")
    print(f"grouping_vars input: {grouping_vars}")
    print(f"kwargs: {kwargs}")
    print(f"self.variables: {self.variables}")
    print(f"self._hue_map: {getattr(self, '_hue_map', None)}")
    if hasattr(self, '_hue_map') and self._hue_map is not None:
        print(f"self._hue_map.levels: {self._hue_map.levels}")
    print(f"self._var_levels before access: {self._var_levels}")
    print(f"self.var_levels (property): {self.var_levels}")
    
    result = list(orig_iter(self, grouping_vars, **kwargs))
    print(f"Number of results: {len(result)}")
    for i, (sv, sd) in enumerate(result):
        print(f"  result[{i}]: sub_vars={sv}, shape={sd.shape}")
    
    for sub_vars, sub_data in result:
        yield sub_vars, sub_data

VectorPlotter.iter_data = debug_iter

print("Calling sns.displot (ORIGINAL)...")
try:
    g = sns.displot(data=tips, x='total_bill', hue='sex', kind='hist')
    print(f"\nResult: OK, axes shape={g.axes.shape}")
except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()
