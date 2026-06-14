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

# Monkey-patch to debug
from seaborn.distributions import _DistributionPlotter
orig_plot = _DistributionPlotter.plot_univariate_histogram

def debug_plot(self, **kwargs):
    print("\n=== DEBUG plot_univariate_histogram ===")
    print(f"variables: {self.variables}")
    print(f"var_levels: {self.var_levels}")
    print(f"var_types: {self.var_types}")
    print(f"comp_data columns: {list(self.comp_data.columns)}")
    print(f"comp_data shape: {self.comp_data.shape}")
    print(f"comp_data head:\n{self.comp_data.head()}")
    print(f"hue in variables: {'hue' in self.variables}")
    
    # Test iter_data
    print("\n--- Testing iter_data ---")
    n = 0
    for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):
        n += 1
        print(f"  group {n}: {sub_vars}, shape={sub_data.shape}")
    print(f"Total groups: {n}")
    
    return orig_plot(self, **kwargs)

_DistributionPlotter.plot_univariate_histogram = debug_plot

print("Calling sns.displot...")
g = sns.displot(data=tips, x='total_bill', hue='sex', kind='hist')
print(f"\nResult: OK, axes shape={g.axes.shape}")
