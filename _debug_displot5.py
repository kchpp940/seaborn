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
    'time': np.random.choice(['Lunch', 'Dinner'], 100),
})

# First test: displot without col
print("Test 1: displot without col")
try:
    g = sns.displot(data=tips, x='total_bill', hue='sex', kind='hist')
    print(f"  OK: axes shape={g.axes.shape}")
except Exception as e:
    print(f"  FAILED: {e}")
plt.close('all')

# Second test: displot with col
print("\nTest 2: displot with col (time)")
try:
    g = sns.displot(data=tips, x='total_bill', hue='sex', col='time', kind='hist')
    print(f"  OK: axes shape={g.axes.shape}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
plt.close('all')
