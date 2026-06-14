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
    'smoker': np.random.choice(['Yes', 'No'], 100),
    'day': np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], 100),
    'time': np.random.choice(['Lunch', 'Dinner'], 100),
    'size': np.random.randint(1, 6, 100)
})

print('Testing relplot...')
g = sns.relplot(data=tips, x='total_bill', y='tip', hue='sex', col='time')
print(f'  relplot OK: type={type(g).__name__}, axes shape={g.axes.shape}')
plt.close('all')

print('Testing catplot...')
g = sns.catplot(data=tips, x='day', y='total_bill', hue='sex', kind='box', col='time')
print(f'  catplot OK: type={type(g).__name__}, axes shape={g.axes.shape}')
plt.close('all')

print('Testing displot...')
g = sns.displot(data=tips, x='total_bill', hue='sex', col='time', kind='hist')
print(f'  displot OK: type={type(g).__name__}, axes shape={g.axes.shape}')
plt.close('all')

print('Testing lmplot...')
g = sns.lmplot(data=tips, x='total_bill', y='tip', hue='sex', col='time')
print(f'  lmplot OK: type={type(g).__name__}, axes shape={g.axes.shape}')
plt.close('all')

print('\nAll figure-level functions work correctly!')
