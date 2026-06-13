import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    index=['alpha', 'beta', 'gamma'],
    columns=['x', 'y', 'z']
)
mask = pd.DataFrame([
    [False, False, True],
    [False, True, False],
    [True, False, False]
], index=data.index, columns=data.columns)

contexts = []

def fmt(row_label, col_label, value, masked, row_idx, col_idx):
    contexts.append({
        'row_label': row_label,
        'col_label': col_label,
        'value': value,
        'masked': masked,
        'row_idx': row_idx,
        'col_idx': col_idx
    })
    return f"{value}"

fig, ax = plt.subplots()
sns.heatmap(data, mask=mask, annot=True, annot_format=fmt, ax=ax)
plt.close()

print("Actual contexts:")
for i, ctx in enumerate(contexts):
    print(f"  [{i}] row_label={ctx['row_label']}, col_label={ctx['col_label']}, value={ctx['value']}, row_idx={ctx['row_idx']}, col_idx={ctx['col_idx']}")

print("\nExpected contexts (row-major):")
expected = [
    {'row_label': 'alpha', 'col_label': 'x', 'value': 1, 'masked': False, 'row_idx': 0, 'col_idx': 0},
    {'row_label': 'alpha', 'col_label': 'y', 'value': 2, 'masked': False, 'row_idx': 0, 'col_idx': 1},
    {'row_label': 'alpha', 'col_label': 'z', 'value': 3, 'masked': True, 'row_idx': 0, 'col_idx': 2},
    {'row_label': 'beta', 'col_label': 'x', 'value': 4, 'masked': False, 'row_idx': 1, 'col_idx': 0},
    {'row_label': 'beta', 'col_label': 'y', 'value': 5, 'masked': True, 'row_idx': 1, 'col_idx': 1},
    {'row_label': 'beta', 'col_label': 'z', 'value': 6, 'masked': False, 'row_idx': 1, 'col_idx': 2},
    {'row_label': 'gamma', 'col_label': 'x', 'value': 7, 'masked': True, 'row_idx': 2, 'col_idx': 0},
    {'row_label': 'gamma', 'col_label': 'y', 'value': 8, 'masked': False, 'row_idx': 2, 'col_idx': 1},
    {'row_label': 'gamma', 'col_label': 'z', 'value': 9, 'masked': False, 'row_idx': 2, 'col_idx': 2},
]
for i, ctx in enumerate(expected):
    print(f"  [{i}] row_label={ctx['row_label']}, col_label={ctx['col_label']}, value={ctx['value']}, row_idx={ctx['row_idx']}, col_idx={ctx['col_idx']}")
