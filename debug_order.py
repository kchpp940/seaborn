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

contexts = []

def fmt(row_label, col_label, value, masked, row_idx, col_idx):
    contexts.append({
        'row_label': row_label,
        'col_label': col_label,
        'value': value,
        'row_idx': row_idx,
        'col_idx': col_idx,
        'y': None
    })
    return f"{value}"

fig, ax = plt.subplots()
plotter = sns.matrix._HeatMapper(
    data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
    annot=True, fmt=".2g", annot_kws=None, cbar=True, cbar_kws=None,
    xticklabels=True, yticklabels=True, mask=None,
    annot_format=fmt
)

print("plotter.row_ind:", plotter.row_ind)
print("plotter.col_ind:", plotter.col_ind)
print("plotter.data.index:", list(plotter.data.index))
print("plotter.original_data.index:", list(plotter.original_data.index))

height, width = data.shape
xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
print("\nxpos:")
print(xpos)
print("\nypos:")
print(ypos)
print("\nxpos.flat:", list(xpos.flat))
print("ypos.flat:", list(ypos.flat))
print("\nannot_data.flat:", list(plotter.annot_data.flat))

print("\nExpected order (row-major):")
for i in range(height * width):
    row = i // width
    col = i % width
    print(f"  i={i}: (row={row}, col={col}), y={ypos.flat[i]}, x={xpos.flat[i]}, val={plotter.annot_data.flat[i]}")
