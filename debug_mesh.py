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

fig, ax = plt.subplots()
plotter = sns.matrix._HeatMapper(
    data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
    annot=True, fmt=".2g", annot_kws=None, cbar=True, cbar_kws=None,
    xticklabels=True, yticklabels=True, mask=None
)

mesh = ax.pcolormesh(plotter.plot_data, cmap=plotter.cmap)
mesh.update_scalarmappable()

print("plot_data:")
print(plotter.plot_data)
print("\nmesh.get_array():")
print(mesh.get_array())
print("\nmesh.get_array().flat:")
print(list(mesh.get_array().flat))
print("\nplotter.annot_data.flat:")
print(list(plotter.annot_data.flat))

height, width = plotter.annot_data.shape
xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)

print("\nZip order:")
for i, (x, y, m, val) in enumerate(zip(
        xpos.flat, ypos.flat,
        mesh.get_array().flat,
        plotter.annot_data.flat)):
    row = i // width
    col = i % width
    print(f"  i={i}: (row={row}, col={col}), x={x}, y={y}, m={m}, val={val}")

print("\nNow with inverted yaxis:")
ax.invert_yaxis()
print("ax.get_ylim():", ax.get_ylim())
