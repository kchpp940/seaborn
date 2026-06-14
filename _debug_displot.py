import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.distributions import _DistributionPlotter
import pandas as pd
import numpy as np

np.random.seed(42)
tips = pd.DataFrame({
    'total_bill': np.random.uniform(10, 50, 100),
    'tip': np.random.uniform(1, 10, 100),
    'sex': np.random.choice(['Male', 'Female'], 100),
    'time': np.random.choice(['Lunch', 'Dinner'], 100),
})

x, y, hue, row, col, weights = 'total_bill', None, 'sex', None, 'time', None
data = tips
kind = "hist"
rug = False
log_scale = None
legend = True
palette = None
hue_order = None
hue_norm = None
color = None
col_wrap = None
row_order = None
col_order = None
height = 5
aspect = 1
facet_kws = None
kwargs = {}

from seaborn._base import _FacetGridBuilder

builder = _FacetGridBuilder("displot")

p = _DistributionPlotter(
    data=data,
    variables=dict(x=x, y=y, hue=hue, weights=weights, row=row, col=col),
)
builder.plotter = p

print(f"1. p.variables: {p.variables}")
print(f"1. p.plot_data columns: {list(p.plot_data.columns)}")
print(f"1. p.plot_data shape: {p.plot_data.shape}")

p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

builder.check_ax(kwargs)
builder.normalize_facet_vars(row=row, col=col, row_order=row_order, col_order=col_order)

print(f"2. p.variables: {p.variables}")
print(f"2. p.plot_data columns: {list(p.plot_data.columns)}")

grid_data = builder.prepare_grid_data(dropna_how=None)
print(f"grid_data columns: {list(grid_data.columns)}")
print(f"grid_data shape: {grid_data.shape}")

builder.init_facet_grid(
    data=grid_data,
    row=p.variables.get("row"),
    col=p.variables.get("col"),
    col_wrap=col_wrap,
    row_order=row_order,
    col_order=col_order,
    height=height,
    aspect=aspect,
    facet_kws=facet_kws,
)
g = builder.g
print(f"3. FacetGrid data columns: {list(g.data.columns)}")

if kind == "kde":
    allowed_types = ["numeric", "datetime"]
else:
    allowed_types = None
p._attach(g, allowed_types=allowed_types, log_scale=log_scale)

print(f"4. After _attach:")
print(f"   p.has_xy_data: {p.has_xy_data}")
print(f"   p.plot_data columns: {list(p.plot_data.columns)}")
print(f"   p.plot_data shape: {p.plot_data.shape}")
print(f"   p.variables: {p.variables}")
