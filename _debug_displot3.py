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
log_scale = None
col_wrap = None
row_order = None
col_order = None
height = 5
aspect = 1
facet_kws = None

# ORIGINAL WAY
from seaborn.axisgrid import FacetGrid

p_orig = _DistributionPlotter(
    data=data,
    variables=dict(x=x, y=y, hue=hue, weights=weights, row=row, col=col),
)
p_orig.map_hue()

for var in ["row", "col"]:
    if var in p_orig.variables and p_orig.variables[var] is None:
        p_orig.variables[var] = f"_{var}_"

if row is not None:
    p_orig._order_registry.register("row", order=row_order)
if col is not None:
    p_orig._order_registry.register("col", order=col_order)

grid_data_orig = p_orig.plot_data.rename(columns=p_orig.variables)
grid_data_orig = grid_data_orig.loc[:, ~grid_data_orig.columns.duplicated()]

col_name = p_orig.variables.get("col")
row_name = p_orig.variables.get("row")

g_orig = FacetGrid(
    data=grid_data_orig, row=row_name, col=col_name,
    col_wrap=col_wrap, row_order=row_order,
    col_order=col_order, height=height,
    aspect=aspect,
    order_registry=p_orig._order_registry,
)

if kind == "kde":
    allowed_types = ["numeric", "datetime"]
else:
    allowed_types = None
p_orig._attach(g_orig, allowed_types=allowed_types, log_scale=log_scale)

print("=== ORIGINAL ===")
print(f"p_orig.plot_data columns: {list(p_orig.plot_data.columns)}")
print(f"p_orig.plot_data head:\n{p_orig.plot_data.head()}")
print(f"p_orig.converters: {p_orig.converters}")

plt.close('all')

# BUILDER WAY
from seaborn._base import _FacetGridBuilder

builder = _FacetGridBuilder("displot")

p_new = _DistributionPlotter(
    data=data,
    variables=dict(x=x, y=y, hue=hue, weights=weights, row=row, col=col),
)
builder.plotter = p_new
p_new.map_hue()

builder.normalize_facet_vars(row=row, col=col, row_order=row_order, col_order=col_order)

grid_data_new = builder.prepare_grid_data(dropna_how=None)

builder.init_facet_grid(
    data=grid_data_new,
    row=p_new.variables.get("row"),
    col=p_new.variables.get("col"),
    col_wrap=col_wrap,
    row_order=row_order,
    col_order=col_order,
    height=height,
    aspect=aspect,
    facet_kws=facet_kws,
)
g_new = builder.g

p_new._attach(g_new, allowed_types=allowed_types, log_scale=log_scale)

print("\n=== BUILDER ===")
print(f"p_new.plot_data columns: {list(p_new.plot_data.columns)}")
print(f"p_new.plot_data head:\n{p_new.plot_data.head()}")
print(f"p_new.converters: {p_new.converters}")
