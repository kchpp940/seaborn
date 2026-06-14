import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.distributions import _DistributionPlotter, Histogram, _assign_default_kwargs, histplot
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

p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

builder.check_ax(kwargs)
builder.normalize_facet_vars(row=row, col=col, row_order=row_order, col_order=col_order)

grid_data = builder.prepare_grid_data(dropna_how=None)

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

if kind == "kde":
    allowed_types = ["numeric", "datetime"]
else:
    allowed_types = None
p._attach(g, allowed_types=allowed_types, log_scale=log_scale)

if not p.has_xy_data:
    print("No xy data, returning early")

if color is None and hue is None:
    color = "C0"

kwargs["legend"] = legend

print(f"p.univariate: {p.univariate}")
print(f"p.var_types: {p.var_types}")
print(f"p._default_discrete(): {p._default_discrete()}")
print(f"axes: {g.axes.flat}")
for i, ax in enumerate(g.axes.flat):
    print(f"ax[{i}] data: {list(ax.collections)}")

hist_kws = kwargs.copy()

estimate_defaults = {}
_assign_default_kwargs(estimate_defaults, Histogram.__init__, histplot)

estimate_kws = {}
for key, default_val in estimate_defaults.items():
    estimate_kws[key] = hist_kws.pop(key, default_val)

if estimate_kws["discrete"] is None:
    estimate_kws["discrete"] = p._default_discrete()

hist_kws["estimate_kws"] = estimate_kws
hist_kws.setdefault("color", color)

print(f"hist_kws: {hist_kws}")
print(f"estimate_kws: {estimate_kws}")

_assign_default_kwargs(hist_kws, p.plot_univariate_histogram, histplot)
print(f"Final hist_kws: {hist_kws}")

try:
    p.plot_univariate_histogram(**hist_kws)
    print("Plotting succeeded!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
