import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.distributions import _DistributionPlotter
from seaborn.axisgrid import FacetGrid
from seaborn._base import _FacetGridBuilder
import pandas as pd
import numpy as np

np.random.seed(42)
tips = pd.DataFrame({
    'total_bill': np.random.uniform(10, 50, 100),
    'tip': np.random.uniform(1, 10, 100),
    'sex': np.random.choice(['Male', 'Female'], 100),
    'time': np.random.choice(['Lunch', 'Dinner'], 100),
})

x, y, hue, row, col, weights = 'total_bill', None, 'sex', None, None, None
data = tips
log_scale = None
col_wrap = row_order = col_order = None
height = 5
aspect = 1
facet_kws = None
kwargs = {}
kind = "hist"
legend = True
palette = hue_order = hue_norm = None
color = None

def step(name, val):
    print(f"\n=== {name} ===")
    print(val)

# === ORIGINAL ===
print("=" * 60)
print("ORIGINAL DISPLOT")
print("=" * 60)

p_orig = _DistributionPlotter(
    data=data,
    variables=dict(x=x, y=y, hue=hue, weights=weights, row=row, col=col),
)
step("p_orig.variables after init", p_orig.variables)
step("p_orig.plot_data columns after init", list(p_orig.plot_data.columns))

p_orig.map_hue(palette=palette, order=hue_order, norm=hue_norm)

# ax check
if "ax" in kwargs:
    kwargs.pop("ax")

for var in ["row", "col"]:
    if var in p_orig.variables and p_orig.variables[var] is None:
        p_orig.variables[var] = f"_{var}_"
step("p_orig.variables after normalize", p_orig.variables)

if row is not None:
    p_orig._order_registry.register("row", order=row_order)
if col is not None:
    p_orig._order_registry.register("col", order=col_order)

grid_data_orig = p_orig.plot_data.rename(columns=p_orig.variables)
grid_data_orig = grid_data_orig.loc[:, ~grid_data_orig.columns.duplicated()]
step("grid_data_orig columns", list(grid_data_orig.columns))

col_name = p_orig.variables.get("col")
row_name = p_orig.variables.get("row")
if facet_kws is None:
    facet_kws = {}

g_orig = FacetGrid(
    data=grid_data_orig, row=row_name, col=col_name,
    col_wrap=col_wrap, row_order=row_order,
    col_order=col_order, height=height,
    aspect=aspect,
    order_registry=p_orig._order_registry,
    **facet_kws,
)
step("g_orig.data columns", list(g_orig.data.columns))
step("g_orig.data.shape", g_orig.data.shape)

allowed_types = None
p_orig._attach(g_orig, allowed_types=allowed_types, log_scale=log_scale)

step("p_orig.comp_data columns after _attach", list(p_orig.comp_data.columns))
step("p_orig.comp_data shape", p_orig.comp_data.shape)
step("p_orig.variables after _attach", p_orig.variables)
step("p_orig.var_types", p_orig.var_types)

# Check iter_data
print("\n--- iter_data check ---")
for i, (sub_vars, sub_data) in enumerate(p_orig.iter_data("hue", from_comp_data=True)):
    print(f"  iter {i}: sub_vars={sub_vars}, shape={sub_data.shape}")

# Now test plotting
print("\n--- Plotting ---")
try:
    hist_kws = {'legend': True}
    from seaborn.distributions import Histogram, _assign_default_kwargs, histplot
    estimate_defaults = {}
    _assign_default_kwargs(estimate_defaults, Histogram.__init__, histplot)
    estimate_kws = {}
    for key, default_val in estimate_defaults.items():
        estimate_kws[key] = hist_kws.pop(key, default_val)
    if estimate_kws["discrete"] is None:
        estimate_kws["discrete"] = p_orig._default_discrete()
    hist_kws["estimate_kws"] = estimate_kws
    hist_kws.setdefault("color", color)
    _assign_default_kwargs(hist_kws, p_orig.plot_univariate_histogram, histplot)
    p_orig.plot_univariate_histogram(**hist_kws)
    print("  PLOTTING SUCCEEDED!")
except Exception as e:
    print(f"  PLOTTING FAILED: {e}")
    import traceback
    traceback.print_exc()

plt.close('all')
