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

# Manually debug relplot
from seaborn.relational import _ScatterPlotter

kind = "scatter"
x, y, hue, col = 'total_bill', 'tip', 'sex', 'time'
row = None
data = tips
legend = "auto"
palette = hue_order = hue_norm = None
sizes = size_order = size_norm = None
markers = True
dashes = style_order = None
col_wrap = row_order = col_order = None
height = 5
aspect = 1
facet_kws = None
kwargs = {}

Plotter = _ScatterPlotter
func = sns.scatterplot

variables = dict(x=x, y=y, hue=hue)
print(f"Initial variables: {variables}")

p = Plotter(
    data=data,
    variables=variables,
    legend=legend,
)
p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
p.map_size(sizes=sizes, order=size_order, norm=size_norm)
p.map_style(markers=markers, dashes=dashes, order=style_order)

# Save variables here - before assign_variables
variables = p.variables
print(f"Plotter variables (before assign_variables): {variables}")
plot_data = p.plot_data

plot_kws = dict(
    palette=p._hue_map.lookup_table if "hue" in p.variables else None,
    hue_order=p._hue_map.levels if "hue" in p.variables else None,
    hue_norm=p._hue_map.norm if "hue" in p.variables else None,
    sizes=sizes, size_order=size_order, size_norm=size_norm,
    markers=markers, dashes=dashes, style_order=style_order,
    legend=False,
)
plot_kws.update(kwargs)
if kind == "scatter":
    plot_kws.pop("dashes")

grid_variables = dict(
    x=x, y=y, row=row, col=col, hue=hue,
)
p.assign_variables(data, grid_variables)
print(f"Plotter variables (after assign_variables): {p.variables}")

plot_variables = {v: f"_{v}" for v in variables}
print(f"plot_variables: {plot_variables}")
plot_kws.update(plot_variables)
print(f"Final plot_kws keys: {list(plot_kws.keys())}")
