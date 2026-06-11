"""Comprehensive tests for the unified matrix context refactoring."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seaborn import matrix as mat
from seaborn import heatmap, clustermap

pass_count = 0
fail_count = 0

def check(name, condition, detail=""):
    global pass_count, fail_count
    if condition:
        print(f"  PASS: {name}")
        pass_count += 1
    else:
        print(f"  FAIL: {name} - {detail}")
        fail_count += 1

# Test 1: _MatrixContext basic properties
print('=== Test 1: _MatrixContext basic properties ===')
df = pd.DataFrame({
    'A': pd.array([1, 2, pd.NA, 4], dtype='Int64'),
    'B': pd.array([5, pd.NA, 7, 8], dtype='Int64'),
    'C': pd.array([9, 10, 11, pd.NA], dtype='Int64'),
}, index=['r1', 'r2', 'r3', 'r4'])

ctx = mat._MatrixContext(df)
check('ctx.df has float64 dtype', all(d == np.float64 for d in ctx.df.dtypes), f"got {ctx.df.dtypes.tolist()}")
check('ctx.values is float64', ctx.values.dtype == np.float64, f"got {ctx.values.dtype}")
check('ctx.shape matches', ctx.shape == (4, 3), f"got {ctx.shape}")
check('ctx.index correct', list(ctx.index) == ['r1', 'r2', 'r3', 'r4'])
check('ctx.columns correct', list(ctx.columns) == ['A', 'B', 'C'])
check('ctx.mask covers NAs',
      ctx.mask.iloc[2, 0] == True and ctx.mask.iloc[1, 1] == True and ctx.mask.iloc[3, 2] == True,
      f"mask:\n{ctx.mask}")
check('ctx.plot_data is masked array', isinstance(ctx.plot_data, np.ma.MaskedArray))
check('ctx.plot_data has 3 masked cells', ctx.plot_data.mask.sum() == 3, f"got {ctx.plot_data.mask.sum()}")
check('ctx.xlabel() correct', ctx.xlabel() == '')
check('ctx.ylabel() correct', ctx.ylabel() == '')
check('ctx.xticklabels() correct', list(ctx.xticklabels()) == ['A', 'B', 'C'])
check('ctx.yticklabels() correct', list(ctx.yticklabels()) == ['r1', 'r2', 'r3', 'r4'])
plt.close('all')

# Test 2: _MatrixContext reindex preserves consistency
print()
print('=== Test 2: _MatrixContext.reindex preserves consistency ===')
row_ind = [2, 0, 3, 1]
col_ind = [1, 2, 0]
ctx2 = ctx.reindex(row_ind=row_ind, col_ind=col_ind)
check('reindexed shape same', ctx2.shape == ctx.shape == (4, 3))
check('reindexed index correct', list(ctx2.index) == ['r3', 'r1', 'r4', 'r2'])
check('reindexed columns correct', list(ctx2.columns) == ['B', 'C', 'A'])
check('reindexed data correct',
      ctx2.df.iloc[0, 0] == ctx.df.iloc[2, 1],
      f"got {ctx2.df.iloc[0, 0]}, expected {ctx.df.iloc[2, 1]}")
check('reindexed mask follows data',
      ctx2.mask.iloc[0, 0] == ctx.mask.iloc[2, 1],
      f"got {ctx2.mask.iloc[0, 0]}, expected {ctx.mask.iloc[2, 1]}")
check('reindexed plot_data consistent',
      ctx2.plot_data[0, 0] == ctx.plot_data[2, 1] if not ctx.plot_data.mask[2, 1] else ctx2.plot_data.mask[0, 0] == ctx.plot_data.mask[2, 1])
plt.close('all')

# Test 3: heatmap with nullable Int64 dtype
print()
print('=== Test 3: heatmap with nullable Int64 ===')
df_int = pd.DataFrame({
    'X': pd.array([10, 20, 30], dtype='Int64'),
    'Y': pd.array([40, 50, 60], dtype='Int64'),
})
ax = heatmap(df_int, annot=True, fmt='.0f')
texts = [t.get_text() for t in ax.texts]
expected = ['10', '40', '20', '50', '30', '60']
check('annotations correct for nullable Int64', texts == expected, f"got {texts}")
mesh = ax.collections[0]
check('plot_data is float64', mesh.get_array().dtype == np.float64, f"got {mesh.get_array().dtype}")
plt.close('all')

# Test 4: heatmap with pd.NA mask (boolean dtype)
print()
print('=== Test 4: heatmap with pd.NA mask ===')
df_data = pd.DataFrame(np.arange(12).reshape(3, 4).astype(float))
mask_na = pd.DataFrame({
    0: [False, True, pd.NA],
    1: [pd.NA, False, True],
    2: [True, pd.NA, False],
    3: [False, True, pd.NA],
}, dtype='boolean')
ax = heatmap(df_data, mask=mask_na)
mesh = ax.collections[0]
masked_count = mesh.get_array().mask.sum()
# 3 NA (True) + 4 True + 0 data NA = 7? Let's count:
# col0: F, T, NA(T) = 2 T
# col1: NA(T), F, T = 2 T
# col2: T, NA(T), F = 2 T
# col3: F, T, NA(T) = 2 T
# Total: 8 True
check('correct number of masked cells', masked_count == 8, f"got {masked_count}")
plt.close('all')

# Test 5: MultiIndex with nullable dtype
print()
print('=== Test 5: MultiIndex + nullable dtype ===')
mi_rows = pd.MultiIndex.from_tuples(
    [('A', 1), ('A', 2), ('B', 1)], names=['letter', 'num'])
mi_cols = pd.MultiIndex.from_tuples(
    [('X', 'a'), ('Y', 'b')], names=['grp', 'sub'])
df_mi = pd.DataFrame(
    np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
    index=mi_rows,
    columns=mi_cols
).astype('Int64')
df_mi.iloc[1, 0] = pd.NA
ax = heatmap(df_mi, annot=True, fmt='.0f')
xtl = [t.get_text() for t in ax.get_xticklabels()]
ytl = [t.get_text() for t in ax.get_yticklabels()]
check('MultiIndex x ticklabels', xtl == ['X-a', 'Y-b'], f"got {xtl}")
check('MultiIndex y ticklabels', ytl == ['A-1', 'A-2', 'B-1'], f"got {ytl}")
# 1 NA cell should be skipped in annotation => 5 texts
check('NA skipped in annotation', len(ax.texts) == 5, f"got {len(ax.texts)} texts")
plt.close('all')

# Test 6: clustermap with nullable dtype + colors alignment
print()
print('=== Test 6: clustermap with nullable dtype and colors ===')
df_clust = pd.DataFrame({
    'A': pd.array([1.0, 2.0, pd.NA, 4.0], dtype='Float64'),
    'B': pd.array([5.0, pd.NA, 7.0, 8.0], dtype='Float64'),
    'C': pd.array([9.0, 10.0, 11.0, 12.0], dtype='Float64'),
    'D': pd.array([13.0, 14.0, pd.NA, 16.0], dtype='Float64'),
}, index=['w', 'x', 'y', 'z'])

row_colors = pd.Series(['red', 'blue', 'green', 'orange'],
                       index=['z', 'y', 'x', 'w'], name='group')  # shuffled

g = clustermap(df_clust, row_colors=row_colors,
               row_cluster=False, col_cluster=False, annot=True, fmt='.1f')

check('row color 0 is orange (index w)', 
      tuple(g.row_colors[0]) == matplotlib.colors.to_rgb('orange'),
      f"got {tuple(g.row_colors[0])}")
check('row color 1 is green (index x)',
      tuple(g.row_colors[1]) == matplotlib.colors.to_rgb('green'),
      f"got {tuple(g.row_colors[1])}")
check('row color 2 is blue (index y)',
      tuple(g.row_colors[2]) == matplotlib.colors.to_rgb('blue'),
      f"got {tuple(g.row_colors[2])}")
check('row color 3 is red (index z)',
      tuple(g.row_colors[3]) == matplotlib.colors.to_rgb('red'),
      f"got {tuple(g.row_colors[3])}")
plt.close('all')

# Test 7: clustermap with pivot_kws and colors
print()
print('=== Test 7: clustermap pivot_kws + colors alignment ===')
df_long = pd.DataFrame({
    'sample': ['s1', 's1', 's2', 's2', 's3', 's3'],
    'gene': ['g1', 'g2', 'g1', 'g2', 'g1', 'g2'],
    'value': pd.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='Float64'),
})
row_colors_pivot = pd.Series(
    ['red', 'blue', 'green'],
    index=['s3', 's1', 's2'],  # shuffled order
    name='sample_group'
)
g = clustermap(df_long,
               pivot_kws=dict(index='sample', columns='gene', values='value'),
               row_colors=row_colors_pivot,
               row_cluster=False, col_cluster=False)

# After pivot, index = ['s1', 's2', 's3'], colors should align
# s1 -> blue, s2 -> green, s3 -> red
check('pivot: data2d index correct', list(g.data2d.index) == ['s1', 's2', 's3'])
check('pivot: row color 0 = blue (s1)',
      tuple(g.row_colors[0]) == matplotlib.colors.to_rgb('blue'))
check('pivot: row color 1 = green (s2)',
      tuple(g.row_colors[1]) == matplotlib.colors.to_rgb('green'))
check('pivot: row color 2 = red (s3)',
      tuple(g.row_colors[2]) == matplotlib.colors.to_rgb('red'))
plt.close('all')

# Test 8: ClusterGrid reindex keeps data and mask in sync
print()
print('=== Test 8: ClusterGrid context reindex sync ===')
mask_clust = pd.DataFrame(False, index=df_clust.index, columns=df_clust.columns, dtype='boolean')
mask_clust.iloc[0, 0] = True
mask_clust.iloc[2, 2] = pd.NA  # NA in mask becomes True
g = mat.ClusterGrid(df_clust, mask=mask_clust, dendrogram_ratio=0.2, colors_ratio=0.03)
before_mask_sum = g.mask.sum().sum()
before_data_shape = g.data2d.shape
check('before reindex: mask shape matches data', g.mask.shape == g.data2d.shape)

# Simulate reorder
yind = [3, 0, 2, 1]
xind = [2, 0, 3, 1]
g.ctx = g.ctx.reindex(row_ind=yind, col_ind=xind)
check('after reindex: mask shape matches data', g.mask.shape == g.data2d.shape)
check('after reindex: mask sum preserved', g.mask.sum().sum() == before_mask_sum,
      f"before={before_mask_sum}, after={g.mask.sum().sum()}")
check('after reindex: data shape preserved', g.data2d.shape == before_data_shape)
check('after reindex: index reordered', list(g.data2d.index) == ['z', 'w', 'y', 'x'])
check('after reindex: mask follows index', g.mask.index.equals(g.data2d.index))
check('after reindex: mask follows columns', g.mask.columns.equals(g.data2d.columns))
plt.close('all')

# Test 9: Single source of truth - all components read from same context
print()
print('=== Test 9: Single source of truth ===')
g = clustermap(df_clust, row_colors=row_colors,
               row_cluster=False, col_cluster=False)
check('ClusterGrid uses _MatrixContext', hasattr(g, 'ctx'))
check('data2d is a property reading ctx.df', g.data2d is g.ctx.df)
check('mask is a property reading ctx.mask', g.mask is g.ctx.mask)
check('ctx has df attribute', hasattr(g.ctx, 'df'))
check('ctx has mask attribute', hasattr(g.ctx, 'mask'))
check('ctx has orig_is_frame attribute', hasattr(g.ctx, 'orig_is_frame'))
plt.close('all')

# Test 10: _HeatMapper with context
print()
print('=== Test 10: _HeatMapper from context ===')
ctx_test = mat._MatrixContext(df_int)
hm = mat._HeatMapper(context=ctx_test, annot=True, fmt='.0f')
check('HeatMapper has ctx attribute', hasattr(hm, 'ctx'))
check('HeatMapper data matches ctx.df', hm.data is hm.ctx.df)
check('HeatMapper annot_data shape correct', hm.annot_data.shape == (3, 2))
plt.close('all')

# Test 11: _DendrogramPlotter with context
print()
print('=== Test 11: _DendrogramPlotter from context ===')
ctx_dend = mat._MatrixContext(df_clust.dropna())
dp = mat._DendrogramPlotter(context=ctx_dend, axis=1, metric='euclidean', method='average')
check('DendrogramPlotter has ctx attribute', hasattr(dp, 'ctx'))
check('DendrogramPlotter data matches context', dp.data.shape == (4, 2) if dp.axis == 1 else dp.data.shape == (2, 4))
plt.close('all')

# Test 12: clustermap with clustering reorders context properly
print()
print('=== Test 12: Clustering reorders context consistently ===')
df_cluster = pd.DataFrame({
    'gene1': [1.0, 2.0, 3.0, 4.0],
    'gene2': [5.0, 6.0, 7.0, 8.0],
    'gene3': [9.0, 10.0, 11.0, 12.0],
    'gene4': [13.0, 14.0, 15.0, 16.0],
}, index=['sample1', 'sample2', 'sample3', 'sample4'])

mask_test = pd.DataFrame(False, index=df_cluster.index, columns=df_cluster.columns)
mask_test.iloc[0, 0] = True
mask_test.iloc[2, 3] = True

row_colors_test = pd.Series(
    ['red', 'blue', 'green', 'orange'],
    index=['sample1', 'sample2', 'sample3', 'sample4'],
    name='group'
)

g = clustermap(df_cluster, mask=mask_test, row_colors=row_colors_test,
               row_cluster=True, col_cluster=True)

# After clustering, data, mask, and ticklabels should all be in same order
heatmap_ylabels = [t.get_text() for t in g.ax_heatmap.get_yticklabels()]
heatmap_xlabels = [t.get_text() for t in g.ax_heatmap.get_xticklabels()]
ctx_ylabel_list = list(g.ctx.index)
ctx_xlabel_list = list(g.ctx.columns)

check('heatmap yticklabels match ctx index', heatmap_ylabels == ctx_ylabel_list,
      f"heatmap: {heatmap_ylabels}, ctx: {ctx_ylabel_list}")
check('heatmap xticklabels match ctx columns', heatmap_xlabels == ctx_xlabel_list,
      f"heatmap: {heatmap_xlabels}, ctx: {ctx_xlabel_list}")
check('mask shape matches data2d shape', g.mask.shape == g.data2d.shape)
check('mask index matches data2d index', g.mask.index.equals(g.data2d.index))
check('mask columns matches data2d columns', g.mask.columns.equals(g.data2d.columns))
plt.close('all')

# Summary
print()
print(f"=== Results: {pass_count} passed, {fail_count} failed ===")
if fail_count > 0:
    exit(1)
