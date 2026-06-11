"""Rigorous tests for context consistency after clustering reorder."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seaborn import clustermap

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

# Test 1: annot DataFrame with different index order - should align by index
print('=== Test 1: annot DataFrame alignment by index after reorder ===')
df = pd.DataFrame({
    'A': [1.0, 2.0, 3.0, 4.0],
    'B': [5.0, 6.0, 7.0, 8.0],
    'C': [9.0, 10.0, 11.0, 12.0],
    'D': [13.0, 14.0, 15.0, 16.0],
}, index=['w', 'x', 'y', 'z'])

# Create annot with shuffled index AND different values per index
annot_df = pd.DataFrame({
    'A': ['wA', 'xA', 'yA', 'zA'],
    'B': ['wB', 'xB', 'yB', 'zB'],
    'C': ['wC', 'xC', 'yC', 'zC'],
    'D': ['wD', 'xD', 'yD', 'zD'],
}, index=['z', 'y', 'x', 'w'])  # shuffled index order

print(f"  data index: {list(df.index)}")
print(f"  annot index: {list(annot_df.index)}")

g = clustermap(df, annot=annot_df, fmt='s',
               row_cluster=False, col_cluster=False)

texts = [t.get_text() for t in g.ax_heatmap.texts]
print(f"  texts (first row should be w's values): {texts[:4]}")

# First row of heatmap = index 'w', annot should be wA, wB, wC, wD
check('annot aligned by index (row 0 = wA)', texts[0] == 'wA', f"got '{texts[0]}'")
check('annot aligned by index (row 0, col 1 = wB)', texts[1] == 'wB', f"got '{texts[1]}'")
plt.close('all')

# Test 2: annot DataFrame with clustering reorder
print()
print('=== Test 2: annot DataFrame after clustering reorder ===')
df2 = pd.DataFrame({
    'gene1': [10, 20, 30, 40],
    'gene2': [50, 60, 70, 80],
    'gene3': [90, 100, 110, 120],
}, index=['sample1', 'sample2', 'sample3', 'sample4'])

annot_df2 = pd.DataFrame(
    [['s1g1', 's1g2', 's1g3'],
     ['s2g1', 's2g2', 's2g3'],
     ['s3g1', 's3g2', 's3g3'],
     ['s4g1', 's4g2', 's4g3']],
    index=['sample4', 'sample3', 'sample2', 'sample1'],  # shuffled
    columns=['gene3', 'gene1', 'gene2']  # shuffled columns
)

g2 = clustermap(df2, annot=annot_df2, fmt='s',
                row_cluster=True, col_cluster=True)

# After clustering, each cell's annot should match its (row_label, col_label)
ytl = [t.get_text() for t in g2.ax_heatmap.get_yticklabels()]
xtl = [t.get_text() for t in g2.ax_heatmap.get_xticklabels()]
texts2 = [t.get_text() for t in g2.ax_heatmap.texts]

print(f"  row order: {ytl}")
print(f"  col order: {xtl}")
print(f"  first 3 texts: {texts2[:3]}")

# Verify each cell has correct annotation based on its label
# Build expected annot matrix aligned to data index/columns
annot_aligned = annot_df2.reindex(index=df2.index, columns=df2.columns)

# Then compare each cell
correct = True
for i, ylabel in enumerate(ytl):
    for j, xlabel in enumerate(xtl):
        expected = annot_aligned.loc[ylabel, xlabel]
        actual = texts2[i * len(xtl) + j]
        if expected != actual:
            correct = False
            print(f"    MISMATCH at ({ylabel}, {xlabel}): expected '{expected}', got '{actual}'")

check('annot DataFrame correctly aligned after clustering reorder', correct)
plt.close('all')

# Test 3: MultiIndex + clustering reorder
print()
print('=== Test 3: MultiIndex labels after clustering reorder ===')
mi_rows = pd.MultiIndex.from_tuples(
    [('groupA', 's1'), ('groupA', 's2'), ('groupB', 's3'), ('groupB', 's4')],
    names=['group', 'sample']
)
mi_cols = pd.MultiIndex.from_tuples(
    [('pathwayX', 'g1'), ('pathwayX', 'g2'), ('pathwayY', 'g3')],
    names=['pathway', 'gene']
)
df_mi = pd.DataFrame(
    np.random.randn(4, 3),
    index=mi_rows,
    columns=mi_cols
)

g3 = clustermap(df_mi, row_cluster=True, col_cluster=True,
                row_colors=None, col_colors=None)

ytl3 = [t.get_text() for t in g3.ax_heatmap.get_yticklabels()]
xtl3 = [t.get_text() for t in g3.ax_heatmap.get_xticklabels()]

print(f"  yticklabels: {ytl3}")
print(f"  xticklabels: {xtl3}")

# All labels should be in "group-sample" / "pathway-gene" format
all_y_hyphen = all('-' in label for label in ytl3)
all_x_hyphen = all('-' in label for label in xtl3)
check('MultiIndex row labels use hyphen format after reorder', all_y_hyphen, f"got {ytl3}")
check('MultiIndex col labels use hyphen format after reorder', all_x_hyphen, f"got {xtl3}")
plt.close('all')

# Test 4: mask + annot DataFrame sync after reorder
print()
print('=== Test 4: mask and annot DataFrame sync after reorder ===')
df4 = pd.DataFrame({
    'A': [1.0, 2.0, 3.0, 4.0],
    'B': [5.0, 6.0, 7.0, 8.0],
    'C': [9.0, 10.0, 11.0, 12.0],
}, index=['r1', 'r2', 'r3', 'r4'])

mask4 = pd.DataFrame(False, index=df4.index, columns=df4.columns)
mask4.loc['r2', 'B'] = True  # mask one cell

annot4 = pd.DataFrame(
    [['r1A', 'r1B', 'r1C'],
     ['r2A', 'r2B', 'r2C'],
     ['r3A', 'r3B', 'r3C'],
     ['r4A', 'r4B', 'r4C']],
    index=['r4', 'r3', 'r2', 'r1'],  # shuffled
    columns=['C', 'A', 'B']  # shuffled
)

g4 = clustermap(df4, mask=mask4, annot=annot4, fmt='s',
                row_cluster=True, col_cluster=True)

# Count annotations - should be total cells - 1 (masked)
n_cells = df4.shape[0] * df4.shape[1]
n_masked = mask4.sum().sum()
n_texts = len(g4.ax_heatmap.texts)
print(f"  total cells: {n_cells}, masked: {n_masked}, texts: {n_texts}")
check('correct number of annotations (masked cells skipped)', 
      n_texts == n_cells - n_masked,
      f"expected {n_cells - n_masked}, got {n_texts}")

# Verify the masked cell has no annotation
# Get the heatmap mesh to find masked cells
mesh = g4.ax_heatmap.collections[0]
masked_mask = mesh.get_array().mask
print(f"  mesh masked cells: {masked_mask.sum()}")
check('masked cells count matches mask', masked_mask.sum() == n_masked)

# Also verify that annotation text matches the cell's data label
ytl4 = [t.get_text() for t in g4.ax_heatmap.get_yticklabels()]
xtl4 = [t.get_text() for t in g4.ax_heatmap.get_xticklabels()]
texts4 = [t.get_text() for t in g4.ax_heatmap.texts]

annot_aligned4 = annot4.reindex(index=df4.index, columns=df4.columns)
text_idx = 0
annot_match = True
for i, ylabel in enumerate(ytl4):
    for j, xlabel in enumerate(xtl4):
        if not masked_mask[i, j]:  # if not masked, there should be an annotation
            expected = annot_aligned4.loc[ylabel, xlabel]
            actual = texts4[text_idx]
            if expected != actual:
                annot_match = False
                print(f"    MISMATCH at ({ylabel}, {xlabel}): expected '{expected}', got '{actual}'")
            text_idx += 1

check('annotation values match after reorder and mask skip', annot_match)
plt.close('all')

# Test 5: row_colors with MultiIndex + clustering
print()
print('=== Test 5: row_colors with MultiIndex index ===')
mi_rows5 = pd.MultiIndex.from_tuples(
    [('A', 1), ('A', 2), ('B', 1), ('B', 2)],
    names=['group', 'id']
)
df5 = pd.DataFrame(
    np.random.randn(4, 3),
    index=mi_rows5,
    columns=['x', 'y', 'z']
)

row_colors5 = pd.Series(
    ['red', 'blue', 'green', 'orange'],
    index=mi_rows5,  # MultiIndex, should align by index
    name='color'
)
# Shuffle the color Series
row_colors5 = row_colors5.iloc[[2, 0, 3, 1]]
print(f"  color index order: {list(row_colors5.index)}")

g5 = clustermap(df5, row_colors=row_colors5,
                row_cluster=False, col_cluster=False)

# First row should be ('A', 1) => 'red'
# Let's verify using the color matrix
ytl5 = [t.get_text() for t in g5.ax_heatmap.get_yticklabels()]
print(f"  heatmap yticklabels: {ytl5}")

# Check that first row color corresponds to first heatmap row label
# row_colors is a list of lists; each inner list is RGB tuple
print(f"  row_colors[0]: {g5.row_colors[0]}")

# Build expected color mapping
expected_colors = {}
for idx, color in zip(row_colors5.index, row_colors5):
    expected_colors[idx] = matplotlib.colors.to_rgb(color)

# Get actual colors from the heatmap
actual_first = tuple(g5.row_colors[0])
expected_first = expected_colors[mi_rows5[0]]  # first row of data = ('A', 1)
print(f"  expected first color (A-1): {expected_first}")
print(f"  actual first color: {actual_first}")
check('MultiIndex row_colors correctly aligned by index', actual_first == expected_first)
plt.close('all')

# Summary
print()
print(f"=== Results: {pass_count} passed, {fail_count} failed ===")
if fail_count > 0:
    exit(1)
