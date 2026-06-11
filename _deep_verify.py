"""
Deep verification test for the unified _MatrixContext refactoring.
Tests all the key scenarios the refactoring was designed to fix.
"""
import sys
sys.path.insert(0, '/Users/pkcha/seaborn')

import numpy as np
import pandas as pd
import seaborn.matrix as mat
from seaborn.matrix import _MatrixContext, heatmap, clustermap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl


def test_nullable_dtypes():
    """Test Int64/Float64 nullable dtypes with pd.NA are correctly handled."""
    print("Test 1: Nullable dtypes with pd.NA...", end=" ")

    df = pd.DataFrame({
        'A': pd.array([1, 2, pd.NA, 4], dtype='Int64'),
        'B': pd.array([1.5, pd.NA, 3.5, 4.5], dtype='Float64'),
        'C': pd.array([10, 20, 30, pd.NA], dtype='Int64'),
    })

    ctx = _MatrixContext(df)

    # Values should be float64 with np.nan
    assert ctx.values.dtype == np.float64, f"Expected float64, got {ctx.values.dtype}"
    assert np.isnan(ctx.values[2, 0]), "Row 2, Col 0 should be NaN"
    assert np.isnan(ctx.values[1, 1]), "Row 1, Col 1 should be NaN"
    assert np.isnan(ctx.values[3, 2]), "Row 3, Col 2 should be NaN"

    # DataFrame should still be clean
    assert list(ctx.columns) == ['A', 'B', 'C']
    assert list(ctx.index) == [0, 1, 2, 3]

    print("PASSED")


def test_mask_with_pdna():
    """Test pd.NA in boolean masks are treated as True (masked)."""
    print("Test 2: Mask with pd.NA...", end=" ")

    data = pd.DataFrame(np.random.randn(4, 4), index=list('abcd'), columns=list('wxyz'))

    # Mask with pd.NA (treated as True = masked)
    mask = pd.DataFrame({
        'w': pd.array([False, False, pd.NA, True], dtype='boolean'),
        'x': pd.array([False, pd.NA, False, False], dtype='boolean'),
        'y': pd.array([pd.NA, False, False, False], dtype='boolean'),
        'z': pd.array([False, False, False, False], dtype='boolean'),
    }, index=list('abcd'))

    ctx = _MatrixContext(data, mask=mask)

    # pd.NA positions should be True in the mask
    assert ctx.mask.loc['c', 'w'] == True, "pd.NA should become True"
    assert ctx.mask.loc['b', 'x'] == True, "pd.NA should become True"
    assert ctx.mask.loc['a', 'y'] == True, "pd.NA should become True"
    assert ctx.mask.loc['a', 'w'] == False, "False stays False"

    # Mask AND bug fix: if mask has mismatched index, it should raise
    bad_mask = pd.DataFrame(np.zeros((4, 4), dtype=bool),
                            index=list('abcf'), columns=list('wxyz'))  # 'f' vs 'd'
    try:
        _MatrixContext(data, mask=bad_mask)
        assert False, "Should have raised ValueError for mismatched mask index"
    except ValueError:
        pass  # Expected

    print("PASSED")


def test_annot_df_alignment():
    """Test DataFrame annot is aligned by index/columns, not position."""
    print("Test 3: DataFrame annot index alignment...", end=" ")

    data = pd.DataFrame(np.random.randn(4, 4),
                        index=['row1', 'row2', 'row3', 'row4'],
                        columns=['col1', 'col2', 'col3', 'col4'])

    # Shuffled annot - same labels, different order
    annot_shuffled = pd.DataFrame(
        [['A1', 'B1', 'C1', 'D1'],
         ['A3', 'B3', 'C3', 'D3'],
         ['A2', 'B2', 'C2', 'D2'],
         ['A4', 'B4', 'C4', 'D4']],
        index=['row1', 'row3', 'row2', 'row4'],  # Shuffled rows
        columns=['col2', 'col1', 'col4', 'col3'],  # Shuffled columns
    )

    ctx = _MatrixContext(data)
    ctx = ctx.with_annot(annot_shuffled)

    # Annot should now be aligned to data order
    annot_arr = ctx.annot

    # Check cell (row1, col1) = 'A1' (was in row1, col2 of shuffled annot)
    assert annot_arr[0, 1] == 'A1', f"Expected 'A1' at (0,1), got {annot_arr[0, 1]}"
    # Check cell (row3, col3) = 'C3' (was in row3['row2'], col3 of shuffled annot)
    assert annot_arr[2, 3] == 'C3', f"Expected 'C3' at (2,3), got {annot_arr[2, 3]}"

    # String annot preserved as object dtype (not coerced to numeric)
    assert annot_arr.dtype == object, f"String annot should be object dtype, got {annot_arr.dtype}"

    print("PASSED")


def test_string_annot_not_coerced():
    """Test string annotations are NOT coerced to float."""
    print("Test 4: String annot not coerced to float...", end=" ")

    data = pd.DataFrame([[1, 2], [3, 4]])
    annot = pd.DataFrame([['low', 'med'], ['med', 'high']])

    ctx = _MatrixContext(data)
    ctx = ctx.with_annot(annot)

    annot_array = ctx.annot

    assert isinstance(annot_array[0, 0], str), f"Expected string, got {type(annot_array[0, 0])}"
    assert annot_array[0, 0] == 'low', f"Expected 'low', got {annot_array[0, 0]}"
    assert annot_array[1, 1] == 'high', f"Expected 'high', got {annot_array[1, 1]}"

    # Try with numpy array of strings
    annot_np = np.array([['a', 'b'], ['c', 'd']])
    ctx2 = _MatrixContext(data)
    ctx2 = ctx2.with_annot(annot_np)
    assert ctx2.annot.dtype == '<U1' or ctx2.annot.dtype == object, "String array preserved"

    print("PASSED")


def test_unified_reindex():
    """Test that reindex reorders data, mask, annot ALL together atomically."""
    print("Test 5: Unified reindex(data, mask, annot atomic)...", end=" ")

    data = pd.DataFrame(
        [[10, 20, 30, 40],
         [50, 60, 70, 80],
         [90, 100, 110, 120]],
        index=['r1', 'r2', 'r3'],
        columns=['c1', 'c2', 'c3', 'c4']
    )

    mask = pd.DataFrame(
        [[False, False, True, False],
         [False, True, False, False],
         [True, False, False, False]],
        index=['r1', 'r2', 'r3'],
        columns=['c1', 'c2', 'c3', 'c4']
    )

    annot = pd.DataFrame(
        [['r1c1', 'r1c2', 'r1c3', 'r1c4'],
         ['r2c1', 'r2c2', 'r2c3', 'r2c4'],
         ['r3c1', 'r3c2', 'r3c3', 'r3c4']],
        index=['r1', 'r2', 'r3'],
        columns=['c1', 'c2', 'c3', 'c4']
    )

    ctx = _MatrixContext(data, mask=mask)
    ctx = ctx.with_annot(annot)

    # Reorder rows: [r3, r1, r2] = indices [2, 0, 1]
    # Reorder cols: [c4, c3, c2, c1] = indices [3, 2, 1, 0]
    ctx2 = ctx.reindex(row_ind=[2, 0, 1], col_ind=[3, 2, 1, 0])

    # Check data order
    assert ctx2.df.index.tolist() == ['r3', 'r1', 'r2']
    assert ctx2.df.columns.tolist() == ['c4', 'c3', 'c2', 'c1']
    assert ctx2.values[0, 0] == 120, f"r3,c4 = 120, got {ctx2.values[0, 0]}"
    assert ctx2.values[1, 1] == 30, f"r1,c3 = 30, got {ctx2.values[1, 1]}"

    # Check mask order (same reorder)
    assert ctx2.mask.index.tolist() == ['r3', 'r1', 'r2']
    assert ctx2.mask.columns.tolist() == ['c4', 'c3', 'c2', 'c1']
    assert ctx2.mask.iloc[0, 0] == False, f"r3,c4 was False"
    assert ctx2.mask.iloc[0, 3] == True, f"r3,c1 was True"
    assert ctx2.mask.iloc[1, 1] == True, f"r1,c3 was True"

    # Check annot order
    assert ctx2.annot[0, 0] == 'r3c4', f"Expected 'r3c4', got {ctx2.annot[0, 0]}"
    assert ctx2.annot[1, 1] == 'r1c3', f"Expected 'r1c3', got {ctx2.annot[1, 1]}"
    assert ctx2.annot[2, 2] == 'r2c2', f"Expected 'r2c2', got {ctx2.annot[2, 2]}"

    print("PASSED")


def test_colors_index_alignment():
    """Test that DataFrame colors with shuffled indices align by label, not position."""
    print("Test 6: Colors index alignment...", end=" ")

    rs = np.random.RandomState(42)
    x = rs.randn(4, 6) + np.arange(6)
    letters = pd.Series(["A", "B", "C", "D", "E", "F"], name="letters")
    df = pd.DataFrame(x, columns=letters)

    # Create col_colors with shuffled index
    from seaborn.palettes import color_palette
    col_pal = color_palette('Dark2', 6)
    col_colors_df = pd.DataFrame({'side': list(col_pal)},
                                  index=letters.tolist())
    shuffled_idx = [5, 3, 1, 0, 2, 4]
    col_colors_shuffled = col_colors_df.iloc[shuffled_idx]

    # ClusterGrid with colors
    cg = mat.ClusterGrid(df, col_colors=col_colors_shuffled,
                         dendrogram_ratio=.2, colors_ratio=.03)

    # After __init__, col_colors should be aligned by index
    expected_first = list(map(mpl.colors.to_rgb, col_pal))[0]
    actual_first = list(cg.col_colors)[0][0]

    assert np.allclose(expected_first, actual_first, atol=1e-10), \
        f"First color mismatch: expected {expected_first}, got {actual_first}"

    print("PASSED")


def test_standalone_heatmap_annot_sync():
    """Test standalone heatmap with DataFrame annot."""
    print("Test 7: Standalone heatmap annot sync...", end=" ")

    data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        columns=['cA', 'cB', 'cC'],
                        index=['rA', 'rB', 'rC'])

    mask = pd.DataFrame([[True, False, False],
                         [False, False, True],
                         [False, True, False]],
                        columns=['cA', 'cB', 'cC'],
                        index=['rA', 'rB', 'rC'])

    # Annot with shuffled rows and columns (same labels, different order)
    annot = pd.DataFrame([
        ['rA_cA', 'rA_cB', 'rA_cC'],
        ['rC_cA', 'rC_cB', 'rC_cC'],
        ['rB_cA', 'rB_cB', 'rB_cC']],
        index=['rA', 'rC', 'rB'],
        columns=['cB', 'cA', 'cC'])

    fig, ax = plt.subplots()
    result = heatmap(data, mask=mask, annot=annot, fmt='s', ax=ax)

    # Get the annotations from the plot
    texts = [t.get_text() for t in ax.texts]

    # Masked cells should have empty annot, unmasked should have correct value
    # Cell rA,cA is masked=True: annot text 'rA_cA' should not appear there
    # But annot[rA, cA] = from shuffled, rA row is index 0, cA column is 'cA'
    # In the shuffled annot, columns are ['cB', 'cA', 'cC']
    # So cA is at position 1, cB is at position 0
    # rA row: ['rA_cB'(cB,pos0), 'rA_cA'(cA,pos1), 'rA_cC'(cC,pos2)]
    # After alignment: rA,cA → 'rA_cA', rA,cB → 'rA_cB', rA,cC → 'rA_cC'

    # Cell (0,1) = rA,cB = unmasked → 'rA_cB'
    # Cell (1,0) = rB,cA = unmasked → 'rB_cA'
    # Cell (2,1) = rC,cB = masked → '' (not in texts or empty)

    # The texts are in row-major order for unmasked cells
    expected_texts = [
        'rA_cB', 'rA_cC',  # rA: cB, cC unmasked
        'rB_cA', 'rB_cB',  # rB: cA, cB unmasked
        'rC_cA', 'rC_cC',  # rC: cA, cC unmasked
    ]

    assert texts == expected_texts, f"\nGot:      {texts}\nExpected: {expected_texts}"

    plt.close(fig)
    print("PASSED")


def test_clustermap_multiindex():
    """Test MultiIndex rows/columns work correctly with clustermap."""
    print("Test 8: MultiIndex rows/columns in clustermap...", end=" ")

    arrays = [
        ['A', 'A', 'B', 'B'],
        ['one', 'two', 'one', 'two'],
    ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples)
    columns = pd.MultiIndex.from_tuples(tuples)

    data = pd.DataFrame(np.random.randn(4, 4), index=index, columns=columns)
    mask = pd.DataFrame(np.zeros((4, 4), dtype=bool), index=index, columns=columns)

    g = clustermap(data, mask=mask, row_cluster=False, col_cluster=False,
                   dendrogram_ratio=.2, colors_ratio=.03)

    # Check tick labels are correct MultiIndex
    xtl = [t.get_text() for t in g.ax_heatmap.get_xticklabels()]
    ytl = [t.get_text() for t in g.ax_heatmap.get_yticklabels()]

    # MultiIndex labels should be combined
    assert len(xtl) == 4, f"Expected 4 xticklabels, got {len(xtl)}"
    assert len(ytl) == 4, f"Expected 4 yticklabels, got {len(ytl)}"

    plt.close('all')
    print("PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Running deep verification tests for _MatrixContext refactoring")
    print("=" * 60)

    test_nullable_dtypes()
    test_mask_with_pdna()
    test_annot_df_alignment()
    test_string_annot_not_coerced()
    test_unified_reindex()
    test_colors_index_alignment()
    test_standalone_heatmap_annot_sync()
    test_clustermap_multiindex()

    print("=" * 60)
    print("All deep verification tests PASSED!")
    print("=" * 60)
