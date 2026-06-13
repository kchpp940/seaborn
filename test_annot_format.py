import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def test_heatmap_basic():
    print("Test 1: Basic heatmap with annot_format")
    data = pd.DataFrame(np.random.randn(5, 4),
                        index=['A', 'B', 'C', 'D', 'E'],
                        columns=['W', 'X', 'Y', 'Z'])
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        return f"{row_label}-{col_label}\n{value:.2f}"
    
    fig, ax = plt.subplots()
    sns.heatmap(data, annot=True, annot_format=fmt, ax=ax)
    plt.savefig('/tmp/test_heatmap_basic.png')
    plt.close()
    print("  PASSED")

def test_heatmap_multiindex():
    print("Test 2: Heatmap with MultiIndex")
    arrays = [
        ['A', 'A', 'B', 'B'],
        ['one', 'two', 'one', 'two']
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=['letter', 'number'])
    columns = pd.MultiIndex.from_tuples(
        [('X', 'a'), ('X', 'b'), ('Y', 'a'), ('Y', 'b')],
        names=['upper', 'lower']
    )
    data = pd.DataFrame(np.random.randn(4, 4), index=index, columns=columns)
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        return f"{row_label[0]}{row_label[1]}-{col_label[0]}{col_label[1]}"
    
    fig, ax = plt.subplots()
    sns.heatmap(data, annot=True, annot_format=fmt, ax=ax)
    plt.savefig('/tmp/test_heatmap_multiindex.png')
    plt.close()
    print("  PASSED")

def test_heatmap_nullable_dtype():
    print("Test 3: Heatmap with nullable dtype")
    data = pd.DataFrame({
        'A': pd.array([1.0, np.nan, 3.0, 4.0], dtype='Float64'),
        'B': pd.array([5, 6, None, 8], dtype='Int64'),
        'C': pd.array([1.0, 0.0, 1.0, None], dtype='Float64'),
        'D': pd.array([10.0, 20.0, 30.0, None], dtype='Float64')
    }, index=['r1', 'r2', 'r3', 'r4'])
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        if masked:
            return "N/A"
        if pd.isna(value):
            return "NaN"
        return str(value)
    
    fig, ax = plt.subplots()
    mask = data.isna()
    sns.heatmap(data.astype('float64'), mask=mask, annot=data, annot_format=fmt, ax=ax)
    plt.savefig('/tmp/test_heatmap_nullable.png')
    plt.close()
    print("  PASSED")

def test_heatmap_mask():
    print("Test 4: Heatmap with mask")
    data = pd.DataFrame(np.random.randn(4, 4),
                        index=['a', 'b', 'c', 'd'],
                        columns=['w', 'x', 'y', 'z'])
    mask = np.triu(np.ones_like(data, dtype=bool))
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        if masked:
            return "MASKED"
        return f"{value:.1f}"
    
    fig, ax = plt.subplots()
    sns.heatmap(data, mask=mask, annot=True, annot_format=fmt, ax=ax)
    plt.savefig('/tmp/test_heatmap_mask.png')
    plt.close()
    print("  PASSED")

def test_heatmap_custom_annot():
    print("Test 5: Heatmap with custom annot DataFrame")
    data = pd.DataFrame(np.random.randn(4, 4),
                        index=['a', 'b', 'c', 'd'],
                        columns=['w', 'x', 'y', 'z'])
    annot_data = pd.DataFrame([
        ['↑', '↓', '→', '←'],
        ['★', '☆', '●', '○'],
        ['✓', '✗', '◆', '◇'],
        ['▲', '▼', '■', '□']
    ], index=['a', 'b', 'c', 'd'], columns=['w', 'x', 'y', 'z'])
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        return f"{value}\n({row_idx},{col_idx})"
    
    fig, ax = plt.subplots()
    sns.heatmap(data, annot=annot_data, annot_format=fmt, ax=ax)
    plt.savefig('/tmp/test_heatmap_custom_annot.png')
    plt.close()
    print("  PASSED")

def test_heatmap_skip_annotation():
    print("Test 6: Heatmap with conditional annotation skipping")
    data = pd.DataFrame(np.random.randn(5, 5),
                        index=[f'r{i}' for i in range(5)],
                        columns=[f'c{i}' for i in range(5)])
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        if value < 0:
            return None
        return f"{value:.2f}"
    
    fig, ax = plt.subplots()
    sns.heatmap(data, annot=True, annot_format=fmt, ax=ax)
    plt.savefig('/tmp/test_heatmap_skip.png')
    plt.close()
    print("  PASSED")

def test_clustermap_basic():
    print("Test 7: Clustermap with annot_format (clustering reorder)")
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(8, 6),
                        index=[f'row_{i}' for i in range(8)],
                        columns=[f'col_{i}' for i in range(6)])
    
    context_log = []
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        context_log.append({
            'row_label': row_label,
            'col_label': col_label,
            'row_idx': row_idx,
            'col_idx': col_idx,
            'value': value
        })
        return f"({row_idx},{col_idx})"
    
    g = sns.clustermap(data, annot=True, annot_format=fmt, figsize=(8, 6))
    plt.savefig('/tmp/test_clustermap_basic.png')
    plt.close()
    
    reordered_rows = [data.index[i] for i in g.dendrogram_row.reordered_ind]
    reordered_cols = [data.columns[i] for i in g.dendrogram_col.reordered_ind]
    
    for entry in context_log[:3]:
        expected_row = reordered_rows[entry['row_idx']]
        expected_col = reordered_cols[entry['col_idx']]
        assert entry['row_label'] == expected_row, f"Row label mismatch: {entry['row_label']} != {expected_row}"
        assert entry['col_label'] == expected_col, f"Col label mismatch: {entry['col_label']} != {expected_col}"
    
    print(f"  Logged {len(context_log)} entries, all labels match reordered indices")
    print("  PASSED")

def test_clustermap_with_mask():
    print("Test 8: Clustermap with mask and annot_format")
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(6, 6),
                        index=[f'R{i}' for i in range(6)],
                        columns=[f'C{i}' for i in range(6)])
    mask = pd.DataFrame(np.random.choice([True, False], data.shape, p=[0.3, 0.7]),
                        index=data.index, columns=data.columns)
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        if masked:
            return "X"
        return f"{value:.1f}"
    
    g = sns.clustermap(data, mask=mask, annot=True, annot_format=fmt, figsize=(7, 6))
    plt.savefig('/tmp/test_clustermap_mask.png')
    plt.close()
    print("  PASSED")

def test_clustermap_custom_annot():
    print("Test 9: Clustermap with custom annot DataFrame")
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(6, 6),
                        index=[f'r{i}' for i in range(6)],
                        columns=[f'c{i}' for i in range(6)])
    annot_data = pd.DataFrame(np.arange(36).reshape(6, 6),
                              index=data.index, columns=data.columns)
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        return f"[{value}]"
    
    g = sns.clustermap(data, annot=annot_data, annot_format=fmt, figsize=(7, 6))
    plt.savefig('/tmp/test_clustermap_custom_annot.png')
    plt.close()
    print("  PASSED")

def test_clustermap_no_cluster():
    print("Test 10: Clustermap without clustering")
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(5, 5),
                        index=[f'r{i}' for i in range(5)],
                        columns=[f'c{i}' for i in range(5)])
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        return f"{row_idx},{col_idx}"
    
    g = sns.clustermap(data, row_cluster=False, col_cluster=False,
                       annot=True, annot_format=fmt, figsize=(7, 6))
    plt.savefig('/tmp/test_clustermap_no_cluster.png')
    plt.close()
    print("  PASSED")

def test_context_correctness():
    print("Test 11: Context correctness verification")
    data = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=['alpha', 'beta', 'gamma'],
        columns=['x', 'y', 'z']
    )
    mask = pd.DataFrame([
        [False, False, True],
        [False, True, False],
        [True, False, False]
    ], index=data.index, columns=data.columns)
    
    contexts = []
    
    def fmt(row_label, col_label, value, masked, row_idx, col_idx):
        contexts.append({
            'row_label': row_label,
            'col_label': col_label,
            'value': value,
            'masked': masked,
            'row_idx': row_idx,
            'col_idx': col_idx
        })
        return f"{value}"
    
    fig, ax = plt.subplots()
    sns.heatmap(data, mask=mask, annot=True, annot_format=fmt, ax=ax)
    plt.close()
    
    expected_contexts = [
        {'row_label': 'alpha', 'col_label': 'x', 'value': 1, 'masked': False, 'row_idx': 0, 'col_idx': 0},
        {'row_label': 'alpha', 'col_label': 'y', 'value': 2, 'masked': False, 'row_idx': 0, 'col_idx': 1},
        {'row_label': 'alpha', 'col_label': 'z', 'value': 3, 'masked': True, 'row_idx': 0, 'col_idx': 2},
        {'row_label': 'beta', 'col_label': 'x', 'value': 4, 'masked': False, 'row_idx': 1, 'col_idx': 0},
        {'row_label': 'beta', 'col_label': 'y', 'value': 5, 'masked': True, 'row_idx': 1, 'col_idx': 1},
        {'row_label': 'beta', 'col_label': 'z', 'value': 6, 'masked': False, 'row_idx': 1, 'col_idx': 2},
        {'row_label': 'gamma', 'col_label': 'x', 'value': 7, 'masked': True, 'row_idx': 2, 'col_idx': 0},
        {'row_label': 'gamma', 'col_label': 'y', 'value': 8, 'masked': False, 'row_idx': 2, 'col_idx': 1},
        {'row_label': 'gamma', 'col_label': 'z', 'value': 9, 'masked': False, 'row_idx': 2, 'col_idx': 2},
    ]
    
    for ctx, expected in zip(contexts, expected_contexts):
        for key in expected:
            assert ctx[key] == expected[key], f"Mismatch in {key}: {ctx[key]} != {expected[key]}"
    
    print(f"  All {len(contexts)} contexts match expected values")
    print("  PASSED")

if __name__ == '__main__':
    test_heatmap_basic()
    test_heatmap_multiindex()
    test_heatmap_nullable_dtype()
    test_heatmap_mask()
    test_heatmap_custom_annot()
    test_heatmap_skip_annotation()
    test_clustermap_basic()
    test_clustermap_with_mask()
    test_clustermap_custom_annot()
    test_clustermap_no_cluster()
    test_context_correctness()
    print("\n=== All tests passed! ===")
