"""Test diagnostics for all axes-level and figure-level functions."""
import os
os.environ["SEABORN_DIAGNOSTICS"] = "1"

import seaborn as sns
import matplotlib
matplotlib.use("Agg")
from seaborn._testing import get_diagnostics
import traceback

tips = sns.load_dataset("tips")

def check_diag(name, diag, expected):
    """Check diagnostic info."""
    print(f"\n=== {name} ===")
    if not diag:
        print(f"  FAIL: No diagnostics found!")
        return False

    all_ok = True

    # Check variables
    if "variables" in diag and "assignments" in diag["variables"]:
        vars_list = list(diag["variables"]["assignments"].keys())
        print(f"  Variables: {vars_list}")
    else:
        print(f"  FAIL: No variables found!")
        all_ok = False

    # Check mappings
    if "mappings" in diag:
        print(f"  Mappings: {len(diag['mappings'])}")
        for m in diag["mappings"]:
            print(f"    - {m['map_type']}: var={m['var']}, levels={m['levels']}, source={m['source']}")
    else:
        print(f"  FAIL: No mappings found!")
        all_ok = False

    # Check layers
    if "layers" in diag and diag["layers"]:
        layer = diag["layers"][0]
        print(f"  Layers: {len(diag['layers'])}")
        print(f"    kind: {layer.get('kind')}")
        print(f"    stat: {layer.get('stat')}")
        print(f"    stat_params keys: {list(layer.get('stat_params', {}).keys())}")
        print(f"    orient: {layer.get('orient')}")
        print(f"    grouping_vars: {layer.get('grouping_vars')}")
        print(f"    group_keys count: {len(layer.get('group_keys', []))}")
        if layer.get('group_keys'):
            print(f"    group_keys[0]: {layer['group_keys'][0]}")
        print(f"    draw_function: {layer.get('draw_function')}")
        print(f"    legend_source: {layer.get('legend_source')}")
        print(f"    legend_method: {layer.get('legend_method')}")

        # Check expected fields
        for field, expected_val in expected.items():
            actual = layer.get(field)
            if expected_val is not None:
                if field == "stat_params" and expected_val == "not_empty":
                    if not actual:
                        print(f"    FAIL: {field} should not be empty!")
                        all_ok = False
                elif field == "group_keys" and expected_val == ">0":
                    if len(actual) == 0:
                        print(f"    FAIL: {field} should have >0 entries!")
                        all_ok = False
                elif expected_val == "not_none":
                    if actual is None:
                        print(f"    FAIL: {field} should not be None!")
                        all_ok = False
                elif isinstance(expected_val, str) and expected_val.startswith("~"):
                    # Partial match
                    if actual is None or expected_val[1:] not in str(actual):
                        print(f"    FAIL: {field} should contain '{expected_val[1:]}', got '{actual}'")
                        all_ok = False
                elif actual != expected_val:
                    print(f"    FAIL: {field} expected '{expected_val}', got '{actual}'")
                    all_ok = False
    else:
        print(f"  FAIL: No layers found!")
        all_ok = False

    # Check legend
    if "legend" in diag and "entries" in diag["legend"] and diag["legend"]["entries"]:
        print(f"  Legend entries: {len(diag['legend']['entries'])}")
    else:
        print(f"  (No legend entries)")

    # Check order
    if "order" in diag and "entries" in diag["order"]:
        print(f"  Order entries: {list(diag['order']['entries'].keys())}")
    else:
        print(f"  (No order entries)")

    return all_ok

all_passed = True

# Test 1: histplot (axes-level)
try:
    ax = sns.histplot(data=tips, x="total_bill", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("histplot (axes-level)", diag, {
        "kind": "hist",
        "stat": "Hist",
        "stat_params": "not_empty",
        "orient": "not_none",
        "draw_function": "ax.bar",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 2: kdeplot (axes-level)
try:
    ax = sns.kdeplot(data=tips, x="total_bill", hue="sex", fill=True)
    diag = get_diagnostics(ax)
    ok = check_diag("kdeplot (axes-level)", diag, {
        "stat": "KDE",
        "stat_params": "not_empty",
        "orient": "not_none",
        "draw_function": "ax.fill_between",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_add_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 3: ecdfplot (axes-level)
try:
    ax = sns.ecdfplot(data=tips, x="total_bill", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("ecdfplot (axes-level)", diag, {
        "stat": "ECDF",
        "stat_params": "not_empty",
        "orient": "not_none",
        "draw_function": "ax.plot",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_add_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 4: rugplot (axes-level)
try:
    ax = sns.rugplot(data=tips, x="total_bill", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("rugplot (axes-level)", diag, {
        "draw_function": "ax.plot",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_add_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 5: barplot (axes-level)
try:
    ax = sns.barplot(data=tips, x="day", y="total_bill", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("barplot (axes-level)", diag, {
        "kind": "bar",
        "stat": "EstimateAggregator",
        "orient": "not_none",
        "draw_function": "ax.bar",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 6: stripplot (axes-level)
try:
    ax = sns.stripplot(data=tips, x="day", y="total_bill", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("stripplot (axes-level)", diag, {
        "draw_function": "ax.scatter",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 7: boxplot (axes-level)
try:
    ax = sns.boxplot(data=tips, x="day", y="total_bill", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("boxplot (axes-level)", diag, {
        "draw_function": "ax.bxp",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 8: violinplot (axes-level)
try:
    ax = sns.violinplot(data=tips, x="day", y="total_bill", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("violinplot (axes-level)", diag, {
        "stat": "KDE",
        "stat_params": "not_empty",
        "orient": "not_none",
        "draw_function": "ax.fill",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 9: pointplot (axes-level)
try:
    ax = sns.pointplot(data=tips, x="day", y="total_bill", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("pointplot (axes-level)", diag, {
        "stat": "EstimateAggregator",
        "orient": "not_none",
        "draw_function": "ax.plot",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 10: countplot (axes-level)
try:
    ax = sns.countplot(data=tips, x="day", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("countplot (axes-level)", diag, {
        "stat": "EstimateAggregator",
        "orient": "not_none",
        "draw_function": "ax.bar",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 11: scatterplot (axes-level)
try:
    ax = sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("scatterplot (axes-level)", diag, {
        "kind": "scatter",
        "draw_function": "ax.scatter",
        "legend_source": "hue+size+style",
        "legend_method": "add_legend_data",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 12: lineplot (axes-level)
try:
    ax = sns.lineplot(data=tips, x="day", y="total_bill", hue="sex")
    diag = get_diagnostics(ax)
    ok = check_diag("lineplot (axes-level)", diag, {
        "kind": "line",
        "stat": "EstimateAggregator",
        "draw_function": "ax.plot",
        "group_keys": ">0",
        "legend_source": "hue+size+style",
        "legend_method": "add_legend_data",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 13: displot kde (figure-level)
try:
    g = sns.displot(data=tips, x="total_bill", hue="sex", kind="kde", fill=True)
    diag = get_diagnostics(g)
    ok = check_diag("displot kde (figure-level)", diag, {
        "kind": "kde",
        "stat": "KDE",
        "stat_params": "not_empty",
        "orient": "not_none",
        "draw_function": "ax.fill_between",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_add_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 14: catplot strip (figure-level)
try:
    g = sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="strip")
    diag = get_diagnostics(g)
    ok = check_diag("catplot strip (figure-level)", diag, {
        "kind": "strip",
        "draw_function": "ax.scatter",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 15: catplot box (figure-level)
try:
    g = sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="box")
    diag = get_diagnostics(g)
    ok = check_diag("catplot box (figure-level)", diag, {
        "kind": "box",
        "draw_function": "ax.bxp",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 16: catplot violin (figure-level)
try:
    g = sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="violin")
    diag = get_diagnostics(g)
    ok = check_diag("catplot violin (figure-level)", diag, {
        "kind": "violin",
        "stat": "KDE",
        "stat_params": "not_empty",
        "orient": "not_none",
        "draw_function": "ax.fill",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

# Test 17: catplot point (figure-level)
try:
    g = sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="point")
    diag = get_diagnostics(g)
    ok = check_diag("catplot point (figure-level)", diag, {
        "kind": "point",
        "stat": "EstimateAggregator",
        "orient": "not_none",
        "draw_function": "ax.plot",
        "group_keys": ">0",
        "legend_source": "hue_map",
        "legend_method": "_configure_legend",
    })
    all_passed = all_passed and ok
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    all_passed = False

print("\n" + "="*60)
if all_passed:
    print("All tests passed!")
else:
    print("Some tests FAILED!")
