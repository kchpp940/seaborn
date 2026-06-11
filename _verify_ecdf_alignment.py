"""End-to-end verification that legacy ECDF and objects ECDF produce identical
results across all stat / complementary / common_norm / weight scenarios."""

import numpy as np
import pandas as pd
from seaborn._core.groupby import GroupBy
from seaborn._stats.ecdf import ECDF as ECDFStat
from seaborn._statistics import ECDF as LegacyECDF


def build_test_data(rng, n=200):
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "hue": rng.choice(["a", "b"], n),
        "weight": rng.uniform(0.5, 2.0, n),
    })
    # Add some edge-case rows
    df = pd.concat([df, pd.DataFrame([
        {"x": np.nan, "hue": "a", "weight": 1.0},
        {"x": np.inf, "hue": "b", "weight": 1.0},
        {"x": 1.5, "hue": "a", "weight": 0.0},
        {"x": 2.5, "hue": "b", "weight": -1.0},
    ])], ignore_index=True)
    return df


class Scale:
    scale_type = "continuous"


def run_legacy(df, stat, complementary, common_norm, use_weights=False):
    """Run legacy _statistics.ECDF per hue group and collect results."""
    est = LegacyECDF(stat=stat, complementary=complementary)
    results = {}
    if common_norm:
        if use_weights:
            whole_weight = float(
                np.where(np.isfinite(df["weight"].values) & (df["weight"].values > 0),
                         df["weight"].values, 0.0).sum()
            )
        else:
            whole_weight = float(
                np.isfinite(df["x"].values).sum()
            )
    else:
        whole_weight = None

    for hue_val, part in df.groupby("hue"):
        x = part["x"].values
        w = part["weight"].values if use_weights else None
        norm_total = whole_weight if common_norm else None
        y, x_vals = est(x, weights=w, norm_total=norm_total)
        results[hue_val] = (y, x_vals)
    return results


def run_objects(df, stat, complementary, common_norm, use_weights=False):
    """Run objects ECDF stat and collect results per hue group."""
    gb = GroupBy(["hue"])
    ecdf = ECDFStat(
        stat=stat,
        complementary=complementary,
        common_norm=common_norm,
    )
    data = df.copy()
    if not use_weights:
        data = data.drop(columns=["weight"])
    res = ecdf(data, gb, "x", {"x": Scale()})
    results = {}
    for hue_val, part in res.groupby("hue"):
        part = part.sort_values("x")
        results[hue_val] = (part[stat].values, part["x"].values)
    return results


def compare(legacy_res, objects_res, label):
    """Compare legacy vs objects outputs, return list of issues."""
    issues = []
    for hue in sorted(set(legacy_res) | set(objects_res)):
        if hue not in legacy_res:
            issues.append(f"  [{label}] hue={hue} missing from legacy")
            continue
        if hue not in objects_res:
            issues.append(f"  [{label}] hue={hue} missing from objects")
            continue

        leg_y, leg_x = legacy_res[hue]
        obj_y, obj_x = objects_res[hue]

        # Compare finite x values and corresponding y values
        leg_finite = np.isfinite(leg_x)
        obj_finite = np.isfinite(obj_x)

        # x values should match (sorted finite values)
        leg_x_finite = np.sort(leg_x[leg_finite])
        obj_x_finite = np.sort(obj_x[obj_finite])
        if not np.allclose(leg_x_finite, obj_x_finite):
            issues.append(
                f"  [{label}] hue={hue} x values differ: "
                f"len(leg)={len(leg_x_finite)} len(obj)={len(obj_x_finite)}"
            )
            continue

        # y values should match (sorted order is same as x)
        # We need to align by x value since the order may differ
        leg_map = dict(zip(leg_x, leg_y))
        obj_map = dict(zip(obj_x, obj_y))

        # Check the -inf anchor
        if -np.inf in leg_map and -np.inf in obj_map:
            if not np.isclose(leg_map[-np.inf], obj_map[-np.inf]):
                issues.append(
                    f"  [{label}] hue={hue} y at -inf differ: "
                    f"leg={leg_map[-np.inf]} obj={obj_map[-np.inf]}"
                )

        # Check all finite values
        for x_val in leg_x_finite:
            ly = leg_map[x_val]
            oy = obj_map[x_val]
            if not np.isclose(ly, oy, rtol=1e-10):
                issues.append(
                    f"  [{label}] hue={hue} y at x={x_val} differ: "
                    f"leg={ly} obj={oy}"
                )
                break
    return issues


def main():
    rng = np.random.default_rng(42)
    df = build_test_data(rng)

    all_issues = []
    configs = []
    for stat in ["proportion", "percent", "count"]:
        for complementary in [False, True]:
            for common_norm in [False, True]:
                for use_weights in [False, True]:
                    configs.append((stat, complementary, common_norm, use_weights))

    for stat, complementary, common_norm, use_weights in configs:
        label = (
            f"stat={stat} comp={complementary} "
            f"common_norm={common_norm} weights={use_weights}"
        )
        leg_res = run_legacy(df, stat, complementary, common_norm, use_weights)
        obj_res = run_objects(df, stat, complementary, common_norm, use_weights)
        issues = compare(leg_res, obj_res, label)
        all_issues.extend(issues)

    if all_issues:
        print(f"Found {len(all_issues)} issues:")
        for issue in all_issues:
            print(issue)
        return False

    print(f"All {len(configs)} configurations match perfectly!")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
