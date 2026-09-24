"""对照 DEM 修复前后的航段参数，输出可追溯差异表。"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "data_before_dem_fix" / "route_parameter_all.csv"
NEW = PROJECT / "data" / "route_parameter_all.csv"
OUTPUT = PROJECT / "data" / "dem_route_fix_comparison.csv"


def main():
    old = pd.read_csv(OLD, encoding="utf-8-sig")
    new = pd.read_csv(NEW, encoding="utf-8-sig")
    comparison = old.merge(new, on=["from", "to"], suffixes=("_old", "_new"),
                           validate="one_to_one")
    if len(comparison) != len(old) or len(comparison) != len(new):
        raise ValueError("修复前后航段集合不同")
    comparison["delta_h_max"] = comparison["h_max_new"] - comparison["h_max_old"]
    comparison["delta_cruise_height"] = (
        comparison["cruise_height_new"] - comparison["cruise_height_old"]
    )
    if (comparison["distance_new"] - comparison["distance_old"]).abs().max() > 1e-8:
        raise ValueError("DEM 修复不应改变水平距离")
    comparison.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"已保存 {OUTPUT.name}，共 {len(comparison)} 条航段")
    print(comparison.reindex(comparison["delta_h_max"].abs().sort_values(
        ascending=False).index)[
        ["from", "to", "h_max_old", "h_max_new", "delta_h_max",
         "cruise_height_old", "cruise_height_new"]
    ].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
