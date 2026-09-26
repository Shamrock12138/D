

import sys
import hashlib
import json
import platform
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "data_before_dem_fix" / "route_parameter_all.csv"
NEW = PROJECT / "data" / "route_parameter_all.csv"
OUTPUT = PROJECT / "data" / "dem_route_fix_comparison.csv"
MANIFEST = PROJECT / "data" / "dem_route_fix_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    MANIFEST.write_text(json.dumps({
        "python": platform.python_version(),
        "old_route_sha256": sha256(OLD),
        "new_route_sha256": sha256(NEW),
        "comparison_sha256": sha256(OUTPUT),
        "validation": "python -m unittest discover -s code/tests -p test_dem_route.py -v",
        "rebuild": "python code/src/build_route_matrices.py",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存 {OUTPUT.name}，共 {len(comparison)} 条航段")
    print(comparison.reindex(comparison["delta_h_max"].abs().sort_values(
        ascending=False).index)[
        ["from", "to", "h_max_old", "h_max_new", "delta_h_max",
         "cruise_height_old", "cruise_height_new"]
    ].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
