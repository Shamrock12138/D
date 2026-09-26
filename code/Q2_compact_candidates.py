

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.physics import load_models
from src.q2.compact_classes import generate_compact_patterns
from src.q2.data_model import load_q2_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-stops", type=int, choices=(1, 2), default=2)
    parser.add_argument("--service", action="append", help="Optional service subset for a smoke test")
    parser.add_argument("--local-load-limit", type=int, default=24,
                        help="Search budget per service and UAV type before two-stop products")
    parser.add_argument("--dry-run", action="store_true", help="Generate in memory only")
    args = parser.parse_args()
    boxes = load_q2_data()["boxes"]
    classes, patterns, counts = generate_compact_patterns(
        boxes, load_models(), max_stops=args.max_stops, services=args.service,
        local_load_limit=args.local_load_limit,
        progress=lambda typ, stops, done, total: print(
            f"{typ} {stops}-stop combinations: {done}; patterns: {total}", flush=True))
    covered = set(counts["class_id"])
    all_classes_covered = covered == set(classes["class_id"])
    if not all_classes_covered:
        raise RuntimeError("Compact candidate pool does not cover every class")
    if args.dry_run:
        print({"n_boxes": len(boxes), "n_classes": len(classes),
               "n_patterns": len(patterns), "n_pattern_class_rows": len(counts),
               "local_load_limit": args.local_load_limit,
               "all_classes_covered": all_classes_covered})
        return
    data = Path(__file__).resolve().parent / "data"
    outputs = {
        "classes": data / "Q2_compact_classes.csv",
        "class_members": data / "Q2_compact_class_members.csv",
        "patterns": data / "Q2_compact_patterns.csv",
        "pattern_counts": data / "Q2_compact_pattern_counts.csv",
    }
    class_output = classes.drop(columns="box_ids")
    class_output.to_csv(outputs["classes"], index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"class_id": row.class_id, "box_id": box_id}
        for row in classes.itertuples(index=False) for box_id in row.box_ids
    ]).to_csv(outputs["class_members"], index=False, encoding="utf-8-sig")
    patterns.to_csv(outputs["patterns"], index=False, encoding="utf-8-sig")
    counts.to_csv(outputs["pattern_counts"], index=False, encoding="utf-8-sig")
    manifest = {
        "max_stops": args.max_stops, "services": args.service,
        "local_load_limit": args.local_load_limit,
        "n_boxes": len(boxes), "n_classes": len(classes),
        "n_patterns": len(patterns), "n_pattern_class_rows": len(counts),
        "all_classes_covered": all_classes_covered,
        "outputs_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                            for key, path in outputs.items()},
    }
    manifest_path = data / "Q2_compact_candidates_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
