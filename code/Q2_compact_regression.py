










import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.physics import load_models
from src.q2.compact_classes import (
    build_box_classes,
    count_signature,
    enumerate_service_loads,
    pattern_signature,
    select_service_loads,
)
from src.q2.data_model import load_q2_data

DATA = Path(__file__).resolve().parent / "data"


BENCHMARKS = {
    "N": "P001",
    "E": "P003",
    "Cmax": "P027",
}


def load_legacy_solution(solution_id):

    tasks = pd.read_csv(
        DATA / f"Q2_moead_tasks_{solution_id}.csv",
        encoding="utf-8-sig",
    )

    deliveries = pd.read_csv(
        DATA / f"Q2_moead_delivery_check_{solution_id}.csv",
        encoding="utf-8-sig",
    )

    return tasks, deliveries


def legacy_solution_signatures(solution_id, boxes):

    classes, box_to_class = build_box_classes(boxes)

    tasks, deliveries = load_legacy_solution(solution_id)

    result = []

    for task in tasks.itertuples(index=False):
        tid = str(task.task_id)

        box_ids = deliveries.loc[
            deliveries["task_id"].astype(str) == tid,
            "box_id",
        ].astype(str)

        counts = Counter(
            box_to_class[box_id]
            for box_id in box_ids
        )

        sig = pattern_signature(
            task.uav_type,
            task.visit_order,
            counts,
        )

        result.append({
            "legacy_task_id": tid,
            "signature": sig,
            "uav_type": task.uav_type,
            "visit_order": task.visit_order,
            "class_counts": dict(counts),
            "n_boxes": sum(counts.values()),
            "start_time_s": float(task.start_time_s),
            "energy_kWh": float(task.energy_kWh),
            "duration_s": float(task.duration_s),
        })

    return result, classes, box_to_class


def build_pattern_signature_index(patterns, pattern_counts):

    grouped = pattern_counts.groupby("pattern_id")

    index = {}

    for row in patterns.itertuples(index=False):
        group = grouped.get_group(row.pattern_id)

        counts = {
            str(item.class_id): int(item.count)
            for item in group.itertuples(index=False)
        }

        sig = pattern_signature(
            row.uav_type,
            row.visit_order,
            counts,
        )

        index.setdefault(sig, []).append(str(row.pattern_id))

    return index


def split_counts_by_service(class_counts, classes):

    class_service = dict(
        zip(classes["class_id"], classes["service"])
    )

    result = {}

    for class_id, amount in class_counts.items():
        service = class_service[class_id]
        result.setdefault(service, {})
        result[service][class_id] = amount

    return result


def local_load_survives(class_counts, uav_type, classes, models, local_load_limit):
    



    max_mass = float(models[uav_type].u["Q_g"])
    max_volume = float(models[uav_type].u["V_g"])

    by_service = split_counts_by_service(class_counts, classes)

    for service, wanted_counts in by_service.items():

        all_loads = list(
            enumerate_service_loads(
                classes,
                service,
                max_mass,
                max_volume,
            )
        )

        selected = select_service_loads(
            all_loads,
            classes,
            service,
            max_mass,
            max_volume,
            local_load_limit,
        )

        wanted_sig = count_signature(wanted_counts)

        selected_sigs = {
            count_signature(counts)
            for counts, _, _ in selected
        }

        if wanted_sig not in selected_sigs:
            return False, service

    return True, None


def compute_legacy_metrics(tasks, deliveries):

    n = len(tasks)
    e = float(tasks["energy_kWh"].sum())
    cmax = max(
        float(row.start_time_s) + float(row.duration_s)
        for row in tasks.itertuples(index=False)
    )
    return {"N": n, "E": e, "Cmax": cmax}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--local-load-limit", type=int, default=24)
    args = parser.parse_args()

    boxes = load_q2_data()["boxes"]
    models = load_models()

    manifest = json.loads(
        (DATA / "Q2_compact_candidates_manifest.json").read_text(encoding="utf-8")
    )

    patterns = pd.read_csv(
        DATA / "Q2_compact_patterns.csv", encoding="utf-8-sig"
    )
    pattern_counts = pd.read_csv(
        DATA / "Q2_compact_pattern_counts.csv", encoding="utf-8-sig"
    )

    generated_index = build_pattern_signature_index(patterns, pattern_counts)
    generated_count = len(patterns)

    classes, _ = build_box_classes(boxes)

    retained_patterns, retained_counts = _select_from_saved(
        patterns, pattern_counts, classes, args.top_k
    )
    retained_index = build_pattern_signature_index(
        retained_patterns, retained_counts
    )
    retained_count = len(retained_patterns)

    print(f"Generated patterns  : {generated_count}")
    print(f"Retained patterns   : {retained_count}")
    print(f"Local load limit    : {args.local_load_limit}")
    print(f"Top K               : {args.top_k}")
    print()

    rows = []

    for objective, solution_id in BENCHMARKS.items():

        sigs, classes_ref, box_to_class = legacy_solution_signatures(
            solution_id, boxes
        )

        tasks, deliveries = load_legacy_solution(solution_id)
        metrics = compute_legacy_metrics(tasks, deliveries)

        print(f"=== {objective} ({solution_id}) ===")
        print(f"  Known:  N={metrics['N']}, E={metrics['E']:.4f}, Cmax={metrics['Cmax']:.0f}")
        print(f"  Total sorties: {len(sigs)}")

        counts = {"SURVIVED": 0, "FINAL_PATTERN_DROP": 0,
                  "LOCAL_LOAD_DROP": 0, "GENERATION_DROP": 0}

        for entry in sigs:
            sig = entry["signature"]

            if sig in retained_index:
                reason = "SURVIVED"
            elif sig in generated_index:
                reason = "FINAL_PATTERN_DROP"
            else:
                survived, bad_service = local_load_survives(
                    entry["class_counts"],
                    entry["uav_type"],
                    classes_ref,
                    models,
                    args.local_load_limit,
                )

                if not survived:
                    reason = "LOCAL_LOAD_DROP"
                else:
                    reason = "GENERATION_DROP"

            counts[reason] += 1

            rows.append({
                "benchmark": objective,
                "solution_id": solution_id,
                "legacy_task_id": entry["legacy_task_id"],
                "uav_type": entry["uav_type"],
                "visit_order": entry["visit_order"],
                "n_boxes": entry["n_boxes"],
                "generated": sig in generated_index,
                "retained": sig in retained_index,
                "drop_stage": reason,
            })

        print(f"  SURVIVED            = {counts['SURVIVED']}")
        print(f"  FINAL_PATTERN_DROP  = {counts['FINAL_PATTERN_DROP']}")
        print(f"  LOCAL_LOAD_DROP     = {counts['LOCAL_LOAD_DROP']}")
        print(f"  GENERATION_DROP     = {counts['GENERATION_DROP']}")
        print()

    out_path = DATA / "Q2_compact_regression.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path} ({len(rows)} rows)")


def _select_from_saved(patterns, pattern_counts, classes, top_k):

    from src.q2.compact_classes import select_compact_patterns

    selected, selected_counts = select_compact_patterns(
        patterns,
        pattern_counts,
        classes,
        top_k,
    )

    return selected, selected_counts


if __name__ == "__main__":
    main()