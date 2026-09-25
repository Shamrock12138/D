"""Q2 class-count closed loop: subset smoke or saved 80-box feasible check."""

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import pandas as pd
import ortools
from ortools.sat.python import cp_model

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.physics import load_models
from src.q2.compact_classes import (
    build_box_classes, build_compact_master, class_timeliness, decode_box_deliveries,
    generate_compact_patterns, materialize_selected_sorties,
    select_compact_patterns,
)
from src.q2.cp_sat_scheduler import _build_model, _deadlines, _decode_resources, _resource_ids
from src.q2.data_model import load_q2_data
from src.q2.route_evaluator import _box_data, evaluate_route


DATA = Path(__file__).resolve().parent / "data"


def _load_saved_compact_candidates(boxes):
    """Load Step 1 artifacts and reject stale or mismatched class identities."""
    manifest = json.loads((DATA / "Q2_compact_candidates_manifest.json").read_text(
        encoding="utf-8"))
    names = {"classes": "Q2_compact_classes.csv",
             "class_members": "Q2_compact_class_members.csv",
             "patterns": "Q2_compact_patterns.csv",
             "pattern_counts": "Q2_compact_pattern_counts.csv"}
    for key, name in names.items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        if actual != manifest["outputs_sha256"][key]:
            raise ValueError(f"Compact candidate artifact changed: {name}")
    classes, _ = build_box_classes(boxes)
    saved_classes = pd.read_csv(DATA / names["classes"], encoding="utf-8-sig")
    pd.testing.assert_frame_equal(classes.drop(columns="box_ids"), saved_classes,
                                  check_dtype=False)
    members = pd.DataFrame([
        {"class_id": row.class_id, "box_id": box_id}
        for row in classes.itertuples(index=False) for box_id in row.box_ids
    ])
    saved_members = pd.read_csv(DATA / names["class_members"], encoding="utf-8-sig")
    pd.testing.assert_frame_equal(members, saved_members, check_dtype=False)
    patterns = pd.read_csv(DATA / names["patterns"], encoding="utf-8-sig")
    counts = pd.read_csv(DATA / names["pattern_counts"], encoding="utf-8-sig")
    if (len(boxes) != int(manifest["n_boxes"])
            or len(classes) != int(manifest["n_classes"])):
        raise ValueError("Compact candidate manifest has stale box/class counts")
    return classes, patterns, counts


def _solver(time_limit_s, workers):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = 2026
    return solver


def _resources(data):
    uav_ids, battery_ids = _resource_ids(data["uavs"], data["batteries"])
    specs = pd.read_csv(DATA / "运输无人机_机型参数.csv", encoding="utf-8-sig")
    capacity = dict(zip(specs["type"].astype(str), specs["E_use"].astype(float)))
    charge_full = data["batteries"].groupby("type")["full_charge_time"].first().to_dict()
    return uav_ids, battery_ids, capacity, charge_full


def _physical_cache_equivalent(patterns, pattern_counts, classes, boxes, models):
    """Re-evaluate retained patterns without the generator's physics cache."""
    class_rows = classes.set_index("class_id")
    grouped = pattern_counts.groupby("pattern_id")
    lookup = _box_data(boxes)
    for row in patterns.itertuples(index=False):
        visit = str(row.visit_order).split(">")
        deliveries = {site: [] for site in visit}
        for item in grouped.get_group(row.pattern_id).itertuples(index=False):
            cls = class_rows.loc[item.class_id]
            deliveries[cls.service].extend(cls.box_ids[:int(item.count)])
        result = evaluate_route(models[row.uav_type], visit, deliveries,
                                box_lookup=lookup)
        if not result["feasible"]:
            return False
        if abs(float(result["duration_s"]) - float(row.duration_s)) > 1e-6:
            return False
        if abs(float(result["energy_kwh"]) - float(row.energy_kWh)) > 1e-8:
            return False
        for item in grouped.get_group(row.pattern_id).itertuples(index=False):
            member = class_rows.loc[item.class_id].box_ids[0]
            if abs(float(result["delivery_offsets"][member])
                   - float(item.delivery_offset_s)) > 1e-6:
                return False
    return True


def _fixed_transport_schedule(tasks, deliveries, boxes, resources, master_starts,
                              time_limit_s, workers):
    """Check the master's exact sortie times in the established Q2 resource model."""
    grouped = deliveries.groupby("task_id")
    task_boxes = {str(tid): tuple(group["box_id"].astype(str)) for tid, group in grouped}
    task_offsets = {str(tid): dict(zip(group["box_id"].astype(str),
                                       group["delivery_offset_s"].astype(float)))
                    for tid, group in grouped}
    deadlines = _deadlines(boxes)
    work = tasks.copy().reset_index(drop=True)
    work["hard_latest_start_s"] = work["latest_start_s"]
    uav_ids, battery_ids, capacity, charge_full = resources
    built = _build_model(work, task_boxes, task_offsets, deadlines, uav_ids,
                         battery_ids, capacity, charge_full, 36000)
    model, selected, starts, _, _, _, metadata = built
    task_ids = work["task_id"].astype(str).tolist()
    if set(task_ids) != set(master_starts) or len(task_ids) != len(set(task_ids)):
        raise ValueError("Master start times must match each unique sortie ID")
    for i, chosen in enumerate(selected):
        model.Add(chosen == 1)
        start_time = master_starts[task_ids[i]]
        if int(start_time) != start_time:
            raise ValueError(f"Non-integer master start time for {task_ids[i]}")
        model.Add(starts[i] == int(start_time))
    solver = _solver(time_limit_s, workers)
    status = solver.Solve(model)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return solver.StatusName(status), None, None
    start_map = {str(row.task_id): int(solver.Value(starts[i]))
                 for i, row in work.iterrows()}
    if start_map != {tid: int(start) for tid, start in master_starts.items()}:
        raise AssertionError("Independent Q2 check changed master start times")
    records = [dict(task_id=meta["task_id"], uav_type=meta["uav_type"],
                    start_time_s=start_map[meta["task_id"]],
                    flight_duration_s=meta["flight_duration_s"],
                    battery_duration_s=meta["battery_duration_s"],
                    charge_s=meta["charge_s"])
               for meta in metadata]
    schedule = _decode_resources(work, records, uav_ids, battery_ids)
    flight_release = {item["task_id"]: item["start_time_s"]
                      + item["flight_duration_s"] for item in records}
    battery_release = {item["task_id"]: item["start_time_s"]
                       + item["battery_duration_s"] for item in records}
    schedule["flight_release_s"] = schedule["task_id"].map(flight_release)
    schedule["battery_release_s"] = schedule["task_id"].map(battery_release)
    return solver.StatusName(status), start_map, schedule


def _nonoverlap(schedule, resource, start, end):
    for _, group in schedule.sort_values(start).groupby(resource):
        if (group[start].iloc[1:].to_numpy()
                < group[end].iloc[:-1].to_numpy() - 1e-9).any():
            return False
    return True


def _q3_communication_check(tasks, deliveries):
    """Check that materialized compact sorties enter the existing Q3 profile."""
    from src.q3.communication.direct_profile import DirectProfileCache, required_segment_keys
    from src.q3.transport.candidate_loader import TransportTaskTemplate
    from src.q3.transport.communication_summary import assemble_task_profile, summarize_profile
    from src.q3.trajectory_generator import load_box_services

    grouped = deliveries.groupby("task_id")
    templates = []
    for row in tasks.itertuples(index=False):
        group = grouped.get_group(row.task_id)
        visit = str(row.visit_order).split(">")
        templates.append(TransportTaskTemplate(
            task_id=str(row.task_id), uav_type=str(row.uav_type),
            n_stops=int(row.n_stops), visit_order=visit,
            route=["O01", *visit, "O01"],
            boxes=group["box_id"].astype(str).tolist(),
            delivery_offsets=dict(zip(group["box_id"].astype(str),
                                      group["delivery_offset_s"].astype(float))),
            energy_kWh=float(row.energy_kWh), duration_s=float(row.duration_s),
            end_SOC=float(row.end_SOC),
            has_hard_deadline=bool(row.has_hard_deadline),
            latest_start_s=float(row.latest_start_s),
        ))
    cache = DirectProfileCache(dt=10.0)
    try:
        cache.build_nodes()
        cache.build_segments(required_segment_keys(templates), verbose=False)
        services = load_box_services()
        summary = [summarize_profile(template, assemble_task_profile(template, cache,
                                                                       services))
                   for template in templates]
    finally:
        cache.close()
    return {"profiles": len(summary), "gaps": sum(row["gap_count"] for row in summary),
            "needs_relay": sum(row["needs_relay"] for row in summary),
            "all_profiles_complete": len(summary) == len(templates)}


def _load_current_anchor_seed(objective, top_k):
    """Prefer the same objective's old anchor over the generic feasible seed."""
    candidate_manifest = DATA / "Q2_compact_candidates_manifest.json"
    candidate_sha = hashlib.sha256(candidate_manifest.read_bytes()).hexdigest()

    prefixes = [
        f"Q2_anchor_{objective}",
        "Q2_compact_feasible",
    ]

    for prefix in prefixes:
        selected_path = DATA / f"{prefix}_selected.csv"
        manifest_path = DATA / f"{prefix}_manifest.json"

        if not selected_path.exists() or not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest.get("candidate_manifest_sha256") != candidate_sha:
            continue

        if int(manifest.get("top_k", -1)) != int(top_k):
            continue

        return (
            pd.read_csv(selected_path, encoding="utf-8-sig"),
            prefix,
        )

    return None, None


def run_smoke(services=("S001", "S002"), top_k=3, master_time_s=30,
              transport_time_s=20, workers=1, q3_comm=False,
              q3_relay=False, objective="N", full_candidates=False,
              save_feasible=False, save_anchor=False):
    """Return an evidence report; save only explicitly requested full results."""
    data = load_q2_data()
    if (save_feasible or save_anchor) and not full_candidates:
        raise ValueError("Full solution outputs require saved full candidates")
    if save_feasible and save_anchor:
        raise ValueError("Choose either feasible or anchor output names")
    if full_candidates:
        boxes = data["boxes"].copy()
        services = tuple(sorted(boxes["service"].unique()))
    else:
        boxes = data["boxes"].loc[data["boxes"]["service"].isin(services)].copy()
    if boxes.empty:
        raise ValueError("No boxes in requested services")
    models = load_models()
    if full_candidates:
        classes, patterns, counts = _load_saved_compact_candidates(boxes)
    else:
        classes, patterns, counts = generate_compact_patterns(
            data["boxes"], models, max_stops=2, services=services)
        classes = classes.loc[classes["service"].isin(services)].reset_index(drop=True)
    full_pattern_count = len(patterns)
    patterns, counts = select_compact_patterns(patterns, counts, classes, top_k)
    cache_equivalent = _physical_cache_equivalent(
        patterns, counts, classes, boxes, models)
    resources = _resources(data)
    model, slots, chosen, starts = build_compact_master(
        patterns, counts, classes, *resources, horizon_s=36000,
        objective=objective)
    seed_source = None

    if full_candidates and objective != "N":
        seed, seed_source = _load_current_anchor_seed(objective, top_k)

        if seed is not None:
            seed_starts = dict(
                zip(
                    seed["task_id"].astype(str),
                    seed["start_time_s"].astype(int),
                )
            )

            for i, slot in enumerate(slots):
                sortie_id = slot["sortie_id"]

                model.AddHint(
                    chosen[i],
                    int(sortie_id in seed_starts)
                )

                if sortie_id in seed_starts:
                    model.AddHint(
                        starts[i],
                        seed_starts[sortie_id]
                    )
    master = _solver(master_time_s, workers)
    master_status = master.Solve(model)
    report = {"services": list(services), "boxes": len(boxes),
              "candidate_source": "saved_full" if full_candidates else "generated_subset",
              "objective": objective,
              "top_k": top_k, "workers": workers,
              "master_time_limit_s": master_time_s,
              "transport_time_limit_s": transport_time_s,
              "python_version": platform.python_version(),
              "ortools_version": ortools.__version__,
              "classes": len(classes), "physical_patterns": full_pattern_count,
              "retained_patterns": len(patterns), "sortie_slots": len(slots),
              "physical_cache_equivalence": cache_equivalent,
              "master_status": master.StatusName(master_status),
              "master_wall_time_s": master.WallTime(), "seed": 2026,
              "seed_source": seed_source,
              "input_sha256": hashlib.sha256(
                  (DATA / "物资需求.csv").read_bytes()).hexdigest()}
    if master_status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        report["all_pass"] = False
        return report
    report["master_objective_value"] = master.ObjectiveValue()
    report["master_best_bound"] = master.BestObjectiveBound()
    objective_value = float(master.ObjectiveValue())
    best_bound = float(master.BestObjectiveBound())
    relative_gap = abs(objective_value - best_bound) / max(
        1.0,
        abs(objective_value),
    )
    report["master_relative_gap"] = relative_gap
    report["anchor_optimal"] = (
        master_status == cp_model.OPTIMAL
    )
    sorties = [dict(slot, start_time_s=int(master.Value(starts[i])))
               for i, (slot, x) in enumerate(zip(slots, chosen)) if master.Value(x)]
    master_starts = {sortie["sortie_id"]: sortie["start_time_s"]
                     for sortie in sorties}
    master_f1 = class_timeliness(sorties, classes, counts)
    pattern_durations = dict(zip(patterns["pattern_id"], patterns["duration_s"]))
    master_cmax = max(sortie["start_time_s"]
                      + float(pattern_durations[sortie["pattern_id"]])
                      for sortie in sorties)
    task_table, delivery_table = materialize_selected_sorties(
        sorties, patterns, counts, classes)
    transport_status, start_map, schedule = _fixed_transport_schedule(
        task_table, delivery_table, boxes, resources, master_starts,
        transport_time_s, workers)
    report.update({"selected_sorties": len(sorties),
                   "repeated_patterns": sum(
                       sum(s["pattern_id"] == pid for s in sorties) > 1
                       for pid in {s["pattern_id"] for s in sorties}),
                   "transport_status": transport_status})
    if schedule is None:
        report["all_pass"] = False
        return report
    for sortie in sorties:
        sortie["start_time_s"] = start_map[sortie["sortie_id"]]
    deliveries = decode_box_deliveries(sorties, classes, counts)
    deadlines = _deadlines(boxes)
    box_rows = boxes.set_index("box_id")
    hard_ok = all(float(row.delivery_time_s) <= math.floor(deadlines[row.box_id]) + 1e-6
                  for row in deliveries.itertuples(index=False)
                  if math.isfinite(deadlines[row.box_id]))
    first_ok = all(float(row.delivery_time_s) <= math.floor(
                       float(box_rows.loc[row.box_id, "first_deadline"])) + 1e-6
                   for row in deliveries.itertuples(index=False)
                   if bool(box_rows.loc[row.box_id, "is_first_batch"])
                   and pd.notna(box_rows.loc[row.box_id, "first_deadline"]))
    medical_ok = all(float(row.delivery_time_s) <= math.floor(
                         float(box_rows.loc[row.box_id, "expected_time"])) + 1e-6
                     for row in deliveries.itertuples(index=False)
                     if box_rows.loc[row.box_id, "cargo_type"] == "医疗物资"
                     and pd.notna(box_rows.loc[row.box_id, "expected_time"]))
    box_class = {box: row.class_id for row in classes.itertuples(index=False)
                 for box in row.box_ids}
    checks = {
        "physical_cache_equivalence": cache_equivalent,
        "class_quantity_conservation": all(
            sum(s["class_counts"].get(row.class_id, 0) for s in sorties) == row.count
            for row in classes.itertuples(index=False)),
        "every_box_exactly_once": (len(deliveries) == len(boxes)
                                   and set(deliveries["box_id"]) == set(boxes["box_id"])
                                   and deliveries["box_id"].is_unique),
        "box_class_identity": all(box_class[row.box_id] == row.class_id
                                  for row in deliveries.itertuples(index=False)),
        "hard_deadlines": hard_ok,
        "first_batch_deadlines": first_ok,
        "medical_deadlines": medical_ok,
        "flight_horizon": bool((schedule["end_time_s"] <= 36000 + 1e-6).all()),
        "uav_nonoverlap": _nonoverlap(schedule, "uav_id", "start_time_s",
                                      "flight_release_s"),
        "battery_nonoverlap": _nonoverlap(schedule, "battery_id", "start_time_s",
                                           "battery_release_s"),
        "master_start_times_preserved": start_map == master_starts,
        "master_F1_preserved": math.isclose(
            class_timeliness(sorties, classes, counts), master_f1,
            rel_tol=0.0, abs_tol=1e-6),
        "master_Cmax_preserved": math.isclose(
            float(schedule["end_time_s"].max()), master_cmax,
            rel_tol=0.0, abs_tol=1e-6),
        "delivery_time_mapping": all(
            abs(row.delivery_time_s - row.start_time_s - row.delivery_offset_s) < 1e-6
            for row in deliveries.itertuples(index=False)),
    }
    report.update({"checks": checks, "all_pass": all(checks.values()),
                   "master_F1_weighted_lateness": master_f1,
                   "master_cmax_s": master_cmax,
                   "transport_cmax_s": float(schedule["end_time_s"].max()),
                   "transport_energy_kWh": float(schedule["energy_kWh"].sum()),
                   "F1_weighted_lateness": class_timeliness(sorties, classes, counts)})
    report["objectives"] = {
        "F1": float(report["F1_weighted_lateness"]),
        "Cmax_s": float(report["transport_cmax_s"]),
        "E_kWh": float(report["transport_energy_kWh"]),
        "N": int(report["selected_sorties"]),
    }
    report["anchor_ready"] = bool(
        report["all_pass"]
        and report["anchor_optimal"]
    )
    if (save_feasible or save_anchor) and report["all_pass"]:
        if len(boxes) != 80:
            raise ValueError("Full feasible outputs require all 80 boxes")
        final_tasks, final_deliveries = materialize_selected_sorties(
            sorties, patterns, counts, classes)
        selected = final_tasks.merge(
            schedule[["task_id", "uav_id", "battery_id", "start_time_s"]],
            on="task_id", validate="one_to_one")
        prefix = f"Q2_anchor_{objective}" if save_anchor else "Q2_compact_feasible"
        outputs = {key: DATA / f"{prefix}_{key}.csv"
                   for key in ("selected", "schedule", "deliveries")}
        selected.to_csv(outputs["selected"], index=False, encoding="utf-8-sig")
        schedule.to_csv(outputs["schedule"], index=False, encoding="utf-8-sig")
        final_deliveries.to_csv(outputs["deliveries"], index=False,
                                encoding="utf-8-sig")
        report["output_sha256"] = {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in outputs.items()
        }
        report["candidate_manifest_sha256"] = hashlib.sha256(
            (DATA / "Q2_compact_candidates_manifest.json").read_bytes()).hexdigest()
        manifest_path = DATA / f"{prefix}_manifest.json"
        manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        report["manifest_file"] = manifest_path.name
    if q3_comm and report["all_pass"]:
        final_tasks, final_deliveries = materialize_selected_sorties(
            sorties, patterns, counts, classes)
        report["q3_communication"] = _q3_communication_check(
            final_tasks, final_deliveries)
    if q3_relay and report["all_pass"]:
        from src.q3.compact_smoke_relay import run_compact_relay_smoke
        final_tasks, final_deliveries = materialize_selected_sorties(
            sorties, patterns, counts, classes)
        report["q3_relay"] = run_compact_relay_smoke(
            final_tasks, final_deliveries, boxes, resources, start_map,
            time_limit_s=transport_time_s, workers=workers)
        relay_report = report["q3_relay"]
        report["minimal_q3_all_pass"] = bool(
            relay_report.get("status") in ("FEASIBLE", "OPTIMAL")
            and relay_report.get("joint_validation_pass")
            and relay_report.get("fine_communication_1s", {}).get("all_pass"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", action="append", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--master-time", type=float, default=30)
    parser.add_argument("--transport-time", type=float, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--objective", choices=("F1", "Cmax", "E", "N"),
                        default="N")
    parser.add_argument("--full-candidates", action="store_true",
                        help="Use saved 80-box compact candidate files")
    parser.add_argument("--save-feasible", action="store_true",
                        help="Write 80-box feasible outputs after all checks pass")
    parser.add_argument("--save-anchor", action="store_true",
                        help="Write objective-specific 80-box anchor outputs")
    parser.add_argument("--q3-comm", action="store_true",
                        help="Also evaluate Q3 direct-link profiles for selected sorties")
    parser.add_argument("--q3-relay", action="store_true",
                        help="Also test in-memory gap/RP/Relay joint scheduling")
    args = parser.parse_args()
    print(json.dumps(run_smoke(tuple(args.service) if args.service else ("S001", "S002"),
                               args.top_k, args.master_time, args.transport_time,
                               args.workers, args.q3_comm, args.q3_relay,
                               args.objective, args.full_candidates,
                               args.save_feasible, args.save_anchor),
                     ensure_ascii=False, indent=2))