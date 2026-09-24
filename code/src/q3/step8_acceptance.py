"""Step 8.5: validate and freeze a completed Q3 minimum joint schedule."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from src.q3.cp_sat_scheduler import DATA, prepare_q3_problem, validate_q3_solution
from src.q3.objectives import evaluate_objectives


OUTPUTS = (
    "q3_joint_transport_schedule.csv",
    "q3_joint_relay_schedule.csv",
    "q3_joint_delivery_schedule.csv",
    "q3_joint_resource_summary.csv",
    "q3_step8_manifest.json",
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accept_step8(freeze=True):
    missing = [name for name in OUTPUTS if not (DATA / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Step8 outputs are incomplete: {missing}")
    manifest = json.loads((DATA / "q3_step8_manifest.json").read_text(encoding="utf-8"))
    status = manifest.get("status")
    if status not in ("FEASIBLE", "OPTIMAL") or manifest.get("validation", {}).get("all_pass") is not True:
        raise AssertionError(f"Step8 status/validation failed: {status}, {manifest.get('validation')}")
    for name, recorded in manifest.get("input_sha256", {}).items():
        if _hash(DATA / name) != recorded:
            raise AssertionError(f"Step8 input changed since solve: {name}")
    tier = manifest.get("tier")
    if tier not in ("tier1", "tier2", "all"):
        raise ValueError(f"Unknown Step8 relay tier: {tier}")
    problem = prepare_q3_problem(tier=tier)
    transport = pd.read_csv(DATA / OUTPUTS[0], encoding="utf-8-sig")
    relay = pd.read_csv(DATA / OUTPUTS[1], encoding="utf-8-sig")
    delivery = pd.read_csv(DATA / OUTPUTS[2], encoding="utf-8-sig")
    summary = pd.read_csv(DATA / OUTPUTS[3], encoding="utf-8-sig")
    if len(summary) != 1:
        raise AssertionError("Step8 resource summary must have exactly one row")
    cmax = float(summary.iloc[0]["joint_Cmax_s"])
    validation = validate_q3_solution(problem, transport, relay, cmax)
    selected_ids = set(transport["task_id"].astype(str))
    expected_deliveries = problem["deliveries"].loc[
        problem["deliveries"]["task_id"].astype(str).isin(selected_ids)
    ]
    validation["checks"]["delivery_schedule_matches_selected_tasks"] = (
        len(delivery) == len(expected_deliveries)
        and set(zip(delivery["task_id"].astype(str), delivery["box_id"].astype(str)))
        == set(zip(expected_deliveries["task_id"].astype(str), expected_deliveries["box_id"].astype(str)))
    )
    start_by_task = dict(zip(transport["task_id"].astype(str), transport["start_time_s"]))
    expected_times = {
        (str(row.task_id), str(row.box_id)): start_by_task[str(row.task_id)] + float(row.delivery_offset_s)
        for row in expected_deliveries.itertuples(index=False)
    }
    validation["checks"]["delivery_times_match_offsets"] = all(
        (str(row.task_id), str(row.box_id)) in expected_times
        and abs(float(row.delivery_time_s) - expected_times[(str(row.task_id), str(row.box_id))]) <= 1e-6
        for row in delivery.itertuples(index=False)
    )
    validation["checks"]["transport_resource_ids_valid"] = all(
        set(group["uav_id"].astype(str)) <= set(problem["uav_ids"][typ])
        and set(group["battery_id"].astype(str)) <= set(problem["battery_ids"][typ])
        for typ, group in transport.groupby("uav_type")
    )
    validation["checks"]["relay_resource_ids_valid"] = (
        set(relay["relay_uav_id"].astype(str)) <= {"R01", "R02"}
        and set(relay["energy_component_id"].astype(str)) <= {f"E0{i}" for i in range(1, 7)}
    )
    allowed = problem["relay"][["gap_id", "task_id", "candidate_id"]].astype(str)
    allowed_keys = set(map(tuple, allowed.itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, relay[["gap_id", "task_id", "candidate_id"]].astype(str).itertuples(index=False, name=None)))
    validation["checks"]["relay_option_in_step7"] = actual_keys <= allowed_keys
    validation["checks"]["joint_cmax_exact"] = abs(validation["actual_cmax_s"] - cmax) <= 1.0 + 1e-9
    validation["all_pass"] = all(validation["checks"].values())
    if not validation["all_pass"]:
        failed = [key for key, passed in validation["checks"].items() if not passed]
        raise AssertionError(f"Step8.5 failed: {failed}")
    result = {
        "status": status,
        "validation": validation,
        "objectives": evaluate_objectives(problem, transport, relay, delivery),
        "input_sha256": {name: _hash(DATA / name) for name in OUTPUTS},
    }
    if freeze:
        target = DATA / "q3_step8_frozen"
        target.mkdir(exist_ok=True)
        for name in OUTPUTS:
            destination = target / name
            if destination.exists() and _hash(destination) != result["input_sha256"][name]:
                raise FileExistsError(f"Frozen Step8 file differs: {destination}")
            if not destination.exists():
                shutil.copy2(DATA / name, destination)
        (target / "acceptance.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result
