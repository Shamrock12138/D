"""Step 8.5: validate and freeze a completed Q3 minimum joint schedule."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from src.q3.cp_sat_scheduler import DATA, prepare_q3_problem, validate_q3_solution
from src.q2 import data_model
from src.q2.compact_classes import decode_box_deliveries
from src.q2.cp_sat_scheduler import _deadlines
from src.q3.objectives import evaluate_objectives
from src.q3.final_communication_validator import validate_fine_communication
from src.q3.relay.operation_profile import load_relay_flight_parameters
from src.q3.session_resources import attach_session_resources


OUTPUTS = (
    "q3_joint_transport_schedule.csv",
    "q3_joint_relay_schedule.csv",
    "q3_joint_delivery_schedule.csv",
    "q3_joint_resource_summary.csv",
    "q3_step8_manifest.json",
)
SESSION_OUTPUT = "q3_joint_relay_session_schedule.csv"


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_frozen_dir():
    """Return the final relay-session-v2 Step8 baseline; never fall back to v1."""
    candidate = DATA / "q3_step8_frozen_v2"
    if ((candidate / OUTPUTS[0]).is_file()
            and (candidate / OUTPUTS[1]).is_file()
            and (candidate / SESSION_OUTPUT).is_file()):
        return candidate
    raise FileNotFoundError(
        f"Final Q3 baseline must use relay_session_v2: {candidate}"
    )


def _verify_solver_inputs(input_sha256, input_dir=DATA):
    """Verify shared Step8 solver inputs, which live outside candidate dirs."""
    for name, recorded in input_sha256.items():
        source = Path(input_dir) / name
        if not source.is_file():
            raise FileNotFoundError(f"Step8 solver input is missing: {source}")
        if _hash(source) != recorded:
            raise AssertionError(f"Step8 input changed since solve: {name}")


def _output_hashes(data_dir, outputs=OUTPUTS):
    """Hash the Step8 artifacts in the candidate directory being accepted."""
    return {name: _hash(Path(data_dir) / name) for name in outputs}


def accept_step8(freeze=True, data_dir=None, freeze_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else DATA
    missing = [name for name in OUTPUTS if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Step8 outputs are incomplete: {missing}")
    manifest = json.loads((data_dir / "q3_step8_manifest.json").read_text(encoding="utf-8"))
    objective_schema = manifest.get("objective_schema", "relay_gap_v1")
    outputs = OUTPUTS + ((SESSION_OUTPUT,) if objective_schema == "relay_session_v2" else ())
    if objective_schema == "relay_session_v2" and not (data_dir / SESSION_OUTPUT).is_file():
        raise FileNotFoundError(f"Session-v2 Step8 output is missing: {data_dir / SESSION_OUTPUT}")
    status = manifest.get("status")
    if status not in ("FEASIBLE", "OPTIMAL"):
        raise AssertionError(f"Step8 solver status failed: {status}")
    solver_input_sha256 = manifest.get("input_sha256", {})
    _verify_solver_inputs(solver_input_sha256)
    tier = manifest.get("tier")
    if tier not in ("tier1", "tier2", "all"):
        raise ValueError(f"Unknown Step8 relay tier: {tier}")
    problem = prepare_q3_problem(tier=tier)
    problem["boxes"] = data_model.load_boxes()
    problem["deadlines"] = _deadlines(problem["boxes"])
    transport = pd.read_csv(data_dir / OUTPUTS[0], encoding="utf-8-sig")
    relay = pd.read_csv(data_dir / OUTPUTS[1], encoding="utf-8-sig")
    delivery = pd.read_csv(data_dir / OUTPUTS[2], encoding="utf-8-sig")
    summary = pd.read_csv(data_dir / OUTPUTS[3], encoding="utf-8-sig")
    if len(summary) != 1:
        raise AssertionError("Step8 resource summary must have exactly one row")
    cmax = float(summary.iloc[0]["joint_Cmax_s"])
    validation = validate_q3_solution(problem, transport, relay, cmax)
    selected_ids = set(transport["sortie_id"].astype(str))
    occ_by_sortie = {occ.sortie_id: occ for occ in problem["occurrences"]}
    selected = [
        {"sortie_id": str(row.sortie_id),
         "pattern_id": str(row.pattern_id),
         "start_time_s": float(row.start_time_s),
         "class_counts": occ_by_sortie[str(row.sortie_id)].class_counts}
        for row in transport.itertuples(index=False)
    ]
    expected_deliveries = decode_box_deliveries(
        selected, problem["classes"], problem["pattern_counts"]
    )
    validation["checks"]["delivery_schedule_matches_selected_tasks"] = (
        len(delivery) == len(expected_deliveries) == 80
        and delivery["box_id"].is_unique
        and set(zip(delivery["sortie_id"].astype(str), delivery["box_id"].astype(str),
                    delivery["class_id"].astype(str)))
        == set(zip(expected_deliveries["sortie_id"].astype(str),
                    expected_deliveries["box_id"].astype(str),
                    expected_deliveries["class_id"].astype(str)))
    )
    expected_times = {
        (str(row.sortie_id), str(row.box_id)): float(row.delivery_time_s)
        for row in expected_deliveries.itertuples(index=False)
    }
    validation["checks"]["delivery_times_match_offsets"] = all(
        (str(row.sortie_id), str(row.box_id)) in expected_times
        and abs(float(row.delivery_time_s) - expected_times[(str(row.sortie_id), str(row.box_id))]) <= 1e-6
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
    if objective_schema == "relay_session_v2":
        session_rows = pd.read_csv(data_dir / SESSION_OUTPUT, encoding="utf-8-sig")
        _, recomputed_sessions, _ = attach_session_resources(
            relay, load_relay_flight_parameters(), relay_options=problem.get("relay"))
        expected = recomputed_sessions.sort_values("relay_session_id").reset_index(drop=True)
        actual = session_rows.sort_values("relay_session_id").reset_index(drop=True)
        validation["checks"]["relay_session_resource_table"] = (
            list(actual.columns) == list(expected.columns)
            and len(actual) == len(expected)
            and (actual["relay_session_id"].astype(str).tolist()
                 == expected["relay_session_id"].astype(str).tolist())
            and all(abs(float(a)-float(b)) <= 1e-6
                    for column in expected.columns
                    if column not in {"relay_session_id", "relay_uav_id",
                                      "candidate_id", "energy_component_id"}
                    for a, b in zip(actual[column], expected[column]))
            and actual["energy_component_id"].astype(str).tolist()
                == expected["energy_component_id"].astype(str).tolist()
        )
        validation["checks"]["relay_session_component_consistency"] = all(
            relay.groupby("relay_session_id")["energy_component_id"].nunique() <= 1
        ) if len(relay) else True
    allowed = problem["relay"][["gap_id", "pattern_id", "candidate_id"]].astype(str)
    allowed_keys = set(map(tuple, allowed.itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, relay[["gap_id", "pattern_id", "candidate_id"]].astype(str).itertuples(index=False, name=None)))
    validation["checks"]["relay_candidate_in_step7"] = actual_keys <= allowed_keys
    validation["checks"]["joint_cmax_exact"] = abs(validation["actual_cmax_s"] - cmax) <= 1.0 + 1e-9
    fine_comm = validate_fine_communication(
        problem=problem, transport=transport, relay=relay, dt=1.0, data_dir=DATA)
    validation["fine_communication_1s"] = fine_comm
    validation["checks"]["fine_communication_1s"] = bool(fine_comm["all_pass"])
    validation["all_pass"] = all(validation["checks"].values())
    if not validation["all_pass"]:
        failed = [key for key, passed in validation["checks"].items() if not passed]
        fine_summary = {
            "unserved_time_samples": fine_comm["unserved_time_samples"],
            "uncovered_link_samples": fine_comm["uncovered_link_samples"],
            "unserved_examples": fine_comm["unserved_examples"][:3],
            "uncovered_examples": fine_comm["uncovered_examples"][:3],
        }
        raise AssertionError(f"Step8.5 failed: {failed}; fine_communication_1s={fine_summary}")
    outputs_sha256 = _output_hashes(data_dir, outputs)
    result = {
        "status": status,
        "validation": validation,
        "objectives": evaluate_objectives(problem, transport, relay, delivery),
        "outputs_sha256": outputs_sha256,
        "solver_input_sha256": solver_input_sha256,
        "objective_schema": objective_schema,
        # Keep the historical field as an alias for downstream compatibility.
        "input_sha256": outputs_sha256,
    }
    if freeze:
        target = Path(freeze_dir) if freeze_dir is not None else (
            data_dir / ("q3_step8_frozen_v2" if objective_schema == "relay_session_v2"
                        else "q3_step8_frozen"))
        target.mkdir(exist_ok=True)
        for name in outputs:
            destination = target / name
            if destination.exists() and _hash(destination) != outputs_sha256[name]:
                raise FileExistsError(f"Frozen Step8 file differs: {destination}")
            if not destination.exists():
                shutil.copy2(data_dir / name, destination)
        (target / "acceptance.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result
