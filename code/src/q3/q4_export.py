"""Explicit Step13 export of an accepted, finally selected Q3 schedule for Q4."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.q2.data_model import load_boxes
from src.q3.cp_sat_scheduler import CANDIDATE_PATTERNS, DATA


OUTPUTS = (
    "q3_joint_transport_schedule.csv",
    "q3_joint_relay_schedule.csv",
    "q3_joint_delivery_schedule.csv",
    "q3_joint_resource_summary.csv",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _enrich_transport(transport):
    if "visit_order" not in transport:
        routes = pd.read_csv(CANDIDATE_PATTERNS, encoding="utf-8-sig")[["pattern_id", "visit_order"]]
        if routes["pattern_id"].duplicated().any():
            raise ValueError("Q3 pattern route map contains duplicate pattern IDs")
        transport = transport.merge(routes, on="pattern_id", how="left", validate="many_to_one")
    if transport["visit_order"].isna().any() or transport["visit_order"].eq("").any():
        raise ValueError("A selected Q3 sortie has no frozen visit_order")
    return transport


def _enrich_delivery(delivery):
    missing = [column for column in ("service", "mass_kg") if column not in delivery]
    if missing:
        boxes = load_boxes()[["box_id", "service", "mass"]].rename(columns={"mass": "mass_kg"})
        delivery = delivery.merge(boxes[["box_id", *missing]], on="box_id", how="left", validate="one_to_one")
    if (len(delivery) != 80 or delivery["box_id"].nunique() != 80
            or delivery[["service", "mass_kg"]].isna().any().any()):
        raise ValueError("Q3 delivery must contain 80 unique boxes with service and mass")
    return delivery


def freeze_q3_for_q4(source_dir, accepted_q3, final_selection_id, target_dir=None):
    """Create Q4 input only after Q3's final selection and independent acceptance.

    ``accepted_q3`` is the final Q3 validator result. A Step8 first feasible
    solution is not exported by this function automatically.
    """
    if not str(final_selection_id or "").strip():
        raise ValueError("Final Q3 selection ID is required; Step8 is not automatically final")
    if (accepted_q3.get("status") not in ("FEASIBLE", "OPTIMAL")
            or accepted_q3.get("validation", {}).get("all_pass") is not True):
        raise ValueError("Selected Q3 schedule has not passed final acceptance")
    source = Path(source_dir)
    target = Path(target_dir) if target_dir is not None else DATA / "q3_final_frozen"
    if target.exists():
        raise FileExistsError(f"Q3 final frozen target already exists: {target}")
    missing = [name for name in OUTPUTS if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Final Q3 schedule is incomplete: {missing}")
    source_hashes = accepted_q3.get("outputs_sha256") or accepted_q3.get("input_sha256", {})
    for name in OUTPUTS:
        if source_hashes.get(name) != _sha256(source / name):
            raise ValueError(f"Selected Q3 file differs from accepted version: {name}")

    transport = _enrich_transport(pd.read_csv(source / OUTPUTS[0], encoding="utf-8-sig"))
    relay = pd.read_csv(source / OUTPUTS[1], encoding="utf-8-sig")
    delivery = _enrich_delivery(pd.read_csv(source / OUTPUTS[2], encoding="utf-8-sig"))
    summary = pd.read_csv(source / OUTPUTS[3], encoding="utf-8-sig")
    if len(summary) != 1 or transport.empty:
        raise ValueError("Selected Q3 summary must have one row and transport must be nonempty")
    required_transport = {"sortie_id", "pattern_id", "visit_order", "uav_type",
                          "start_time_s", "end_time_s", "charge_end_s"}
    required_relay = {"sortie_id", "dispatch_time_s", "uav_release_time_s", "energy_release_time_s"}
    required_delivery = {"sortie_id", "box_id", "service", "mass_kg", "delivery_time_s"}
    for label, frame, columns in (("transport", transport, required_transport),
                                  ("relay", relay, required_relay),
                                  ("delivery", delivery, required_delivery)):
        absent = columns - set(frame)
        if absent:
            raise ValueError(f"Selected Q3 {label} lacks fields: {sorted(absent)}")
    if transport["sortie_id"].duplicated().any():
        raise ValueError("Selected Q3 transport has duplicate sortie IDs")
    routes = dict(zip(transport["sortie_id"].astype(str), transport["visit_order"].astype(str)))
    if any(str(row.service) not in routes[str(row.sortie_id)].split(">")
           for row in delivery.itertuples(index=False)):
        raise ValueError("A Q3 delivered box is not on its sortie's route")
    if not set(relay["sortie_id"].astype(str)) <= set(routes):
        raise ValueError("Q3 relay links an unknown transport sortie")

    target.mkdir(parents=True)
    for name, frame in zip(OUTPUTS, (transport, relay, delivery, summary)):
        frame.to_csv(target / name, index=False, encoding="utf-8-sig")
    acceptance = {
        "status": accepted_q3["status"],
        "validation": accepted_q3["validation"],
        "final_selection_id": str(final_selection_id),
        "source_sha256": {name: _sha256(source / name) for name in OUTPUTS},
        "outputs_sha256": {name: _sha256(target / name) for name in OUTPUTS},
    }
    (target / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return acceptance
