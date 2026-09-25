"""Transport-only Q4 preview from a saved COMM transport snapshot.

This never reads relay candidates and never writes the final code/data/q4 folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import Q4


DATA = Path(__file__).resolve().parent / "data"
PREFIX = "Q2_anchor_COMM"
OUT = DATA / "q4_precompute_comm"
TRANSPORT_KINDS = Q4.KINDS[:6]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_comm_snapshot() -> tuple[dict, dict]:
    names = {key: DATA / f"{PREFIX}_{key}.csv" for key in ("selected", "schedule", "deliveries")}
    manifest_path = DATA / f"{PREFIX}_manifest.json"
    absent = [path.name for path in (*names.values(), manifest_path) if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Saved COMM transport snapshot is missing: {absent}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("objective") != "COMM" or manifest.get("all_pass") is not True or manifest.get("boxes") != 80:
        raise ValueError("COMM snapshot did not pass transport and 80-box validation")
    capture_path = DATA / "Q4_capture_COMM_manifest.json"
    if not capture_path.is_file():
        raise FileNotFoundError("COMM transport-only capture provenance is missing")
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    if (capture.get("status") != "PROVISIONAL_TRANSPORT_ONLY"
            or capture.get("q3_relay_feasibility_claimed") is not False
            or capture.get("comm_manifest_sha256") != sha(manifest_path)):
        raise ValueError("COMM capture provenance does not match transport snapshot")
    for key, path in names.items():
        if manifest.get("output_sha256", {}).get(key) != sha(path):
            raise ValueError(f"COMM snapshot hash mismatch: {path.name}")
    candidate_manifest = DATA / "Q2_compact_candidates_manifest.json"
    if manifest.get("candidate_manifest_sha256") != sha(candidate_manifest):
        raise ValueError("COMM candidate-class manifest changed")
    class_path = DATA / "Q2_compact_classes.csv"
    class_hash = json.loads(candidate_manifest.read_text(encoding="utf-8"))["outputs_sha256"]["classes"]
    if class_hash != sha(class_path):
        raise ValueError("Q2 box-class mapping changed")

    selected = Q4.read_csv(names["selected"])
    schedule = Q4.read_csv(names["schedule"])
    deliveries = Q4.read_csv(names["deliveries"])
    classes = {row["class_id"]: row for row in Q4.read_csv(class_path)}
    if len(selected) != len(schedule) or not selected:
        raise ValueError("COMM selected/schedule row count mismatch")
    selected_by_id = {Q4.normalize_sortie(row, "COMM selected"): row for row in selected}
    schedule_by_id = {Q4.normalize_sortie(row, "COMM schedule"): row for row in schedule}
    if len(selected_by_id) != len(selected) or len(schedule_by_id) != len(schedule) or set(selected_by_id) != set(schedule_by_id):
        raise ValueError("COMM selected/schedule sortie IDs mismatch")

    transport = []
    for sid, row in selected_by_id.items():
        scheduled = schedule_by_id[sid]
        if row["uav_type"] != scheduled["uav_type"] or row["visit_order"] != scheduled["visit_order"]:
            raise ValueError(f"COMM route or UAV type mismatch: {sid}")
        if abs(float(row["start_time_s"]) - float(scheduled["start_time_s"])) > 1e-6:
            raise ValueError(f"COMM start time mismatch: {sid}")
        sites = tuple(site.strip() for site in row["visit_order"].split(">"))
        if not sites or not set(sites) <= set(Q4.SERVICES):
            raise ValueError(f"COMM invalid route: {sid}")
        start = float(scheduled["start_time_s"])
        end = float(scheduled["end_time_s"])
        release = float(scheduled["flight_release_s"])
        charge = float(scheduled["battery_release_s"])
        if not start < end <= release or charge < end:
            raise ValueError(f"COMM invalid resource occupation: {sid}")
        transport.append({"sortie_id": sid, "_sites": sites, "uav_type": row["uav_type"],
                          "start_time_s": start, "end_time_s": end,
                          "uav_release_time_s": release, "charge_end_s": charge,
                          "source_uav_id": scheduled["uav_id"],
                          "source_battery_id": scheduled["battery_id"]})
    box_ids = set()
    preview_delivery = []
    for row in deliveries:
        sid = Q4.normalize_sortie(row, "COMM delivery")
        cls = classes[row["class_id"]]
        box = row["box_id"]
        if sid not in selected_by_id or box in box_ids or cls["service"] not in selected_by_id[sid]["visit_order"].split(">"):
            raise ValueError("COMM box assignment is missing, duplicated or outside its route")
        box_ids.add(box)
        preview_delivery.append({"sortie_id": sid, "box_id": box, "service": cls["service"],
                                 "_mass_kg": float(cls["mass"]),
                                 "delivery_time_s": float(row["delivery_time_s"])})
    if len(box_ids) != 80:
        raise ValueError(f"COMM snapshot has {len(box_ids)} distinct delivered boxes, expected 80")
    if set().union(*(set(row["_sites"]) for row in transport)) != set(Q4.SERVICES):
        raise ValueError("COMM snapshot does not cover all 15 service areas")
    source = {"manifest_sha256": sha(manifest_path), "files_sha256": {key: sha(path) for key, path in names.items()},
              "selected_sorties": len(transport), "reported_gaps": manifest.get("COMM_total_gap_count"),
              "master_status": manifest.get("master_status"), "transport_status": manifest.get("transport_status"),
              "relay_capacity_constraints_in_master": capture["relay_capacity_constraints_in_comm_master"],
              "q3_relay_feasibility_claimed": False}
    return {"transport": transport, "relay": [], "delivery": preview_delivery,
            "sha256": source["files_sha256"]}, source


def preview_solution(solution: dict) -> dict:
    return {"K": solution["K"], "scheme": solution["scheme"],
            "group_services": [group["services"] for group in solution["groups"]],
            "group_transport_resources": [
                {kind: group["resources"][kind] for kind in TRANSPORT_KINDS}
                for group in solution["groups"]],
            "group_transport_work_s": [group["work_s"] for group in solution["groups"]],
            "group_boxes": [group["boxes"] for group in solution["groups"]],
            "group_mass_kg": [group["mass_kg"] for group in solution["groups"]],
            "transport_total_resources": sum(solution["counts"][kind] for kind in TRANSPORT_KINDS),
            "transport_shortage": {kind: solution["shortage"][kind] for kind in TRANSPORT_KINDS},
            "transport_redundancy": {kind: solution["redundancy"][kind] for kind in TRANSPORT_KINDS},
            "transport_work_cv": solution["work_cv"]}


def main() -> None:
    snapshot, source = load_comm_snapshot()
    result = Q4.solve(snapshot)
    validation = Q4.validate_q4(snapshot, result)
    if not validation["all_pass"]:
        raise AssertionError(f"transport-only Q4 preview validation failed: {validation['checks']}")
    preview = {"status": "PROVISIONAL_TRANSPORT_ONLY", "source": source,
               "dependency_blocks_transport_only": result["blocks"],
               "unpartitioned_transport_pool": {kind: result["baseline"][kind] for kind in TRANSPORT_KINDS},
               "K": {}}
    for k in (2, 3):
        item = result["solutions"][str(k)]
        if item["status"] != "EXACT":
            preview["K"][str(k)] = {"status": item["status"], "reason": item["reason"]}
            continue
        preview["K"][str(k)] = {
            "partitions_checked": item["partitions_checked"],
            "transport_pareto_count": item["pareto_count"],
            "transport_shortage_first": preview_solution(item["shortage_first"]),
            "transport_balance_first": preview_solution(item["balance_first"]),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "blocks.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("block", "services"))
        for i, block in enumerate(result["blocks"], 1):
            writer.writerow((f"B{i}", ">".join(block)))
    with (OUT / "transport_groups.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("status", "K", "scheme", "group", "services", "boxes", "mass_kg", "work_s", *TRANSPORT_KINDS))
        for item in preview["K"].values():
            if "partitions_checked" not in item:
                continue
            for scheme in ("transport_shortage_first", "transport_balance_first"):
                s = item[scheme]
                for i, sites in enumerate(s["group_services"]):
                    writer.writerow((preview["status"], s["K"], scheme, f"G{i + 1}", ">".join(sites),
                                     s["group_boxes"][i], s["group_mass_kg"][i],
                                     s["group_transport_work_s"][i],
                                     *(s["group_transport_resources"][i][kind] for kind in TRANSPORT_KINDS)))
    print(json.dumps({"status": preview["status"], "source": source,
                      "blocks": len(result["blocks"]),
                      "K2": preview["K"]["2"].get("partitions_checked"),
                      "K3": preview["K"]["3"].get("partitions_checked"),
                      "output": str(OUT / "preview.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
