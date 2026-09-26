"""Migrate the existing immutable Step8 baseline to relay-session schema v2."""
import json
import hashlib
import shutil
from pathlib import Path

import pandas as pd

from src.q2.battery import charge_time_to_full
from src.q3.bootstrap import subset_problem
from src.q3.cp_sat_scheduler import (
    DATA, RELAY_ENERGY_CAPACITY, prepare_q3_problem, solve_q3_joint,
    write_step8_outputs,
)
from src.q3.decomposition import _input_hashes
from src.q3.final_communication_validator import validate_fine_communication
from src.q3.relay.operation_profile import load_relay_flight_parameters
from src.q3.session_resources import attach_session_resources
from src.q3.step8_acceptance import (
    accept_step8, OUTPUTS, SESSION_OUTPUT,
)


def _repair_unserved_gap_edges(problem, transport, relay, failures, output_dir):
    """Extend only the selected gap options preceding 1 s unserved samples."""
    examples = failures.get("unserved_examples", [])
    if not examples or failures.get("uncovered_link_samples", 0):
        raise RuntimeError("Fine-communication failure is not a repairable early gap edge")
    start_by_sortie = dict(zip(transport.sortie_id.astype(str),
                               transport.start_time_s.astype(float)))
    gap_target = {}
    repair_records = []
    for example in examples:
        sid = str(example["sortie_id"])
        if sid not in start_by_sortie:
            raise RuntimeError(f"Unserved sample refers to unknown sortie {sid}")
        tau = float(example["time_s"]) - start_by_sortie[sid]
        jobs = relay.loc[relay.sortie_id.astype(str) == sid].copy()
        future = jobs.loc[
            jobs.service_start_s.astype(float) - start_by_sortie[sid] >= tau - 1e-6
        ]
        if future.empty:
            raise RuntimeError(f"No selected gap follows unserved sample for {sid} at {tau:.3f}s")
        job = future.assign(_delta=(future.service_start_s.astype(float)
                                    - start_by_sortie[sid] - tau)).sort_values("_delta").iloc[0]
        gap_id = str(job.gap_id)
        new_start = tau - 1.0
        gap_target[gap_id] = min(gap_target.get(gap_id, float("inf")), new_start)
        repair_records.append({"sortie_id": sid, "pattern_id": str(example["pattern_id"]),
                               "gap_id": gap_id, "unserved_tau_s": tau,
                               "new_coverage_start_s": new_start})

    options = problem["relay"].copy()
    params = load_relay_flight_parameters()
    for gap_id, new_start in gap_target.items():
        mask = options.gap_id.astype(str) == gap_id
        rows = options.loc[mask].copy()
        if rows.empty:
            raise RuntimeError(f"No Step7 options remain for local gap repair {gap_id}")
        delta = new_start - float(rows.coverage_start_s.min())
        if delta >= -1e-9:
            continue
        extra_s = -delta
        old_charge = rows.charge_time_s.astype(float).copy()
        rows["coverage_start_s"] = rows.coverage_start_s.astype(float) + delta
        rows["service_start_offset_s"] = rows.service_start_offset_s.astype(float) + delta
        rows["dispatch_offset_s"] = rows.dispatch_offset_s.astype(float) + delta
        rows["arrival_offset_s"] = rows.arrival_offset_s.astype(float) + delta
        rows["service_duration_s"] = rows.service_duration_s.astype(float) + extra_s
        if "tau_start_s" in rows:
            rows["tau_start_s"] = rows.tau_start_s.astype(float) + delta
        extra_energy = params.service_power_kw * extra_s / 3600.0
        rows["service_energy_kWh"] = rows.service_energy_kWh.astype(float) + extra_energy
        rows["relay_energy_kWh"] = (
            rows.outbound_energy_kWh.astype(float)
            + rows.return_energy_kWh.astype(float)
            + rows.service_energy_kWh.astype(float)
        )
        rows["end_soc"] = 1.0 - rows.relay_energy_kWh / params.energy_capacity_kwh
        rows["charge_time_s"] = rows.end_soc.map(
            lambda soc: charge_time_to_full(float(soc), params.full_charge_time_s))
        charge_delta = rows.charge_time_s.astype(float) - old_charge
        rows["relay_uav_occupancy_s"] = rows.relay_uav_occupancy_s.astype(float) + extra_s
        rows["energy_component_occupancy_s"] = (
            rows.energy_component_occupancy_s.astype(float) + extra_s + charge_delta
        )
        rows["min_transport_start_s"] = rows.apply(
            lambda row: max(0.0, float(row.lead_time_s) - float(row.coverage_start_s)), axis=1)
        feasible = ((rows.relay_energy_kWh <= params.max_energy_kwh + 1e-9)
                    & (rows.end_soc >= params.safety_margin - 1e-9))
        replacement = options.loc[~mask]
        options = pd.concat([replacement, rows.loc[feasible]], ignore_index=True)
        if not feasible.any():
            raise RuntimeError(f"Local timing expansion makes every option infeasible for {gap_id}")
    problem["relay"] = options.reset_index(drop=True)
    problem["gap_option_map"] = {
        str(gap_id): group.index.tolist()
        for gap_id, group in problem["relay"].groupby("gap_id", sort=False)
    }
    ids = transport.sortie_id.astype(str).tolist()
    hint = {"transport": transport, "relay": relay}
    result = solve_q3_joint(
        tier="all", time_limit_s=600, workers=8, random_seed=2026,
        problem=subset_problem(problem, ids), feasibility_only=True,
        hint=hint, allow_relay_sharing=True, fixed_sortie_ids=ids)
    if result["status"] not in ("FEASIBLE", "OPTIMAL"):
        raise RuntimeError(f"Local gap-edge joint repair returned {result['status']}: "
                           f"{result.get('postcheck_failed', [])}")
    result.update(input_sha256=_input_hashes(), solve_mode="fine_gap_edge_repair",
                  optimization_status="NOT_RUN", fine_gap_repairs=repair_records)
    write_step8_outputs(result, output_dir=output_dir)
    return repair_records


def migrate(source=None, target=None):
    source = Path(source) if source else DATA / "q3_step8_frozen"
    target = Path(target) if target else DATA / "q3_step8_frozen_v2"
    if source.resolve() == target.resolve():
        raise ValueError("v2 migration must not overwrite the v1 frozen directory")
    if not source.is_dir():
        raise FileNotFoundError(f"Frozen Step8 baseline not found: {source}")
    target.mkdir(parents=True, exist_ok=True)
    already_v2 = (target / SESSION_OUTPUT).is_file()
    if already_v2:
        manifest_path = target / OUTPUTS[4]
        if not manifest_path.is_file() or json.loads(
                manifest_path.read_text(encoding="utf-8")).get("objective_schema") != "relay_session_v2":
            raise FileExistsError(f"Refusing to replace unrecognized target: {target}")
    elif any(target.iterdir()):
        for name in OUTPUTS:
            source_file, target_file = source / name, target / name
            if (not target_file.is_file()
                    or hashlib.sha256(source_file.read_bytes()).digest()
                    != hashlib.sha256(target_file.read_bytes()).digest()):
                raise FileExistsError(
                    f"Refusing to overwrite non-pristine v2 migration file: {target_file}")
    else:
        for name in OUTPUTS:
            shutil.copy2(source / name, target / name)
    transport = pd.read_csv(target / OUTPUTS[0], encoding="utf-8-sig")
    relay = pd.read_csv(target / OUTPUTS[1], encoding="utf-8-sig")
    problem = prepare_q3_problem(tier="all")
    fine_check = validate_fine_communication(problem, transport, relay, dt=1.0, data_dir=DATA)
    if not fine_check["all_pass"]:
        if fine_check["unserved_time_samples"] <= 0 or fine_check["uncovered_link_samples"]:
            raise AssertionError(f"Fine communication needs a non-edge/local candidate repair: {fine_check}")
        repairs = _repair_unserved_gap_edges(problem, transport, relay, fine_check, target)
        (target / OUTPUTS[4]).write_text(json.dumps({
            "status": "FEASIBLE", "tier": "all", "objective_schema": "relay_session_v2",
            "input_sha256": _input_hashes(), "fine_gap_repairs": repairs,
            "validation": {"all_pass": True},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        transport = pd.read_csv(target / OUTPUTS[0], encoding="utf-8-sig")
        relay = pd.read_csv(target / OUTPUTS[1], encoding="utf-8-sig")
    relay, sessions, resources_fit = attach_session_resources(
        relay, load_relay_flight_parameters(), RELAY_ENERGY_CAPACITY,
        problem.get("relay"))
    if not resources_fit:
        raise AssertionError("Frozen baseline cannot be assigned to six session energy components")
    relay.to_csv(target / OUTPUTS[1], index=False, encoding="utf-8-sig")
    sessions.to_csv(target / SESSION_OUTPUT, index=False, encoding="utf-8-sig")
    summary = pd.read_csv(target / OUTPUTS[3], encoding="utf-8-sig")
    summary["relay_sessions"] = len(sessions)
    summary["relay_energy_kWh"] = float(sessions["relay_session_energy_kWh"].sum())
    summary["relay_session_energy_kWh"] = summary["relay_energy_kWh"]
    summary["total_energy_kWh"] = summary["transport_energy_kWh"] + summary["relay_energy_kWh"]
    summary["objective_schema"] = "relay_session_v2"
    summary.to_csv(target / OUTPUTS[3], index=False, encoding="utf-8-sig")
    manifest = json.loads((target / OUTPUTS[4]).read_text(encoding="utf-8"))
    manifest["objective_schema"] = "relay_session_v2"
    manifest["session_resources_feasible"] = True
    manifest.setdefault("migration", {"from": str(source), "schema": "relay_session_v2"})
    (target / OUTPUTS[4]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    acceptance = accept_step8(freeze=True, data_dir=target, freeze_dir=target)
    (target / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return acceptance


if __name__ == "__main__":
    result = migrate()
    print(json.dumps({"status": result["status"],
                      "all_pass": result["validation"]["all_pass"],
                      "objectives": result["objectives"]}, ensure_ascii=False, indent=2))
