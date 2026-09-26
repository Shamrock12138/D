"""Migrate and accept a Step8 baseline using the relay-session-v2 schema.

This script only performs schema migration and acceptance. It never repairs or
re-solves a schedule; failed validation is reported to the caller.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from src.q3.cp_sat_scheduler import DATA, prepare_q3_problem
from src.q3.relay.operation_profile import load_relay_flight_parameters
from src.q2.battery import soc_after_task
from src.q3.session_resources import attach_session_resources
from src.q3.step8_acceptance import accept_step8, OUTPUTS, SESSION_OUTPUT


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _restore_gap_fields(relay, transport, relay_params):
    """Restore gap-level SOC/release fields in older v2 rows if needed."""
    fields = {"relay_energy_kWh", "energy_component_occupancy_s",
              "dispatch_offset_s", "sortie_id"}
    if relay.empty or not fields <= set(relay.columns):
        return relay
    starts = transport.set_index("sortie_id")["start_time_s"].to_dict()
    relay = relay.copy()
    relay["end_soc"] = relay["relay_energy_kWh"].map(
        lambda energy: soc_after_task(float(energy), relay_params.energy_capacity_kwh)
    )
    relay["energy_release_time_s"] = [
        float(starts[str(row.sortie_id)])
        + float(row.dispatch_offset_s)
        + float(offset)
        for row, offset in zip(
            relay.itertuples(index=False), relay["energy_component_occupancy_s"]
        )
    ]
    return relay


def migrate(source=None, target=None):
    source = Path(source) if source else DATA / "q3_step8_frozen"
    target = Path(target) if target else DATA / "q3_step8_frozen_v2"
    if source.resolve() == target.resolve():
        raise ValueError("v2 migration must not overwrite the v1 frozen directory")

    already_v2 = (target / SESSION_OUTPUT).is_file()
    if already_v2:
        manifest_path = target / OUTPUTS[4]
        if (not manifest_path.is_file()
                or json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "objective_schema") != "relay_session_v2"):
            raise FileExistsError(f"Refusing to replace unrecognized target: {target}")
    else:
        if not source.is_dir():
            raise FileNotFoundError(f"Frozen Step8 baseline not found: {source}")
        target.mkdir(parents=True, exist_ok=True)
        if any(target.iterdir()):
            for name in OUTPUTS:
                source_file, target_file = source / name, target / name
                if (not target_file.is_file()
                        or _sha256(source_file) != _sha256(target_file)):
                    raise FileExistsError(
                        f"Refusing to overwrite non-pristine v2 migration file: {target_file}"
                    )
        else:
            for name in OUTPUTS:
                shutil.copy2(source / name, target / name)

        problem = prepare_q3_problem(tier="all")
        transport = pd.read_csv(target / OUTPUTS[0], encoding="utf-8-sig")
        relay = pd.read_csv(target / OUTPUTS[1], encoding="utf-8-sig")
        relay, sessions, resources_fit = attach_session_resources(
            relay, load_relay_flight_parameters(), 6, problem.get("relay")
        )
        if not resources_fit:
            raise AssertionError(
                "Frozen baseline cannot be assigned to six session energy components"
            )
        relay.to_csv(target / OUTPUTS[1], index=False, encoding="utf-8-sig")
        sessions.to_csv(target / SESSION_OUTPUT, index=False, encoding="utf-8-sig")

        summary = pd.read_csv(target / OUTPUTS[3], encoding="utf-8-sig")
        summary["relay_sessions"] = len(sessions)
        summary["relay_energy_kWh"] = float(sessions["relay_session_energy_kWh"].sum())
        summary["relay_session_energy_kWh"] = summary["relay_energy_kWh"]
        summary["total_energy_kWh"] = (
            summary["transport_energy_kWh"] + summary["relay_energy_kWh"]
        )
        summary["objective_schema"] = "relay_session_v2"
        summary.to_csv(target / OUTPUTS[3], index=False, encoding="utf-8-sig")

        manifest = json.loads(
            (target / OUTPUTS[4]).read_text(encoding="utf-8")
        )
        manifest["objective_schema"] = "relay_session_v2"
        manifest["session_resources_feasible"] = True
        manifest.setdefault("migration", {
            "from": str(source), "schema": "relay_session_v2"
        })
        (target / OUTPUTS[4]).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # Upgrade old v2 gap rows whose session-level values occupied gap-level
    # columns. New v2 writes keep the meanings separate through attach_session_resources.
    problem = prepare_q3_problem(tier="all")
    transport = pd.read_csv(target / OUTPUTS[0], encoding="utf-8-sig")
    relay = pd.read_csv(target / OUTPUTS[1], encoding="utf-8-sig")
    relay = _restore_gap_fields(
        relay, transport, load_relay_flight_parameters()
    )
    relay, sessions, resources_fit = attach_session_resources(
        relay, load_relay_flight_parameters(), 6, problem.get("relay")
    )
    if not resources_fit:
        raise AssertionError("Frozen baseline exceeds session energy-component capacity")
    relay.to_csv(target / OUTPUTS[1], index=False, encoding="utf-8-sig")
    sessions.to_csv(target / SESSION_OUTPUT, index=False, encoding="utf-8-sig")

    checked = accept_step8(freeze=False, data_dir=target)
    manifest_path = target / OUTPUTS[4]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation"] = checked["validation"]
    manifest["validation_source"] = "src.q3.step8_acceptance.accept_step8"
    if manifest.get("fine_gap_repairs"):
        manifest["schedule_postprocessing_note"] = (
            "This preserved legacy candidate includes a recorded fine-gap timing "
            "repair. The migration script did not alter or re-solve that schedule; "
            "the current acceptance validation was recomputed independently."
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    acceptance = accept_step8(freeze=True, data_dir=target, freeze_dir=target)
    (target / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return acceptance


if __name__ == "__main__":
    result = migrate()
    print(json.dumps({
        "status": result["status"],
        "all_pass": result["validation"]["all_pass"],
        "objectives": result["objectives"],
    }, ensure_ascii=False, indent=2))
