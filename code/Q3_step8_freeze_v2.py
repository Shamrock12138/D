"""Migrate the existing immutable Step8 baseline to relay-session schema v2."""
import json
import hashlib
import shutil
from pathlib import Path

import pandas as pd

from src.q3.cp_sat_scheduler import DATA, RELAY_ENERGY_CAPACITY, prepare_q3_problem
from src.q3.relay.operation_profile import load_relay_flight_parameters
from src.q3.session_resources import attach_session_resources
from src.q3.step8_acceptance import accept_step8, OUTPUTS, SESSION_OUTPUT


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
    relay, sessions, resources_fit = attach_session_resources(
        relay, load_relay_flight_parameters(), RELAY_ENERGY_CAPACITY,
        prepare_q3_problem(tier="all").get("relay"))
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
    manifest["migration"] = {"from": str(source), "schema": "relay_session_v2"}
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
