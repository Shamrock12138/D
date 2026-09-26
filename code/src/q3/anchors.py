

import hashlib
import json
import math
import pandas as pd
from ortools.sat.python import cp_model

from src.q3.cp_sat_scheduler import (
    DATA, _build_q3_model, _decode_q3_resources, prepare_q3_problem,
    validate_q3_solution,
)
from src.q2 import data_model
from src.q2.cp_sat_scheduler import _deadlines
from src.q3.objectives import (
    ENERGY_SCALE, F1_TIME_SCALE, OBJECTIVE_NAMES, Q3_MULTI_OBJECTIVE_TIER,
    energy_units, evaluate_objectives, soft_box_targets, time_units,
)
from src.q3.final_communication_validator import validate_fine_communication
from src.q3.step8_acceptance import accept_step8, resolve_frozen_dir


def _build_objective_expression(objective, model, problem, select, starts,
                                relay_select, joint_cmax, metadata=None):
    if metadata is None:
        if objective == "F2_joint_cmax_s":
            return joint_cmax
        if objective == "F4_total_sorties":
            return sum(select) + sum(relay_select)
        if objective == "F3_total_energy_kWh":
            transport_energy = [energy_units(v) for v in problem["tasks"]["energy_kWh"]]
            relay_energy = [energy_units(v) for v in problem["relay"]["relay_energy_kWh"]]
            return sum(v * select[i] for i, v in enumerate(transport_energy)) + sum(
                v * relay_select[i] for i, v in enumerate(relay_energy))
        targets = soft_box_targets(problem["boxes"], problem["deadlines"])
        task_index = {str(task_id): i for i, task_id in enumerate(problem["tasks"]["task_id"].astype(str))}
        terms = []
        for row in problem["deliveries"].itertuples(index=False):
            if str(row.box_id) not in targets:
                continue
            expected, weight = targets[str(row.box_id)]
            i = task_index[str(row.task_id)]
            late = model.NewIntVar(0, problem["horizon_s"] * F1_TIME_SCALE,
                                   f"soft_late_{row.box_id}_{i}")
            model.Add(late >= F1_TIME_SCALE * starts[i] + time_units(row.delivery_offset_s)
                      - time_units(expected)).OnlyEnforceIf(select[i])
            model.Add(late == 0).OnlyEnforceIf(select[i].Not())
            terms.append(int(weight) * late)
        return sum(terms)
    if objective == "F2_joint_cmax_s":
        return joint_cmax
    if objective == "F4_total_sorties":
        return sum(select) + sum(metadata["relay_session_starts"])
    if objective == "F3_total_energy_kWh":
        transport_energy = sum(
            energy_units(occ.energy_kWh) * select[i]
            for i, occ in enumerate(problem["occurrences"]))
        relay_energy = sum(
            energy_units(meta["service_energy_kWh"]) * relay_select[i]
            + energy_units(meta["outbound_energy_kWh"] + meta["return_energy_kWh"])
              * metadata["relay_session_starts"][i]
            for i, meta in enumerate(metadata["relay"]))
        return transport_energy + relay_energy
    if objective != "F1_timeliness":
        raise ValueError(f"Unknown Q3 objective: {objective}")
    class_params = problem["class_params"]
    tardiness_terms = []
    for i, occ in enumerate(problem["occurrences"]):
        for class_id, amount in occ.class_counts.items():
            item = class_params[class_id]
            expected = float(item["expected_time_s"])
            if math.isfinite(float(item["hard_deadline_s"])) or not math.isfinite(expected):
                continue
            late = model.NewIntVar(0, problem["horizon_s"] * F1_TIME_SCALE,
                                   f"soft_late_{class_id}_{i}")
            offset_units = time_units(occ.delivery_offsets.get(class_id, 0.0))
            expected_units = time_units(expected)
            model.Add(late >= F1_TIME_SCALE * starts[i] + offset_units - expected_units).OnlyEnforceIf(select[i])
            model.Add(late == 0).OnlyEnforceIf(select[i].Not())
            tardiness_terms.append(int(item["priority"]) * int(amount) * late)
    return sum(tardiness_terms)


def solve_anchor(problem, objective, time_limit_s=600, workers=8, random_seed=2026,
                 baseline=None):
    if objective not in OBJECTIVE_NAMES:
        raise ValueError(f"Unknown Q3 objective: {objective}")
    built = _build_q3_model(problem, allow_relay_sharing=True)
    model, select, starts, relay_select, relay_starts, transport_cmax, relay_cmax, joint_cmax, metadata = built
    expression = _build_objective_expression(
        objective, model, problem, select, starts, relay_select, joint_cmax, metadata
    )
    model.Minimize(expression)
    if baseline is not None:
        transport_start = dict(zip(baseline["transport"]["sortie_id"].astype(str),
                                   baseline["transport"]["start_time_s"].astype(int)))
        relay_keys = set(map(tuple, baseline["relay"][["sortie_id", "gap_id", "candidate_id"]]
                             .astype(str).itertuples(index=False, name=None)))
        for i, occ in enumerate(problem["occurrences"]):
            sid = str(occ.sortie_id)
            model.AddHint(select[i], int(sid in transport_start))
            model.AddHint(starts[i], transport_start.get(sid, 0))
        baseline_relay_starts = dict(zip(
            map(tuple, baseline["relay"][["sortie_id", "gap_id", "candidate_id"]]
                .astype(str).itertuples(index=False, name=None)),
            baseline["relay"]["dispatch_time_s"].astype(int)))
        relay_ids = dict(zip(
            map(tuple, baseline["relay"][["sortie_id", "gap_id", "candidate_id"]]
                .astype(str).itertuples(index=False, name=None)),
            baseline["relay"]["relay_uav_id"].astype(str)))
        for i, meta in enumerate(metadata["relay"]):
            key = (str(meta["sortie_id"]), str(meta["gap_id"]), str(meta["candidate_id"]))
            model.AddHint(relay_select[i], int(key in relay_keys))
            model.AddHint(relay_starts[i], baseline_relay_starts.get(key, 0))
            for relay_id in metadata["relay_ids"]:
                model.AddHint(metadata["relay_assign"][(i, relay_id)],
                              int(relay_ids.get(key) == relay_id))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(random_seed)
    status = solver.Solve(model)
    record = {
        "objective": objective, "status": solver.StatusName(status),
        "wall_time_s": solver.WallTime(), "best_bound": solver.BestObjectiveBound(),
        "tier": problem["tier"],
    }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return record
    transport, relay, delivery = _decode_q3_resources(
        problem, solver, select, starts, relay_select, relay_starts, metadata
    )
    validation = validate_q3_solution(problem, transport, relay, int(solver.Value(joint_cmax)))
    fine_comm = validate_fine_communication(
        problem=problem,
        transport=transport,
        relay=relay,
        dt=1.0,
        data_dir=DATA,
    )
    validation["fine_communication_1s"] = fine_comm
    validation["checks"]["fine_communication_1s"] = bool(fine_comm["all_pass"])
    validation["all_pass"] = all(validation["checks"].values())
    if not validation["all_pass"]:
        failed = [name for name, passed in validation["checks"].items() if not passed]
        raise AssertionError(
            f"Anchor {objective} failed final validation: {failed}; "
            f"fine_communication_1s={fine_comm}"
        )
    record.update(evaluate_objectives(problem, transport, relay, delivery))
    solver_objective_integer = int(solver.Value(expression))
    record["solver_objective_integer"] = solver_objective_integer
    objective_scale = F1_TIME_SCALE if objective == "F1_timeliness" else (
        ENERGY_SCALE if objective == "F3_total_energy_kWh" else 1
    )
    solver_proxy_value = solver_objective_integer / objective_scale
    record["solver_objective_value"] = solver_proxy_value
    record["solver_objective_matches_reported"] = (
        abs(record[objective] - solver_proxy_value) <= 1e-7
    )
    if objective == "F3_total_energy_kWh":
        record["solver_objective_kind"] = "gap_service_plus_session_flight_proxy"
    elif not record["solver_objective_matches_reported"]:
        raise AssertionError(f"Anchor {objective} solver/report objective mismatch")
    record["validation"] = validation
    record["transport"] = transport
    record["relay"] = relay
    record["delivery"] = delivery
    return record


def run_anchors(time_limit_s=600, workers=8, random_seed=2026):
    try:
        frozen = resolve_frozen_dir()
    except FileNotFoundError:
        acceptance = accept_step8(freeze=True)
        frozen = resolve_frozen_dir()
    else:
        acceptance = accept_step8(freeze=False, data_dir=frozen)
    baseline = {
        "transport": pd.read_csv(frozen / "q3_joint_transport_schedule.csv", encoding="utf-8-sig"),
        "relay": pd.read_csv(frozen / "q3_joint_relay_schedule.csv", encoding="utf-8-sig"),
    }
    problem = prepare_q3_problem(tier=Q3_MULTI_OBJECTIVE_TIER)
    problem["tier"] = Q3_MULTI_OBJECTIVE_TIER
    problem["boxes"] = data_model.load_boxes()
    problem["deadlines"] = _deadlines(problem["boxes"])
    output = DATA / "q3_anchors"
    output.mkdir(exist_ok=True)
    rows = []
    for objective in OBJECTIVE_NAMES:
        result = solve_anchor(problem, objective, time_limit_s, workers, random_seed, baseline)
        print(f"{objective}: {result['status']}", flush=True)
        if result["status"] not in ("FEASIBLE", "OPTIMAL"):
            raise RuntimeError(f"Anchor {objective} returned {result['status']}; no ideal point is available")
        prefix = objective.split("_")[0]
        for field in ("transport", "relay", "delivery"):
            result[field].to_csv(output / f"{prefix}_{field}.csv", index=False, encoding="utf-8-sig")
        rows.append({key: value for key, value in result.items()
                     if key not in ("transport", "relay", "delivery", "validation")})
    table = pd.DataFrame(rows)
    table.to_csv(output / "q3_anchor_table.csv", index=False, encoding="utf-8-sig")
    ideal = {name: float(table[name].min()) for name in OBJECTIVE_NAMES}
    manifest = {
        "objectives": {
            "F1_timeliness": "sum of priority-weighted positive delays in seconds for boxes without a hard deadline",
            "F2_joint_cmax_s": "maximum of final transport landing and final relay return to O01",
            "F3_total_energy_kWh": (
                "reported physical session energy; the CP-SAT F3 anchor uses "
                "a gap-service plus session-flight search proxy"
            ),
            "F4_total_sorties": "transport plus relay sortie count",
        },
        "ideal_point_incumbent": ideal,
        "solver_anchors_proven_optimal_for_encoded_objectives": bool(
            (table["status"] == "OPTIMAL").all()
        ),
        "ideal_point_proven_optimal": False,
        "F1_time_scale": F1_TIME_SCALE,
        "F3_energy_scale": ENERGY_SCALE,
        "anchor_time_limit_s_each": time_limit_s,
        "random_seed": random_seed,
        "tier": problem["tier"],
        "step8_acceptance_sha256": hashlib.sha256(
            (frozen / "acceptance.json").read_bytes()).hexdigest(),
        "input_sha256": acceptance["input_sha256"],
    }
    (output / "q3_anchor_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return table
