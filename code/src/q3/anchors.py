"""Step 9: four single-objective anchors on the *joint* Q3 CP-SAT model."""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

from src.q3.cp_sat_scheduler import (
    DATA, _build_q3_model, _decode_q3_resources, prepare_q3_problem,
    validate_q3_solution,
)
from src.q3.objectives import OBJECTIVE_NAMES, evaluate_objectives, soft_box_targets
from src.q3.step8_acceptance import accept_step8


ENERGY_SCALE = 1_000_000  # integer micro-kWh; exact kWh is reported after decoding


def _objective_expressions(model, problem, select, starts, relay_select, joint_cmax):
    tasks = problem["tasks"]
    relay = problem["relay"]
    task_index = {str(task_id): i for i, task_id in enumerate(tasks["task_id"].astype(str))}
    targets = soft_box_targets(problem["boxes"], problem["deadlines"])
    tardiness_terms = []
    for row in problem["deliveries"].itertuples(index=False):
        box_id = str(row.box_id)
        if box_id not in targets:
            continue
        expected, weight = targets[box_id]
        if not float(weight).is_integer():
            raise ValueError("F1 priority weights must be integers for CP-SAT")
        i = task_index[str(row.task_id)]
        lateness = model.NewIntVar(0, problem["horizon_s"], f"soft_late_{box_id}_{i}")
        conservative_offset = math.ceil(float(row.delivery_offset_s))
        model.Add(lateness >= starts[i] + conservative_offset - math.floor(expected)).OnlyEnforceIf(select[i])
        model.Add(lateness == 0).OnlyEnforceIf(select[i].Not())
        tardiness_terms.append(int(weight) * lateness)
    transport_energy = [math.ceil(float(v) * ENERGY_SCALE) for v in tasks["energy_kWh"]]
    relay_energy = [math.ceil(float(v) * ENERGY_SCALE) for v in relay["relay_energy_kWh"]]
    return {
        "F1_timeliness": sum(tardiness_terms),
        "F2_joint_cmax_s": joint_cmax,
        "F3_total_energy_kWh": sum(v * select[i] for i, v in enumerate(transport_energy))
        + sum(v * relay_select[i] for i, v in enumerate(relay_energy)),
        "F4_total_sorties": sum(select) + sum(relay_select),
    }


def solve_anchor(problem, objective, time_limit_s=600, workers=8, random_seed=2026,
                 baseline=None):
    if objective not in OBJECTIVE_NAMES:
        raise ValueError(f"Unknown Q3 objective: {objective}")
    built = _build_q3_model(problem)
    model, select, starts, relay_select, relay_starts, transport_cmax, relay_cmax, joint_cmax, metadata = built
    expressions = _objective_expressions(model, problem, select, starts, relay_select, joint_cmax)
    model.Minimize(expressions[objective])
    if baseline is not None:
        transport_start = dict(zip(baseline["transport"]["task_id"].astype(str),
                                   baseline["transport"]["start_time_s"].astype(int)))
        relay_keys = set(map(tuple, baseline["relay"][["gap_id", "task_id", "candidate_id"]]
                             .astype(str).itertuples(index=False, name=None)))
        for i, row in problem["tasks"].iterrows():
            task_id = str(row.task_id)
            model.AddHint(select[i], int(task_id in transport_start))
            model.AddHint(starts[i], transport_start.get(task_id, 0))
        for i, row in problem["relay"].iterrows():
            key = (str(row.gap_id), str(row.task_id), str(row.candidate_id))
            model.AddHint(relay_select[i], int(key in relay_keys))
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
    if not validation["all_pass"]:
        raise AssertionError(f"Anchor {objective} failed Q3 validation: {validation['checks']}")
    record.update(evaluate_objectives(problem, transport, relay, delivery))
    record["solver_objective_integer"] = int(solver.Value(expressions[objective]))
    record["validation"] = validation
    record["transport"] = transport
    record["relay"] = relay
    record["delivery"] = delivery
    return record


def run_anchors(time_limit_s=600, workers=8, random_seed=2026):
    acceptance = accept_step8(freeze=True)
    frozen = DATA / "q3_step8_frozen"
    baseline = {
        "transport": pd.read_csv(frozen / "q3_joint_transport_schedule.csv", encoding="utf-8-sig"),
        "relay": pd.read_csv(frozen / "q3_joint_relay_schedule.csv", encoding="utf-8-sig"),
    }
    step8_manifest = json.loads((frozen / "q3_step8_manifest.json").read_text(encoding="utf-8"))
    problem = prepare_q3_problem(tier=step8_manifest["tier"])
    problem["tier"] = step8_manifest["tier"]
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
            "F3_total_energy_kWh": "transport plus relay task energy",
            "F4_total_sorties": "transport plus relay sortie count",
        },
        "ideal_point_incumbent": ideal,
        "ideal_point_proven_optimal": bool((table["status"] == "OPTIMAL").all()),
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
