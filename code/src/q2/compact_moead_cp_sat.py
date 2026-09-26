






from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

from src.q2.battery import charge_time_to_full, soc_after_task
from src.q2.compact_classes import (
    class_timeliness, decode_box_deliveries, materialize_selected_sorties,
    select_compact_patterns,
)
from src.q2.data_model import load_q2_data
from src.q2.moead import build_neighbors, generate_weights, update_archive


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OBJECTIVE_NAMES = ("F1", "Cmax", "E", "N")
ANCHOR_NAMES = ("F1", "Cmax", "E", "N")
ARCHIVE_TOLERANCE = (1e-6, 1.0, 1e-5, 0.0)
OBJECTIVE_RANGES = (433804.83, 6080.03, 60.80, 29.0)
OBJECTIVE_UNIT = {"F1": 1000, "Cmax": 1, "E": 1_000_000, "N": 1}
NORMALIZATION_SCALE = 1_000_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_compact_problem(top_k: int = 3, data_dir: Path = DATA) -> dict:

    if int(top_k) != 3:
        raise ValueError("Current compact MOEA/D anchors require top_k=3")
    data_dir = Path(data_dir)
    manifest_path = data_dir / "Q2_compact_candidates_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_names = {
        "classes": "Q2_compact_classes.csv",
        "class_members": "Q2_compact_class_members.csv",
        "patterns": "Q2_compact_patterns.csv",
        "pattern_counts": "Q2_compact_pattern_counts.csv",
    }
    for key, filename in artifact_names.items():
        if _sha256(data_dir / filename) != manifest["outputs_sha256"][key]:
            raise ValueError(f"Compact candidate artifact changed: {filename}")
    data = load_q2_data()
    from Q2_compact_smoke import _load_saved_compact_candidates, _resources
    classes, patterns, counts = _load_saved_compact_candidates(data["boxes"])
    patterns, counts = select_compact_patterns(patterns, counts, classes, top_k=3)
    uav_ids, battery_ids, energy_capacity, charge_full = _resources(data)
    class_supply = {str(row.class_id): int(row.count)
                    for row in classes.itertuples(index=False)}
    if (len(data["boxes"]), len(classes), len(patterns)) != (80, 62, 2225):
        raise ValueError(
            "Expected compact dimensions 80 boxes / 62 classes / 2225 patterns; "
            f"got {len(data['boxes'])} / {len(classes)} / {len(patterns)}"
        )
    pattern_counts = {
        str(pid): {str(row.class_id): int(row.count)
                   for row in group.itertuples(index=False)}
        for pid, group in counts.groupby("pattern_id", sort=False)
    }
    offsets = {(str(row.pattern_id), str(row.class_id)): float(row.delivery_offset_s)
               for row in counts.itertuples(index=False)}
    return {
        "data_dir": data_dir, "candidate_manifest": manifest,
        "candidate_manifest_sha256": _sha256(manifest_path),
        "classes": classes, "patterns": patterns, "pattern_counts": counts,
        "pattern_class_counts": pattern_counts, "delivery_offsets": offsets,
        "boxes": data["boxes"], "class_supply": class_supply,
        "uav_ids": uav_ids, "battery_ids": battery_ids,
        "energy_capacity": energy_capacity, "charge_full": charge_full,
        "horizon_s": 36000, "top_k": 3,
    }


def canonicalize_sortie_ids(problem: dict, sortie_ids) -> tuple[str, ...]:

    counts = Counter()
    seen_input = set()
    for value in sortie_ids:
        text = str(value)
        if text in seen_input:
            raise ValueError(f"Duplicate compact sortie id: {text}")
        seen_input.add(text)
        pattern_id, sep, copy = text.rpartition("-")
        if not sep or not copy.isdigit():
            raise ValueError(f"Invalid compact sortie id: {text}")
        if pattern_id not in problem["pattern_class_counts"]:
            raise ValueError(f"Unknown compact pattern in sortie id: {text}")
        counts[pattern_id] += 1
    normalized = []
    for pattern_id in sorted(counts):
        max_copies = min(problem["class_supply"][cid] // amount
                         for cid, amount in problem["pattern_class_counts"][pattern_id].items())
        if counts[pattern_id] > max_copies:
            raise ValueError(f"Too many copies of compact pattern {pattern_id}")
        normalized.extend(f"{pattern_id}-{i}" for i in range(1, counts[pattern_id] + 1))
    return tuple(sorted(normalized))


def _canonicalize_id_map(problem: dict, sortie_ids):

    per_pattern = Counter()
    mapping = {}
    for old in map(str, sortie_ids):
        pid = old.rpartition("-")[0]
        per_pattern[pid] += 1
        mapping[old] = f"{pid}-{per_pattern[pid]}"
    canonical = canonicalize_sortie_ids(problem, mapping.values())
    return canonical, mapping


def compute_residual_demand(problem: dict, fixed_sortie_ids) -> dict[str, int]:
    fixed = canonicalize_sortie_ids(problem, fixed_sortie_ids)
    residual = dict(problem["class_supply"])
    for sortie_id in fixed:
        pid = sortie_id.rpartition("-")[0]
        for cid, amount in problem["pattern_class_counts"][pid].items():
            residual[cid] -= amount
            if residual[cid] < 0:
                raise ValueError(f"Fixed occurrences oversupply class {cid}")
    return residual


def load_compact_anchor(objective: str, problem: dict,
                        data_dir: Path | None = None) -> dict:

    objective = str(objective)
    if objective not in OBJECTIVE_NAMES:
        raise ValueError(f"Unknown compact objective: {objective}")
    data_dir = Path(data_dir or problem["data_dir"])
    prefix = f"Q2_anchor_{objective}"
    manifest = json.loads((data_dir / f"{prefix}_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("candidate_manifest_sha256") != problem["candidate_manifest_sha256"]:
        raise ValueError(f"{prefix} was generated from a different candidate pool")
    if int(manifest.get("top_k", -1)) != 3:
        raise ValueError(f"{prefix} does not use top_k=3")
    if manifest.get("objective") != objective or manifest.get("all_pass") is not True:
        raise ValueError(f"{prefix} objective or validation status is invalid")
    dims = (int(manifest.get("boxes", -1)), int(manifest.get("classes", -1)),
            int(manifest.get("retained_patterns", -1)), int(manifest.get("sortie_slots", -1)))
    if dims != (80, 62, 2225, 2313):
        raise ValueError(f"{prefix} dimensions are incompatible: {dims}")
    for key in ("selected", "schedule", "deliveries"):
        path = data_dir / f"{prefix}_{key}.csv"
        if _sha256(path) != manifest.get("output_sha256", {}).get(key):
            raise ValueError(f"{prefix} output hash mismatch: {path.name}")
    selected = pd.read_csv(data_dir / f"{prefix}_selected.csv", encoding="utf-8-sig")
    ids = selected["task_id"].astype(str).tolist()
    canonical, id_map = _canonicalize_id_map(problem, ids)
    starts = {id_map[old]: float(start)
              for old, start in zip(ids, selected["start_time_s"].astype(float))}
    values = evaluate_compact_solution(problem, canonical, starts)
    manifest_objectives = manifest.get("objectives", {})
    recorded_values = (
        float(manifest_objectives.get("F1", manifest.get("F1_weighted_lateness", math.nan))),
        float(manifest_objectives.get("Cmax_s", manifest.get("transport_cmax_s", math.nan))),
        float(manifest_objectives.get("E_kWh", manifest.get("transport_energy_kWh", math.nan))),
        int(manifest_objectives.get("N", manifest.get("selected_sorties", -1))),
    )
    tolerances = (1e-5, 1e-5, 1e-7, 0)
    if any(abs(float(a) - float(b)) > tol
           for a, b, tol in zip(values, recorded_values, tolerances)):
        raise ValueError(f"{prefix} selected CSV objectives disagree with its manifest")
    return {
        "sortie_ids": canonical, "starts": {k: int(round(v)) for k, v in starts.items()},
        "objectives": values, "solver_status": manifest.get("master_status", "FEASIBLE"),
        "origin_weight": tuple(1.0 if name == objective else 0.0 for name in OBJECTIVE_NAMES),
        "anchor_objective": objective,
        "proven_optimal": bool(manifest.get("anchor_optimal", False)),
        "status": "OPTIMAL" if manifest.get("anchor_optimal") else "BEST_KNOWN_FEASIBLE",
        "relative_gap": float(manifest.get("master_relative_gap", 0.0)),
    }


def load_compact_anchors(problem: dict, data_dir: Path | None = None) -> dict:
    return {name: load_compact_anchor(name, problem, data_dir) for name in ANCHOR_NAMES}


def check_anchor_compatibility(anchors: dict) -> bool:
    if set(anchors) != set(ANCHOR_NAMES):
        raise ValueError("Exactly the F1/Cmax/E/N compact anchors are required")
    for name, solution in anchors.items():
        if len(solution["objectives"]) != 4 or not solution["sortie_ids"]:
            raise ValueError(f"{name} anchor is incomplete")
        if set(solution["starts"]) != set(solution["sortie_ids"]):
            raise ValueError(f"{name} anchor start map does not match occurrences")
    if anchors["Cmax"]["proven_optimal"]:
        raise ValueError("Cmax anchor currently must be recorded as best-known feasible")
    return True


def evaluate_compact_solution(problem: dict, sortie_ids, starts: dict) -> tuple:
    ids = canonicalize_sortie_ids(problem, sortie_ids)
    if set(ids) != set(map(str, starts)):
        raise ValueError("Start times must match canonical sortie IDs")
    patterns = problem["patterns"].set_index("pattern_id")
    sorties, cmax, energy = [], 0.0, 0.0
    for sid in ids:
        pid = sid.rpartition("-")[0]
        row = patterns.loc[pid]
        start = float(starts[sid])
        sorties.append({"sortie_id": sid, "pattern_id": pid,
                        "class_counts": problem["pattern_class_counts"][pid],
                        "start_time_s": start})
        cmax = max(cmax, start + float(row.duration_s))
        energy += float(row.energy_kWh)
    f1 = class_timeliness(sorties, problem["classes"], problem["pattern_counts"])
    return float(f1), float(cmax), float(energy), int(len(ids))


def _make_local_slots(problem: dict, fixed_ids, residual: dict) -> list[dict]:
    fixed_counts = Counter(sid.rpartition("-")[0] for sid in fixed_ids)
    slots = []
    for sid in fixed_ids:
        pid = sid.rpartition("-")[0]
        slots.append({"sortie_id": sid, "pattern_id": pid,
                      "class_counts": problem["pattern_class_counts"][pid], "fixed": True})
    for pid in sorted(problem["pattern_class_counts"]):
        amounts = problem["pattern_class_counts"][pid]
        if any(amount > residual[cid] for cid, amount in amounts.items()):
            continue
        maximum = min(residual[cid] // amount for cid, amount in amounts.items())
        for copy in range(1, maximum + 1):
            slots.append({"sortie_id": f"{pid}-{fixed_counts[pid] + copy}",
                          "pattern_id": pid, "class_counts": amounts, "fixed": False})
    return slots


def build_compact_local_model(problem: dict, fixed_sortie_ids, residual_demand=None):

    fixed = canonicalize_sortie_ids(problem, fixed_sortie_ids)
    residual = (compute_residual_demand(problem, fixed) if residual_demand is None
                else dict(residual_demand))
    if set(residual) != set(problem["class_supply"]) or any(int(v) < 0 for v in residual.values()):
        raise ValueError("Residual demand must be nonnegative and cover every class")
    slots = _make_local_slots(problem, fixed, residual)
    pattern_rows = problem["patterns"].set_index("pattern_id")
    class_rows = problem["classes"].set_index("class_id")
    model = cp_model.CpModel()
    chosen, starts, active_ends = [], [], []
    flight_intervals, battery_intervals = defaultdict(list), defaultdict(list)
    f1_terms, energy_terms = [], []
    horizon = int(problem["horizon_s"])
    battery_horizon = horizon + math.ceil(max(
        charge_time_to_full(0, value) for value in problem["charge_full"].values()))
    fixed_supply = Counter()

    for i, slot in enumerate(slots):
        row, pid = pattern_rows.loc[slot["pattern_id"]], slot["pattern_id"]
        typ = str(row.uav_type)
        flight = max(1, math.ceil(float(row.duration_s)))
        soc = soc_after_task(float(row.energy_kWh), problem["energy_capacity"][typ])
        charge = charge_time_to_full(soc, problem["charge_full"][typ])
        battery = max(flight, math.ceil(float(row.duration_s) + charge))
        latest = horizon - flight
        if math.isfinite(float(row.latest_start_s)):
            latest = min(latest, math.floor(float(row.latest_start_s)))
        if latest < 0:
            raise ValueError(f"Occurrence has no feasible start domain: {slot['sortie_id']}")
        x, s = model.NewBoolVar(f"x_{i}"), model.NewIntVar(0, latest, f"s_{i}")
        f_end = model.NewIntVar(flight, horizon, f"fend_{i}")
        b_end = model.NewIntVar(battery, battery_horizon, f"bend_{i}")
        active = model.NewIntVar(0, horizon, f"active_{i}")
        if slot["fixed"]:
            model.Add(x == 1)
            model.Add(active == f_end)
            for cid, amount in slot["class_counts"].items():
                fixed_supply[cid] += amount
        else:
            model.Add(active == f_end).OnlyEnforceIf(x)
            model.Add(active == 0).OnlyEnforceIf(x.Not())
        model.Add(s == 0).OnlyEnforceIf(x.Not())
        model.Add(f_end == flight).OnlyEnforceIf(x.Not())
        model.Add(b_end == battery).OnlyEnforceIf(x.Not())
        flight_intervals[typ].append(model.NewOptionalIntervalVar(
            s, flight, f_end, x, f"flight_{i}"))
        battery_intervals[typ].append(model.NewOptionalIntervalVar(
            s, battery, b_end, x, f"battery_{i}"))
        chosen.append(x)
        starts.append(s)
        active_ends.append(active)
        energy_terms.append(int(round(float(row.energy_kWh) * 1_000_000)) * x)
        for cid, amount in slot["class_counts"].items():
            cls = class_rows.loc[cid]
            deadline, expected = float(cls.hard_deadline_s), float(cls.expected_time_s)
            offset = problem["delivery_offsets"][(pid, cid)]
            if math.isfinite(deadline):
                model.Add(s + math.ceil(offset) <= math.floor(deadline)).OnlyEnforceIf(x)
            if not math.isfinite(deadline) and math.isfinite(expected):
                offset_ms = math.ceil(1000 * (offset - expected))
                late = model.NewIntVar(0, max(0, 1000 * horizon + offset_ms),
                                       f"late_{i}_{cid}")
                model.Add(late >= 1000 * s + offset_ms).OnlyEnforceIf(x)
                model.Add(late == 0).OnlyEnforceIf(x.Not())
                f1_terms.append(int(amount) * int(cls.priority) * late)

    for cid, supply in problem["class_supply"].items():
        terms = [slot["class_counts"][cid] * chosen[i]
                 for i, slot in enumerate(slots) if not slot["fixed"]
                 and slot["class_counts"].get(cid, 0)]
        model.Add(fixed_supply[cid] + sum(terms) == supply)
        if residual[cid] and not terms:
            raise ValueError(f"No local pattern covers residual class {cid}")
    for typ, intervals in flight_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(problem["uav_ids"][typ]))
    for typ, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(problem["battery_ids"][typ]))
    cmax = model.NewIntVar(0, horizon, "cmax")
    model.AddMaxEquality(cmax, active_ends)
    objectives = {"F1": sum(f1_terms), "Cmax": cmax, "E": sum(energy_terms),
                  "N": sum(chosen)}
    return model, slots, chosen, starts, objectives


def add_compact_tchebycheff_objective(model, objective_vars, weight, ideal_point,
                                      ranges=OBJECTIVE_RANGES):
    if not (len(weight) == len(ideal_point) == len(ranges) == 4):
        raise ValueError("Four objective weights, ideal values and ranges required")
    distances, weighted = [], []
    for name, coefficient, ideal, span in zip(OBJECTIVE_NAMES, weight, ideal_point, ranges):
        unit = OBJECTIVE_UNIT[name]
        ideal_units = int(round(float(ideal) * unit))
        range_units = max(1, int(round(float(span) * unit)))
        distance = model.NewIntVar(-100_000_000, 100_000_000, f"norm_{name}")
        numerator = NORMALIZATION_SCALE * (objective_vars[name] - ideal_units)
        model.Add(distance * range_units >= numerator)
        model.Add(distance * range_units <= numerator + range_units - 1)
        distances.append(distance)
        weighted.append(int(round(float(coefficient) * 1_000_000)) * distance)
    peak = model.NewIntVar(-100_000_000_000_000, 100_000_000_000_000,
                           "tchebycheff_peak")
    for term in weighted:
        model.Add(peak >= term)
    model.Minimize(peak + 10_000 * sum(distances))
    return distances, peak


def compact_subproblem_cache_key(fixed_sortie_ids, residual_demand, weight,
                                 ideal_point, ranges):
    return (tuple(sorted(map(str, fixed_sortie_ids))),
            tuple(sorted((str(k), int(v)) for k, v in residual_demand.items())),
            tuple(round(float(x), 8) for x in weight),
            tuple(round(float(x), 6) for x in ideal_point),
            tuple(round(float(x), 6) for x in ranges))


def solve_compact_local_subproblem(problem, fixed_sortie_ids, weight, ideal_point,
                                   ranges=OBJECTIVE_RANGES, seed_solution=None,
                                   time_limit_s=2.0, workers=2, cache=None,
                                   random_seed=2026):
    import numpy as np
    fixed = canonicalize_sortie_ids(problem, fixed_sortie_ids)
    residual = compute_residual_demand(problem, fixed)
    key = compact_subproblem_cache_key(fixed, residual, weight, ideal_point, ranges)
    if cache is not None and key in cache:
        return {**cache[key], "cache_hit": True}
    model, slots, chosen, starts, objectives = build_compact_local_model(problem, fixed, residual)
    add_compact_tchebycheff_objective(model, objectives, weight, ideal_point, ranges)
    seed_solution = seed_solution or {}
    seed_ids = set(canonicalize_sortie_ids(problem, seed_solution.get("sortie_ids", ())))
    seed_starts = seed_solution.get("starts", {})
    for i, slot in enumerate(slots):
        if not slot["fixed"]:
            model.AddHint(chosen[i], int(slot["sortie_id"] in seed_ids))
        if slot["sortie_id"] in seed_starts:
            model.AddHint(starts[i], int(round(float(seed_starts[slot["sortie_id"]]))))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.01, float(time_limit_s))
    solver.parameters.num_search_workers = max(1, int(workers))
    solver.parameters.random_seed = int(random_seed)
    status_code = solver.Solve(model)
    result = {"solver_status": solver.StatusName(status_code),
              "wall_time_s": solver.WallTime(), "cache_hit": False}
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [slot for i, slot in enumerate(slots) if solver.Value(chosen[i])]
        ids, id_map = _canonicalize_id_map(problem, [slot["sortie_id"] for slot in selected])
        raw_starts = {slot["sortie_id"]: int(solver.Value(starts[i]))
                      for i, slot in enumerate(slots) if solver.Value(chosen[i])}
        start_map = {id_map[old]: raw_starts[old] for old in raw_starts}
        result.update({"sortie_ids": ids, "starts": start_map,
                       "objectives": evaluate_compact_solution(problem, ids, start_map)})
    if cache is not None and result["solver_status"] in ("OPTIMAL", "FEASIBLE", "INFEASIBLE"):
        cache[key] = dict(result)
    return result


def _destroy(problem, solution, weight, rng, fraction=None):
    import numpy as np
    ids = canonicalize_sortie_ids(problem, solution["sortie_ids"])
    if len(ids) <= 1:
        return tuple()
    n_remove = min(len(ids) - 1, max(2, math.ceil(
        float(fraction if fraction is not None else rng.uniform(0.15, 0.35)) * len(ids))))
    pattern_rows = problem["patterns"].set_index("pattern_id")
    class_rows = problem["classes"].set_index("class_id")
    scores = []
    for sid in ids:
        pid = sid.rpartition("-")[0]
        row, start = pattern_rows.loc[pid], float(solution.get("starts", {}).get(sid, 0.0))
        f1 = 0.0
        for cid, amount in problem["pattern_class_counts"][pid].items():
            cls = class_rows.loc[cid]
            expected, deadline = float(cls.expected_time_s), float(cls.hard_deadline_s)
            if not math.isfinite(deadline) and math.isfinite(expected):
                delivery = start + problem["delivery_offsets"][(pid, cid)]
                f1 += amount * int(cls.priority) * max(0, delivery - expected)
        cmax = start + float(row.duration_s)
        e_box = float(row.energy_kWh) / max(1, int(row.n_boxes))
        low_load = 1.0 / max(1, int(row.n_boxes))
        score = (float(weight[0]) * f1 / OBJECTIVE_RANGES[0]
                 + float(weight[1]) * cmax / OBJECTIVE_RANGES[1]
                 + float(weight[2]) * e_box / OBJECTIVE_RANGES[2]
                 + float(weight[3]) * low_load / OBJECTIVE_RANGES[3])
        scores.append(max(1e-9, score))
    p = np.asarray(scores, dtype=float)
    p /= p.sum()
    removed = set(map(str, rng.choice(np.asarray(ids, dtype=object),
                                      size=n_remove, replace=False, p=p)))
    return tuple(sorted(set(ids) - removed))


def destroy_and_recombine_compact(problem, parent_a, parent_b, weight, rng,
                                  weighted=True):
    a, b = set(parent_a["sortie_ids"]), set(parent_b["sortie_ids"])
    common = set(canonicalize_sortie_ids(problem, a & b))
    if len(common) <= 1:
        return tuple()
    reference = {**parent_a, "sortie_ids": tuple(sorted(common))}
    fixed = _destroy(problem, reference, weight if weighted else (0, 0, 0, 0), rng)
    return tuple(sorted(set(fixed) & common))


def _scalar_value(objectives, weight, ideal, ranges):
    d = [(float(f) - float(z)) / max(1e-9, float(r))
         for f, z, r in zip(objectives, ideal, ranges)]
    return max(float(w) * x for w, x in zip(weight, d)) + 0.01 * sum(d)


def _initial_population(problem, anchors, weights, cache, rng, local_time_s,
                        workers, stats):
    anchor_list = [anchors[name] for name in ANCHOR_NAMES]
    ideal = tuple(min(a["objectives"][j] for a in anchor_list) for j in range(4))
    pure = {tuple(1.0 if j == k else 0.0 for j in range(4)): anchors[name]
            for k, name in enumerate(ANCHOR_NAMES)}
    population = []
    for weight in weights:
        if tuple(weight) in pure:
            sol = dict(pure[tuple(weight)])
            sol["origin_weight"] = tuple(weight)
            population.append(sol)
            continue
        nearest = min(anchor_list, key=lambda anchor: sum(
            (float(weight[j]) - (1.0 if j == ANCHOR_NAMES.index(anchor["anchor_objective"])
                                 else 0.0)) ** 2 for j in range(4)))
        fixed = _destroy(problem, nearest, weight, rng)
        child = solve_compact_local_subproblem(
            problem, fixed, weight, ideal, OBJECTIVE_RANGES, nearest,
            local_time_s, workers, cache, int(rng.integers(1, 2**31 - 1)))
        if child.get("cache_hit"):
            stats["cache_hits"] += 1
        else:
            stats["solver_calls"] += 1
        if "objectives" not in child:
            stats["initialization_fallbacks"] += 1
            child = dict(nearest)
        child["origin_weight"] = tuple(weight)
        population.append(child)
    return population


def run_compact_moead(top_k=3, H=5, T=10, generations=20, nr=2,
                      local_time_s=2.0, polish_time_s=20.0, workers=2,
                      seed=2026, data_dir=DATA, verbose=True):
    import numpy as np
    problem = load_compact_problem(top_k, data_dir)
    anchors = load_compact_anchors(problem)
    check_anchor_compatibility(anchors)
    weights = generate_weights(4, int(H))
    neighbors = build_neighbors(weights, min(int(T), len(weights)))
    rng, cache = np.random.default_rng(int(seed)), {}
    stats = {"solver_calls": 0, "cache_hits": 0, "initialization_fallbacks": 0,
             "generation_log": [], "anchor_statuses": {
                 name: anchors[name]["status"] for name in ANCHOR_NAMES}}
    population = _initial_population(problem, anchors, weights, cache, rng,
                                     local_time_s, workers, stats)
    ideal = [min(sol["objectives"][j] for sol in population) for j in range(4)]
    archive, seen = [], set()
    for sol in population:
        signature = tuple(sol["sortie_ids"])
        if signature not in seen:
            update_archive(archive, sol, duplicate_tolerance=ARCHIVE_TOLERANCE)
            seen.add(signature)
    for generation in range(int(generations)):
        feasible = 0
        for i, weight in enumerate(weights):
            p1, p2 = rng.choice(neighbors[i], size=2, replace=True)
            parent_a, parent_b = population[int(p1)], population[int(p2)]
            fixed = destroy_and_recombine_compact(
                problem, parent_a, parent_b, weight, rng,
                weighted=bool(rng.integers(0, 2)))
            if not fixed:
                fixed = _destroy(problem, parent_a, weight, rng)
            child = solve_compact_local_subproblem(
                problem, fixed, weight, tuple(ideal), OBJECTIVE_RANGES, parent_a,
                local_time_s, workers, cache, int(rng.integers(1, 2**31 - 1)))
            if child.get("cache_hit"):
                stats["cache_hits"] += 1
            else:
                stats["solver_calls"] += 1
            if "objectives" not in child:
                continue
            feasible += 1
            child["origin_weight"] = tuple(weight)
            for j in range(4):
                ideal[j] = min(ideal[j], child["objectives"][j])
            signature = tuple(child["sortie_ids"])
            if signature not in seen:
                update_archive(archive, child, duplicate_tolerance=ARCHIVE_TOLERANCE)
                seen.add(signature)
            updated = 0
            for neighbor_index in neighbors[i]:
                current = population[neighbor_index]
                if _scalar_value(child["objectives"], weights[neighbor_index], ideal,
                                 OBJECTIVE_RANGES) < _scalar_value(
                                     current["objectives"], weights[neighbor_index], ideal,
                                     OBJECTIVE_RANGES):
                    population[neighbor_index] = dict(child)
                    updated += 1
                    if updated >= int(nr):
                        break
        row = {"generation": generation + 1, "archive_size": len(archive),
               "feasible_children": feasible, "ideal_point": json.dumps(ideal),
               "solver_calls": stats["solver_calls"], "cache_size": len(cache)}
        stats["generation_log"].append(row)
        if verbose:
            print(f"Compact MOEA/D generation {generation + 1}/{generations}: "
                  f"archive={len(archive)}, feasible={feasible}", flush=True)

    if float(polish_time_s) > 0 and archive:
        polished = []
        for original in list(archive):
            equal = (0.25,) * 4
            fixed_schedule = solve_compact_local_subproblem(
                problem, original["sortie_ids"], equal, tuple(ideal), OBJECTIVE_RANGES,
                original, max(1.0, float(polish_time_s) / 4), workers, cache,
                int(rng.integers(1, 2**31 - 1)))
            if fixed_schedule.get("cache_hit"):
                stats["cache_hits"] += 1
            else:
                stats["solver_calls"] += 1
            candidate = fixed_schedule if "objectives" in fixed_schedule else original
            fixed = _destroy(problem, candidate, equal, rng, fraction=0.2)
            lns = solve_compact_local_subproblem(
                problem, fixed, equal, tuple(ideal), OBJECTIVE_RANGES, candidate,
                max(1.0, float(polish_time_s) / 2), workers, cache,
                int(rng.integers(1, 2**31 - 1)))
            if lns.get("cache_hit"):
                stats["cache_hits"] += 1
            else:
                stats["solver_calls"] += 1
            polished.extend([candidate, lns] if "objectives" in lns else [candidate])
        archive, seen = [], set()
        for sol in polished:
            signature = tuple(sol["sortie_ids"])
            if signature not in seen:
                update_archive(archive, sol, duplicate_tolerance=ARCHIVE_TOLERANCE)
                seen.add(signature)
    stats.update({
        "H": int(H), "T": int(T), "nr": int(nr), "generations": int(generations),
        "local_time_s": float(local_time_s), "polish_time_s": float(polish_time_s),
        "workers": int(workers), "seed": int(seed), "weights": [list(w) for w in weights],
        "ideal_point": ideal, "cache_hits": stats["cache_hits"], "cache_size": len(cache),
        "pareto_size": len(archive), "top_k": int(top_k),
        "candidate_manifest_sha256": problem["candidate_manifest_sha256"],
        "anchor_objectives": {name: list(anchors[name]["objectives"]) for name in ANCHOR_NAMES},
        "anchor_statuses": {name: anchors[name]["status"] for name in ANCHOR_NAMES},
        "cmax_anchor_gap": anchors["Cmax"]["relative_gap"],
    })
    return population, archive, stats


def _groups_nonoverlap(frame, resource, start, end):
    for _, group in frame.sort_values(start).groupby(resource):
        if (group[start].iloc[1:].to_numpy()
                < group[end].iloc[:-1].to_numpy() - 1e-9).any():
            return False
    return True


def validate_compact_moead_solution(problem: dict, solution: dict) -> dict:
    ids = canonicalize_sortie_ids(problem, solution["sortie_ids"])
    starts = {str(k): int(round(float(v))) for k, v in solution["starts"].items()}
    totals = Counter()
    for sid in ids:
        totals.update(problem["pattern_class_counts"][sid.rpartition("-")[0]])
    checks = {"class_quantity_conservation": all(
        totals[cid] == amount for cid, amount in problem["class_supply"].items())}
    selected_sorties = [{"sortie_id": sid, "pattern_id": sid.rpartition("-")[0],
                         "class_counts": problem["pattern_class_counts"][sid.rpartition("-")[0]],
                         "start_time_s": starts[sid]} for sid in ids]
    deliveries = decode_box_deliveries(selected_sorties, problem["classes"],
                                       problem["pattern_counts"])
    checks["every_box_exactly_once"] = (len(deliveries) == 80 and deliveries["box_id"].is_unique
                                        and set(deliveries["box_id"]) == set(problem["boxes"]["box_id"]))
    class_by_box = {bid: row.class_id for row in problem["classes"].itertuples(index=False)
                    for bid in row.box_ids}
    checks["box_class_identity"] = all(class_by_box[r.box_id] == r.class_id
                                       for r in deliveries.itertuples(index=False))
    cls = problem["classes"].set_index("class_id")
    checks["hard_deadlines"] = all(
        not math.isfinite(float(cls.loc[r.class_id].hard_deadline_s))
        or float(r.delivery_time_s) <= math.floor(float(cls.loc[r.class_id].hard_deadline_s)) + 1e-6
        for r in deliveries.itertuples(index=False))
    objectives = evaluate_compact_solution(problem, ids, starts)
    checks["starts_integer"] = all(float(v).is_integer() for v in solution["starts"].values())
    tolerances = (1e-5, 1e-5, 1e-7, 0)
    checks["objective_values_match"] = all(
        abs(float(actual) - float(expected)) <= tol
        for actual, expected, tol in zip(objectives, solution["objectives"], tolerances))
    tasks, delivery_table = materialize_selected_sorties(
        selected_sorties, problem["patterns"], problem["pattern_counts"], problem["classes"])
    from Q2_compact_smoke import _fixed_transport_schedule
    resources = (problem["uav_ids"], problem["battery_ids"], problem["energy_capacity"],
                 problem["charge_full"])
    status, _, schedule = _fixed_transport_schedule(
        tasks, delivery_table, problem["boxes"], resources, starts, 10.0, 2)
    checks["transport_resource_schedule"] = status in ("FEASIBLE", "OPTIMAL") and schedule is not None
    if schedule is None:
        checks.update({"uav_nonoverlap": False, "battery_nonoverlap": False, "flight_horizon": False})
    else:
        checks["uav_nonoverlap"] = _groups_nonoverlap(schedule, "uav_id", "start_time_s", "flight_release_s")
        checks["battery_nonoverlap"] = _groups_nonoverlap(schedule, "battery_id", "start_time_s", "battery_release_s")
        checks["flight_horizon"] = bool((schedule["end_time_s"] <= problem["horizon_s"] + 1e-6).all())
    checks["F1_recomputed"] = abs(objectives[0] - class_timeliness(
        selected_sorties, problem["classes"], problem["pattern_counts"])) <= 1e-6
    expected_cmax = max(starts[sid] + float(problem["patterns"].set_index("pattern_id").loc[
        sid.rpartition("-")[0], "duration_s"]) for sid in ids)
    checks["Cmax_recomputed"] = abs(objectives[1] - expected_cmax) <= 1e-6
    checks["energy_recomputed"] = abs(objectives[2] - sum(
        float(problem["patterns"].set_index("pattern_id").loc[
            sid.rpartition("-")[0], "energy_kWh"]) for sid in ids)) <= 1e-7
    checks["sorties_recomputed"] = objectives[3] == len(ids)
    return {"checks": checks, "all_pass": all(checks.values()),
            "objectives": dict(zip(OBJECTIVE_NAMES, objectives)),
            "transport_status": status, "_schedule": schedule}


def save_compact_moead_results(population, archive, stats, data_dir=DATA):
    data_dir = Path(data_dir)
    output_root = data_dir / "q2_compact_moead"
    output_root.mkdir(parents=True, exist_ok=True)
    problem = load_compact_problem(stats["top_k"], data_dir)
    summary_rows, accepted = [], []
    for index, solution in enumerate(archive, start=1):
        validation = validate_compact_moead_solution(problem, solution)
        resource_schedule = validation.pop("_schedule", None)
        solution_id = f"P{index:03d}"
        objective = solution["objectives"]
        summary_rows.append({"solution_id": solution_id, "F1": objective[0],
                             "Cmax_s": objective[1], "energy_kWh": objective[2],
                             "sorties": objective[3],
                             "status": solution.get("solver_status", "FEASIBLE"),
                             "all_pass": validation["all_pass"]})
        if not validation["all_pass"]:
            continue
        selected_sorties = [{"sortie_id": sid, "pattern_id": sid.rpartition("-")[0],
                             "class_counts": problem["pattern_class_counts"][sid.rpartition("-")[0]],
                             "start_time_s": solution["starts"][sid]}
                            for sid in solution["sortie_ids"]]
        selected, deliveries = materialize_selected_sorties(
            selected_sorties, problem["patterns"], problem["pattern_counts"], problem["classes"])
        if resource_schedule is None:
            continue
        selected = selected.merge(
            resource_schedule[["task_id", "uav_id", "battery_id", "start_time_s"]],
            on="task_id", validate="one_to_one")
        directory = output_root / solution_id
        directory.mkdir(exist_ok=True)
        selected.to_csv(directory / "selected.csv", index=False, encoding="utf-8-sig")
        resource_schedule.to_csv(directory / "schedule.csv", index=False, encoding="utf-8-sig")
        deliveries.to_csv(directory / "deliveries.csv", index=False, encoding="utf-8-sig")
        (directory / "validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        accepted.append(solution_id)
    pareto = pd.DataFrame(summary_rows)
    if not pareto.empty:
        pareto = pareto.loc[pareto["all_pass"]]
    pareto.to_csv(data_dir / "Q2_compact_moead_pareto.csv", index=False, encoding="utf-8-sig")
    pop_rows = [{"F1": s["objectives"][0], "Cmax_s": s["objectives"][1],
                 "energy_kWh": s["objectives"][2], "sorties": s["objectives"][3],
                 "sortie_ids": "|".join(s["sortie_ids"])} for s in population]
    pd.DataFrame(pop_rows).to_csv(
        data_dir / "Q2_compact_moead_population.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stats["generation_log"]).to_csv(
        data_dir / "Q2_compact_moead_generation_log.csv", index=False, encoding="utf-8-sig")
    manifest = dict(stats)
    manifest.update({"objective_names": list(OBJECTIVE_NAMES),
                     "pareto_solution_ids": accepted, "pareto_size": len(accepted),
                     "candidate_sha256": problem["candidate_manifest_sha256"],
                     "anchor_sha256": {n: _sha256(data_dir / f"Q2_anchor_{n}_manifest.json")
                                       for n in ANCHOR_NAMES}})
    (data_dir / "Q2_compact_moead_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
