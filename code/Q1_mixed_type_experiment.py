import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""Exact Q1 single-objective experiments with a UAV type chosen per sortie.

Each sortie serves one service area and returns to O01. Cargo boxes are indivisible.
The experiment enforces each type's safe payload, volume, and energy limits, but
does not schedule individual aircraft or enforce fleet, battery, or deadlines.
"""

import csv
import hashlib
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
INPUT_NAMES = (
    "物资需求.csv",
    "运输无人机_机型参数.csv",
    "Q1_max_payload.csv",
    "Q1_single_objective.csv",
    "distance_matrix.csv",
    "climb_height_matrix.csv",
    "descent_height_matrix.csv",
)


def read_csv(name):
    with (DATA / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows, fields):
    with (DATA / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def route_matrix(name):
    return {row[""]: row for row in read_csv(name)}


def expand_boxes(cargo_rows, service):
    boxes = []
    for row in cargo_rows:
        if row["service"] != service:
            continue
        for _ in range(int(row["total_boxes"])):
            boxes.append({
                "id": f"{service}-B{len(boxes) + 1:03d}",
                "cargo_type": row["cargo_type"],
                "mass": float(row["mass_per_box"]),
                "volume": float(row["volume_per_box"]),
            })
    if not boxes:
        raise ValueError(f"No boxes for {service}")
    if len(boxes) > 20:
        raise ValueError(f"Exact subset solver requires at most 20 boxes: {service}")
    return boxes


def round_trip_energy(u, service, mass, distance, climb):
    """Same outbound-loaded, return-empty formula as src.physics."""
    q_max = float(u["Q_g"])
    l_zero = float(u["L_0"])
    l_full = float(u["L_F"])
    e_use = float(u["E_use"])
    empty_mass = float(u["M_g0"])
    efficiency = float(u["η_up"])
    equivalent_range = l_zero - (l_zero - l_full) * (mass / q_max) ** 1.5
    outbound = (
        e_use * float(distance["O01"][service]) / equivalent_range
        + (empty_mass + mass) * 9.81 * float(climb["O01"][service])
        / (3.6e6 * efficiency)
    )
    inbound = (
        e_use * float(distance[service]["O01"]) / l_zero
        + empty_mass * 9.81 * float(climb[service]["O01"])
        / (3.6e6 * efficiency)
    )
    return outbound + inbound


def sortie_time(u, service, n_boxes, distance, climb, descent):
    def leg(origin, destination):
        return (
            float(climb[origin][destination]) / float(u["v_k^up"])
            + float(distance[origin][destination]) / float(u["v_k^cr"])
            + float(descent[origin][destination]) / float(u["v_k^down"])
        )

    return (
        float(u["T_setup"])
        + float(u["T_load"]) * n_boxes
        + leg("O01", service)
        + leg(service, "O01")
        + float(u["T_handover"])
        + float(u["T_handover_p"]) * n_boxes
    )


def build_candidates(boxes, service, uavs, safe_limits, distance, climb, descent):
    size = 1 << len(boxes)
    masses = [0.0] * size
    volumes = [0.0] * size
    box_counts = [0] * size
    candidates = {uav_type: [None] * size for uav_type in uavs}
    feasible_counts = {uav_type: 0 for uav_type in uavs}

    for mask in range(1, size):
        bit = mask & -mask
        index = bit.bit_length() - 1
        previous = mask ^ bit
        masses[mask] = masses[previous] + boxes[index]["mass"]
        volumes[mask] = volumes[previous] + boxes[index]["volume"]
        box_counts[mask] = box_counts[previous] + 1

        for uav_type, u in uavs.items():
            safe_mass = min(float(u["Q_g"]), safe_limits[(uav_type, service)])
            if masses[mask] > safe_mass + 1e-9:
                continue
            if volumes[mask] > float(u["V_g"]) + 1e-9:
                continue
            energy = round_trip_energy(u, service, masses[mask], distance, climb)
            available = (1.0 - float(u["ρ_g"]) / 100.0) * float(u["E_use"])
            if energy > available + 1e-9:
                continue
            duration = sortie_time(
                u, service, box_counts[mask], distance, climb, descent
            )
            candidates[uav_type][mask] = (energy, duration)
            feasible_counts[uav_type] += 1

    return candidates, masses, volumes, box_counts, feasible_counts


def solve_partition(candidates, allowed_types, objective="E"):
    """Exact set partition DP for E, N, or cumulative T."""
    if objective not in ("E", "N", "T"):
        raise ValueError(f"Unknown objective: {objective}")
    allowed_types = tuple(sorted(allowed_types))
    full_mask = len(next(iter(candidates.values()))) - 1
    best_batch = [None] * (full_mask + 1)
    for mask in range(1, full_mask + 1):
        for uav_type in allowed_types:
            metrics = candidates[uav_type][mask]
            if metrics is None:
                continue
            choice = (metrics[0], metrics[1], uav_type)
            if best_batch[mask] is None:
                best_batch[mask] = choice
            elif objective == "T":
                if (choice[1], choice[0], choice[2]) < (
                    best_batch[mask][1], best_batch[mask][0], best_batch[mask][2]
                ):
                    best_batch[mask] = choice
            elif choice < best_batch[mask]:
                best_batch[mask] = choice

    def objective_key(result):
        energy, sorties, duration = result[:3]
        if objective == "N":
            return (sorties, energy, duration)
        if objective == "T":
            return (duration, energy, sorties)
        return (energy, sorties, duration)

    @lru_cache(maxsize=None)
    def solve(remaining):
        if remaining == 0:
            return (0.0, 0, 0.0, ())
        first = remaining & -remaining
        subset = remaining
        best = None
        while subset:
            batch = best_batch[subset]
            if subset & first and batch is not None:
                tail = solve(remaining ^ subset)
                candidate = (
                    batch[0] + tail[0],
                    1 + tail[1],
                    batch[1] + tail[2],
                    ((subset, batch[2]),) + tail[3],
                )
                if best is None or objective_key(candidate) < objective_key(best):
                    best = candidate
            subset = (subset - 1) & remaining
        if best is None:
            raise ValueError("No feasible partition for the allowed UAV types")
        return best

    return solve(full_mask)


def validate_solution(solution, boxes, service, candidates, masses, volumes, uavs, safe_limits):
    full_mask = (1 << len(boxes)) - 1
    covered = 0
    energy = 0.0
    duration = 0.0
    for mask, uav_type in solution[3]:
        if covered & mask:
            raise AssertionError(f"Box assigned twice in {service}")
        covered |= mask
        metrics = candidates[uav_type][mask]
        if metrics is None:
            raise AssertionError(f"Infeasible sortie in {service}")
        u = uavs[uav_type]
        if masses[mask] > safe_limits[(uav_type, service)] + 1e-9:
            raise AssertionError(f"Payload limit exceeded in {service}")
        if volumes[mask] > float(u["V_g"]) + 1e-9:
            raise AssertionError(f"Volume limit exceeded in {service}")
        energy += metrics[0]
        duration += metrics[1]
    if covered != full_mask or not math.isclose(energy, solution[0], abs_tol=1e-8):
        raise AssertionError(f"Coverage or energy mismatch in {service}")
    if not math.isclose(duration, solution[2], abs_tol=1e-8):
        raise AssertionError(f"Time mismatch in {service}")


def solution_plan_rows(solution, boxes, service, candidates, masses, volumes,
                       box_counts, uavs, safe_limits):
    rows = []
    for sortie_number, (mask, uav_type) in enumerate(solution[3], start=1):
        chosen = [box for i, box in enumerate(boxes) if mask & (1 << i)]
        cargo_counts = Counter(box["cargo_type"] for box in chosen)
        metrics = candidates[uav_type][mask]
        u = uavs[uav_type]
        rows.append({
            "service": service,
            "sortie_id": f"{service}-{sortie_number:02d}",
            "type": uav_type,
            "box_ids": "|".join(box["id"] for box in chosen),
            "cargo_counts": "|".join(
                f"{name}:{count}" for name, count in sorted(cargo_counts.items())
            ),
            "n_boxes": box_counts[mask],
            "mass_kg": masses[mask], "volume_m3": volumes[mask],
            "energy_kWh": metrics[0], "time_s": metrics[1],
            "safe_payload_kg": safe_limits[(uav_type, service)],
            "available_energy_kWh": (
                1.0 - float(u["ρ_g"]) / 100.0
            ) * float(u["E_use"]),
        })
    return rows


def input_manifest():
    files = {}
    for name in INPUT_NAMES:
        files[f"data/{name}"] = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    return {
        "experiment": "Q1 mixed UAV type per sortie, three single objectives",
        "run_command": "python code/Q1_mixed_type_experiment.py",
        "solver": "exact subset dynamic programming; deterministic; no random seed",
        "objectives": {
            "E-opt": "min E, then N, then cumulative T",
            "N-opt": "min N, then E, then cumulative T",
            "T-opt": "min cumulative T, then E, then N",
        },
        "python_version": sys.version.split()[0],
        "scope": (
            "O01-service-O01; indivisible boxes; each sortie chooses A/B/C; "
            "safe payload, volume and available energy enforced; "
            "no fleet scheduling, shared-battery or deadline constraints"
        ),
        "input_sha256": files,
    }


def main():
    cargo = read_csv("物资需求.csv")
    uavs = {row["type"]: row for row in read_csv("运输无人机_机型参数.csv")}
    safe_limits = {
        (row["type"], row["service"]): float(row["m_energy"])
        for row in read_csv("Q1_max_payload.csv")
    }
    distance = route_matrix("distance_matrix.csv")
    climb = route_matrix("climb_height_matrix.csv")
    descent = route_matrix("descent_height_matrix.csv")
    services = sorted({row["service"] for row in cargo})
    strategies = tuple(sorted(uavs)) + ("Mixed",)
    reference_energy = {
        (row["type"], row["service"]): float(row["E_total_kWh"])
        for row in read_csv("Q1_single_objective.csv")
        if row["strategy"] == "E-opt"
    }
    plan_rows = []
    n_plan_rows = []
    t_plan_rows = []
    summary_rows = []
    comparison_rows = []
    objective_rows = []
    totals = {name: {"N": 0, "E": 0.0, "T": 0.0} for name in strategies}
    objective_totals = {
        name: {"N": 0, "E": 0.0, "T": 0.0, "boxes": 0}
        for name in ("N-opt", "E-opt", "T-opt")
    }
    objective_type_counts = {
        name: Counter() for name in objective_totals
    }
    total_type_counts = Counter()

    for service in services:
        boxes = expand_boxes(cargo, service)
        candidates, masses, volumes, box_counts, feasible_counts = build_candidates(
            boxes, service, uavs, safe_limits, distance, climb, descent
        )
        solutions = {}
        for name in strategies:
            allowed = uavs if name == "Mixed" else (name,)
            solution = solve_partition(candidates, allowed)
            validate_solution(
                solution, boxes, service, candidates, masses, volumes,
                uavs, safe_limits,
            )
            solutions[name] = solution
            totals[name]["N"] += solution[1]
            totals[name]["E"] += solution[0]
            totals[name]["T"] += solution[2]
            if name != "Mixed" and (name, service) in reference_energy:
                if not math.isclose(
                    solution[0], reference_energy[(name, service)], abs_tol=1e-6
                ):
                    raise AssertionError(
                        f"Fixed-type E-opt mismatch for {name}/{service}"
                    )

        for name in strategies:
            solution = solutions[name]
            comparison_rows.append({
                "service": service, "strategy": name, "N_f": solution[1],
                "E_total_kWh": solution[0], "T_total_s": solution[2],
                "delta_E_vs_C_kWh": solution[0] - solutions["C"][0],
            })

        objective_solutions = {
            "E-opt": solutions["Mixed"],
            "N-opt": solve_partition(candidates, uavs, objective="N"),
            "T-opt": solve_partition(candidates, uavs, objective="T"),
        }
        for label, solution in objective_solutions.items():
            validate_solution(
                solution, boxes, service, candidates, masses, volumes,
                uavs, safe_limits,
            )
            type_counts = Counter(uav_type for _, uav_type in solution[3])
            objective_type_counts[label].update(type_counts)
            objective_totals[label]["N"] += solution[1]
            objective_totals[label]["E"] += solution[0]
            objective_totals[label]["T"] += solution[2]
            objective_totals[label]["boxes"] += len(boxes)
            objective_rows.append({
                "service": service, "objective": label,
                "total_boxes": len(boxes),
                "N_f": solution[1], "E_total_kWh": solution[0],
                "T_total_s": solution[2],
                "A_sorties": type_counts["A"],
                "B_sorties": type_counts["B"],
                "C_sorties": type_counts["C"],
            })
            if label != "E-opt":
                comparison_rows.append({
                    "service": service, "strategy": f"Mixed-{label[0]}",
                    "N_f": solution[1], "E_total_kWh": solution[0],
                    "T_total_s": solution[2],
                    "delta_E_vs_C_kWh": solution[0] - solutions["C"][0],
                })

        mixed = objective_solutions["E-opt"]
        type_counts = Counter(uav_type for _, uav_type in mixed[3])
        total_type_counts.update(type_counts)
        summary_rows.append({
            "service": service, "total_boxes": len(boxes),
            "N_f": mixed[1], "E_total_kWh": mixed[0],
            "T_total_s": mixed[2],
            "A_sorties": type_counts["A"], "B_sorties": type_counts["B"],
            "C_sorties": type_counts["C"],
            "E_C_only_kWh": solutions["C"][0],
            "E_saving_vs_C_kWh": solutions["C"][0] - mixed[0],
            "feasible_batches_A": feasible_counts["A"],
            "feasible_batches_B": feasible_counts["B"],
            "feasible_batches_C": feasible_counts["C"],
        })

        for label, target in (
            ("E-opt", plan_rows),
            ("N-opt", n_plan_rows),
            ("T-opt", t_plan_rows),
        ):
            target.extend(solution_plan_rows(
                objective_solutions[label], boxes, service, candidates,
                masses, volumes, box_counts, uavs, safe_limits,
            ))
        print(
            f"{service}: {mixed[1]} sorties, {mixed[0]:.6f} kWh, "
            f"types A/B/C={type_counts['A']}/{type_counts['B']}/{type_counts['C']}"
        )

    comparison_rows.extend({
        "service": "all", "strategy": name,
        "N_f": totals[name]["N"],
        "E_total_kWh": totals[name]["E"],
        "T_total_s": totals[name]["T"],
        "delta_E_vs_C_kWh": totals[name]["E"] - totals["C"]["E"],
    } for name in strategies)
    for label in ("N-opt", "T-opt"):
        result = objective_totals[label]
        comparison_rows.append({
            "service": "all", "strategy": f"Mixed-{label[0]}",
            "N_f": result["N"], "E_total_kWh": result["E"],
            "T_total_s": result["T"],
            "delta_E_vs_C_kWh": result["E"] - totals["C"]["E"],
        })
    for label, result in objective_totals.items():
        type_counts = objective_type_counts[label]
        objective_rows.append({
            "service": "all", "objective": label,
            "total_boxes": result["boxes"],
            "N_f": result["N"], "E_total_kWh": result["E"],
            "T_total_s": result["T"],
            "A_sorties": type_counts["A"],
            "B_sorties": type_counts["B"],
            "C_sorties": type_counts["C"],
        })
    summary_rows.append({
        "service": "all", "total_boxes": sum(row["total_boxes"] for row in summary_rows),
        "N_f": totals["Mixed"]["N"],
        "E_total_kWh": totals["Mixed"]["E"],
        "T_total_s": totals["Mixed"]["T"],
        "A_sorties": total_type_counts["A"],
        "B_sorties": total_type_counts["B"],
        "C_sorties": total_type_counts["C"],
        "E_C_only_kWh": totals["C"]["E"],
        "E_saving_vs_C_kWh": totals["C"]["E"] - totals["Mixed"]["E"],
        "feasible_batches_A": sum(row["feasible_batches_A"] for row in summary_rows),
        "feasible_batches_B": sum(row["feasible_batches_B"] for row in summary_rows),
        "feasible_batches_C": sum(row["feasible_batches_C"] for row in summary_rows),
    })

    write_csv("Q1_mixed_type_plan.csv", plan_rows, list(plan_rows[0]))
    write_csv("Q1_mixed_type_N_opt_plan.csv", n_plan_rows, list(n_plan_rows[0]))
    write_csv("Q1_mixed_type_T_opt_plan.csv", t_plan_rows, list(t_plan_rows[0]))
    write_csv("Q1_mixed_type_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv("Q1_mixed_type_comparison.csv", comparison_rows, list(comparison_rows[0]))
    write_csv("Q1_mixed_type_objectives.csv", objective_rows, list(objective_rows[0]))
    with (DATA / "Q1_mixed_type_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(input_manifest(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for label in ("N-opt", "E-opt", "T-opt"):
        result = objective_totals[label]
        print(
            f"TOTAL {label}: {result['N']} sorties, "
            f"{result['E']:.9f} kWh, {result['T']:.3f} cumulative seconds"
        )


if __name__ == "__main__":
    main()
