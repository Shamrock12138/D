













import math
from collections import defaultdict
from itertools import combinations_with_replacement
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np


def generate_weights(m: int, H: int) -> List[Tuple[float, ...]]:
    








    weights = []
    for combo in combinations_with_replacement(range(H + 1), m - 1):
        combo = sorted(combo)
        points = [combo[0]]
        for i in range(1, m - 1):
            points.append(combo[i] - combo[i - 1])
        points.append(H - combo[-1])
        lam = tuple(p / H for p in points)
        weights.append(lam)
    return weights


def build_neighbors(weights: List[Tuple[float, ...]], T: int) -> List[List[int]]:
    








    n = len(weights)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = np.array(weights[i]) - np.array(weights[j])
            dist[i, j] = np.sqrt(np.dot(diff, diff))
    neighbors = []
    for i in range(n):
        order = np.argsort(dist[i])[:T]
        neighbors.append(list(order))
    return neighbors


def tchebycheff(
    objectives: Tuple[float, ...],
    weight: Tuple[float, ...],
    ideal_point: Tuple[float, ...],
    ranges: Tuple[float, ...],
    rho: float = 0.01,
) -> float:
    



    value = 0.0
    augment = 0.0
    for j in range(len(objectives)):
        dj = abs(objectives[j] - ideal_point[j]) / max(ranges[j], 1e-9)
        term = weight[j] * dj
        value = max(value, term)
        augment += term
    return value + rho * augment


def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:

    at_least_one_strict = False
    for va, vb in zip(a, b):
        if va > vb:
            return False
        if va < vb:
            at_least_one_strict = True
    return at_least_one_strict


def update_archive(
    archive: List[Dict],
    solution: Dict,
    key_fn: Optional[Callable[[Dict], Tuple[float, ...]]] = None,
    duplicate_tolerance: Optional[Tuple[float, ...]] = None,
) -> bool:
    










    obj = key_fn(solution) if key_fn else solution["objectives"]

    for existing in list(archive):
        existing_obj = key_fn(existing) if key_fn else existing["objectives"]
        tolerances = duplicate_tolerance or (0.0,) * len(obj)
        if len(tolerances) != len(obj):
            raise ValueError("duplicate_tolerance 与目标维数不一致")
        if all(abs(float(a) - float(b)) <= float(tol)
               for a, b, tol in zip(existing_obj, obj, tolerances)):
            if dominates(obj, existing_obj):
                archive.remove(existing)
                continue
            return False
        if dominates(existing_obj, obj):
            return False
        if dominates(obj, existing_obj):
            archive.remove(existing)

    archive.append(dict(solution))
    return True


def select_parent(neighbor_indices: List[int], rng: np.random.Generator) -> int:

    return int(rng.choice(neighbor_indices))


def select_parents(
    neighbor_indices: List[int], rng: np.random.Generator
) -> Tuple[int, int]:

    if len(neighbor_indices) < 2:
        return neighbor_indices[0], neighbor_indices[0]
    choices = rng.choice(neighbor_indices, size=2, replace=False).tolist()
    p1, p2 = int(choices[0]), int(choices[1])
    return p1, p2


def run_standard_moead(
    n_obj: int,
    H: int,
    T: int,
    max_generations: int,
    nr: int,
    random_seed: int,
    initializer: Callable[[List[Tuple[float, ...]], np.random.Generator], List[Dict]],
    evaluator: Callable[[Dict, Tuple[float, ...]], Dict],
    objective_ranges: Tuple[float, ...],
    rho: float = 0.01,
    verbose: bool = True,
) -> Tuple[List[Dict], List[Dict], Dict]:
    




















    rng = np.random.default_rng(random_seed)
    weights = generate_weights(n_obj, H)
    M = len(weights)
    neighbors = build_neighbors(weights, T)

    if verbose:
        print(f"MOEA/D: {n_obj}obj H={H} pop={M} T={T} gen={max_generations}")

    population = initializer(weights, rng)
    if len(population) != M:
        raise ValueError(f"初始化器应返回 {M} 个解，实际 {len(population)}")

    ideal_point = list(population[0]["objectives"])
    for sol in population:
        for j in range(n_obj):
            if sol["objectives"][j] < ideal_point[j]:
                ideal_point[j] = sol["objectives"][j]
    ideal_point = tuple(ideal_point)

    archive: List[Dict] = []
    for sol in population:
        update_archive(archive, sol)

    stats = {
        "generation": [],
        "archive_size": [],
        "ideal_point": [],
        "n_evals": 0,
    }

    for gen in range(max_generations):
        for i in range(M):
            p1, p2 = select_parents(neighbors[i], rng)
            parent_a = population[p1]
            parent_b = population[p2]

            child = evaluator(parent_a, parent_b, weights[i])
            if child is None:
                continue

            stats["n_evals"] += 1

            obj = child["objectives"]
            for j in range(n_obj):
                if obj[j] < ideal_point[j]:
                    ideal_point[j] = obj[j]
            ideal_tuple = tuple(ideal_point)

            update_archive(archive, child)

            replacements = 0
            for j in neighbors[i]:
                g_child = tchebycheff(
                    obj, weights[j], ideal_tuple, objective_ranges, rho,
                )
                g_existing = tchebycheff(
                    population[j]["objectives"], weights[j],
                    ideal_tuple, objective_ranges, rho,
                )
                if g_child <= g_existing:
                    population[j] = child
                    replacements += 1
                    if replacements >= nr:
                        break

        stats["generation"].append(gen + 1)
        stats["archive_size"].append(len(archive))
        stats["ideal_point"].append(ideal_tuple)

        if verbose and (gen + 1) % max(1, max_generations // 10) == 0:
            print(
                f"  gen {gen + 1}/{max_generations} | "
                f"archive={len(archive)} | "
                f"ideal={tuple(round(v, 1) for v in ideal_tuple)}",
                flush=True,
            )

    stats["final_archive_size"] = len(archive)
    return population, archive, stats
