u"""纯 MOEA/D 多目标进化算法核心。

不包含任何 Q2/Q3/Q4 业务逻辑，仅负责：
  - 权重向量生成 (simplex lattice)
  - 邻域构建 (欧氏距离)
  - Tchebycheff 标量化
  - Pareto 支配与档案维护
  - 种群管理与子问题更新

设计目标:
  - Q2/Q3/Q4 共用同一套 MOEA/D 核心
  - 问题专用逻辑通过 callback/子类 注入
"""

import math
from collections import defaultdict
from itertools import combinations_with_replacement
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np


def generate_weights(m: int, H: int) -> List[Tuple[float, ...]]:
    u"""Simplex lattice 生成 m 目标、H 分割的权重向量。

    Args:
        m: 目标数
        H: 分割数

    Returns:
        权重列表，每个权重各分量和为 1.0。
    """
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
    u"""按权重向量的欧氏距离构建邻域。

    Args:
        weights: 权重列表
        T: 邻域大小（含自身）

    Returns:
        neighbors[i]: 按距离排序的邻域索引列表，长度为 T。
    """
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
    u"""Augmented Tchebycheff 标量化函数。

    g = max_j { w_j * |f_j - z_j*| / r_j } + rho * Σ_j w_j * |f_j - z_j*| / r_j
    """
    value = 0.0
    augment = 0.0
    for j in range(len(objectives)):
        dj = abs(objectives[j] - ideal_point[j]) / max(ranges[j], 1e-9)
        term = weight[j] * dj
        value = max(value, term)
        augment += term
    return value + rho * augment


def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    u"""判断 a 是否 Pareto-支配 b（所有分量 ≤ b，且至少一个严格 <）。"""
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
) -> bool:
    u"""向 Pareto 档案添加新解，维护非支配集。

    Args:
        archive: 当前档案列表，原地修改
        solution: 候选解 dict，需至少包含 "objectives" 键
        key_fn: 提取目标向量的函数，默认取 solution["objectives"]

    Returns:
        True 若解被加入档案。
    """
    obj = key_fn(solution) if key_fn else solution["objectives"]

    for existing in list(archive):
        existing_obj = key_fn(existing) if key_fn else existing["objectives"]
        if dominates(existing_obj, obj):
            return False
        if dominates(obj, existing_obj):
            archive.remove(existing)

    archive.append(dict(solution))
    return True


def select_parent(neighbor_indices: List[int], rng: np.random.Generator) -> int:
    u"""从邻域中随机选择一个父代索引。"""
    return int(rng.choice(neighbor_indices))


def select_parents(
    neighbor_indices: List[int], rng: np.random.Generator
) -> Tuple[int, int]:
    u"""从邻域中随机选择两个不同的父代索引。"""
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
    u"""标准 MOEA/D 主循环。

    Args:
        n_obj: 目标数
        H: 权重分割数
        T: 邻域大小
        max_generations: 最大代数
        nr: 每个后代最多替换的邻居数
        random_seed: 随机种子
        initializer: 生成初始种群的函数 (weights, rng) -> [solution, ...]
        evaluator: 评估/改进方案的函数 (solution, weight) -> solution
        objective_ranges: 归一化尺度
        rho: Tchebycheff 增强项系数
        verbose: 是否输出进度

    Returns:
        (population, archive, stats):
            population: 最终种群（每子问题一个解）
            archive: Pareto 档案
            stats: 收敛统计
    """
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