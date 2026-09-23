u"""
Pareto 多目标货箱组批优化
=========================

ε-约束 + 加权法生成非支配解集, 目标: min (N_f, ΣE, ΣT)
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def _build_constraint_matrix(batches, box_count):
    u"""构建覆盖矩阵 A: A[j,b]=1 当货箱 j 在组合 b 中"""
    n = len(batches)
    A = np.zeros((box_count, n))
    for b_idx, batch in enumerate(batches):
        for j in batch["indices"]:
            A[j, b_idx] = 1.0
    return A


def _solve_one(c_vec, A, bounds, integrality,
               A_extra=None, lb_extra=None, ub_extra=None,
               time_limit=10):
    u"""求解单次 MILP, 返回 selected 索引列表或 None"""
    if A_extra is not None:
        A_full = np.vstack([A, A_extra])
        lb_full = np.concatenate([np.ones(A.shape[0]), lb_extra])
        ub_full = np.concatenate([np.ones(A.shape[0]), ub_extra])
    else:
        A_full = A
        lb_full = np.ones(A.shape[0])
        ub_full = np.ones(A.shape[0])

    con = LinearConstraint(A_full, lb_full, ub_full)
    res = milp(c=c_vec, constraints=con, bounds=bounds,
               integrality=integrality,
               options={"disp": False, "time_limit": time_limit})
    if not res.success:
        return None
    selected = [b for b, x in enumerate(res.x) if x > 0.5]
    return selected


def _is_dominated(point, all_points):
    u"""point 被 all_points 中任一点支配则返回 True (三个指标均 min)"""
    p = np.array(point, dtype=float)
    for q in all_points:
        qa = np.array(q, dtype=float)
        if np.all(qa <= p) and np.any(qa < p):
            return True
    return False


# 加权向量: 覆盖 (N, E, T) 空间的不同区域
_WEIGHT_VECTORS = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 5, 0),
    (1, 0, 5),
    (1, 10, 0),
    (1, 0, 10),
    (1, 1, 1),
]


def solve_pareto_frontier(batches, box_count, energies, times,
                          n_extra_E=3, n_extra_T=3):
    u"""
    生成 Pareto 前沿 (N_f, ΣE, ΣT) 的非支配集.

    策略:
      1. 多组加权向量求 MILP → 覆盖凸包点
      2. ε-约束: 在 [E_min, E_max] 和 [T_min, T_max] 上各取 n_extra 个限值,
         固定限值下用 min(N_f) 搜索 → 补充非凸点
      3. 去重 + 非支配过滤

    返回: [{"N": int, "E": float, "T": float, "selected": [b_idx]}]
    """
    n_batches = len(batches)
    if n_batches == 0:
        return []
    if n_batches == 1:
        return [{"N": 1, "E": energies[0], "T": times[0], "selected": [0]}]

    A = _build_constraint_matrix(batches, box_count)
    bounds = Bounds(np.zeros(n_batches), np.ones(n_batches))
    integrality = np.ones(n_batches, dtype=int)

    E_arr = np.array([energies[b] for b in range(n_batches)])
    T_arr = np.array([times[b] for b in range(n_batches)])
    ones = np.ones(n_batches)

    raw = []

    large = n_batches > 5000
    wvs = _WEIGHT_VECTORS
    tl_weighted = max(10, min(30, n_batches // 600))
    if large:
        wvs = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1)]
        tl_weighted = max(30, min(90, n_batches // 200))
    for w_N, w_E, w_T in wvs:
        c_vec = w_N * ones + w_E * E_arr + w_T * T_arr
        sel = _solve_one(c_vec, A, bounds, integrality, time_limit=tl_weighted)
        if sel:
            raw.append((
                len(sel),
                sum(energies[b] for b in sel),
                sum(times[b] for b in sel),
                tuple(sorted(sel)),
            ))

    if not raw:
        return []

    E_vals = [r[1] for r in raw]
    E_lo, E_hi = min(E_vals), max(E_vals)
    tl_eps = min(tl_weighted, 30)
    if E_hi - E_lo > 1e-6:
        for E_lim in np.linspace(E_lo, E_hi, n_extra_E + 2)[1:-1]:
            sel = _solve_one(ones, A, bounds, integrality,
                             A_extra=E_arr.reshape(1, -1),
                             lb_extra=np.array([-np.inf]),
                             ub_extra=np.array([E_lim]),
                             time_limit=tl_eps)
            if sel:
                raw.append((
                    len(sel),
                    sum(energies[b] for b in sel),
                    sum(times[b] for b in sel),
                    tuple(sorted(sel)),
                ))

    T_vals = [r[2] for r in raw]
    T_lo, T_hi = min(T_vals), max(T_vals)
    if T_hi - T_lo > 1e-6:
        for T_lim in np.linspace(T_lo, T_hi, n_extra_T + 2)[1:-1]:
            sel = _solve_one(ones, A, bounds, integrality,
                             A_extra=T_arr.reshape(1, -1),
                             lb_extra=np.array([-np.inf]),
                             ub_extra=np.array([T_lim]),
                             time_limit=tl_eps)
            if sel:
                raw.append((
                    len(sel),
                    sum(energies[b] for b in sel),
                    sum(times[b] for b in sel),
                    tuple(sorted(sel)),
                ))

    seen = set()
    unique = []
    for r in raw:
        key = (r[3], r[0], round(r[1], 6), round(r[2], 6))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    obj_arr = np.array([[r[0], r[1], r[2]] for r in unique])
    keep = []
    for i in range(len(unique)):
        if not _is_dominated(obj_arr[i], obj_arr):
            keep.append({
                "N": unique[i][0],
                "E": unique[i][1],
                "T": unique[i][2],
                "selected": list(unique[i][3]),
            })

    return keep