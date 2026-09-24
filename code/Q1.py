import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hashlib
import json
import math
from collections import Counter
from functools import lru_cache
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
from time import time

from src.physics import TransportPhysicsModel, load_models

r"""
问题一：单点往返运输能力与货箱组批方案
======================================

O01→Si→O01 直接往返，单服务区、不可跨服务区组批。

物理模型（src/physics）
-----------------------
  等效航程: L_g(q) = L_0 - (L_0 - L_F)*(q/Q_g)^{3/2}
  水平能耗: E_hor = E_use * d / L_g(q)               [kWh]
  爬升能耗: E_up  = (M_g0+q)*g*H_up / (3.6e6*η_up)   [kWh]
  下降能耗: 0

────────────────────────── 输出文件 ──────────────────────────

步 1 - 最大安全载荷  -> Q1_fixed_max_payload.csv
  每 (type, service): m_energy, E_round_kWh, 绑定约束（质量/能量）

步 2 - 固定机型三目标基准  -> Q1_fixed_single_objective.csv
  min N_f / min ΣE / min ΣT, 每 (type, service, strategy): N_f, E, T

步 3 - 混合机型三目标优化  -> Q1_mixed_{N,E,T}_opt_plan.csv + Q1_mixed_objectives.csv
                              + Q1_mixed_comparison.csv + Q1_mixed_manifest.json
  每架次任选 A/B/C, 精确集合划分 DP 求 N/E/T 极值

步 4 - ρ 敏感性  -> Q1_sensitivity_payload_summary.csv + Q1_sensitivity_payload_detail.csv
                    + Q1_sensitivity_mixed_summary.csv + Q1_sensitivity_mixed_plan.csv
  payload_summary: 每 (rho, type) 质量/能量受限区数 & m_max 均值/最小/最大
  payload_detail:  每 (rho, type, service) 最大安全载荷明细
  mixed_summary:   每 (rho, service, objective) N_f, E, T, A/B/C 架次
  mixed_plan:      每 (rho, objective, service) 逐架次机型/箱号/能耗/时间方案
"""

PROJECT = Path(__file__).resolve().parent
G = 9.81


def load_cargo():
    return pd.read_csv(PROJECT / "data" / "物资需求.csv")


# ═══════════════════════════════════════════════════════════════
# 步 1: 最大安全载荷
# ═══════════════════════════════════════════════════════════════

def build_max_payload_table(models, verbose=False):
    u"""给定 models dict, 计算 3 x 15 最大安全载荷 (纯函数, 不写文件)

    可被基准步1和敏感性分析复用. 返回 df_max.
    """
    results = []
    for g, model in models.items():
        u = model.u
        E_avail = model.available_energy
        if verbose:
            print(f"\n机型 {g}:")
            print(f"  M_g0={u['M_g0']}kg  Q_g={u['Q_g']}kg  V_g={u['V_g']}m^3")
            print(f"  L_0={int(u['L_0'])}m  L_F={int(u['L_F'])}m  "
                  f"E_use={u['E_use']}kWh  ρ={u['ρ_g']}%  "
                  f"E_avail={E_avail:.4f}kWh")
            print(f"  {'服务区':<8} {'距离(m)':>10} {'H_up(m)':>10} "
                  f"{'m_energy(kg)':>14} {'E_round':>10} {'绑定约束':>12}")
            print(f"  {'-'*62}")

        for si in model.routes["distance"].columns:
            if not si.startswith("S"):
                continue
            m_max = model.max_safe_payload(si)
            H_up = float(model.routes["climb_height"].loc["O01", si])
            d = float(model.routes["distance"].loc["O01", si])
            E_rt = model.round_trip_energy(m_max, si)

            if m_max >= u["Q_g"] - 1e-6:
                binding = "质量(Q_g)"
            else:
                binding = "能量"

            results.append({
                "type": g, "service": si,
                "distance": d, "H_up_out": H_up,
                "Q_g": u["Q_g"], "V_g": u["V_g"],
                "m_energy": round(m_max, 4),
                "E_round_kWh": round(E_rt, 6),
                "E_avail_kWh": round(E_avail, 6),
                "binding": binding,
            })

            if verbose:
                print(f"  {si:<8} {d:>10.0f} {H_up:>10.1f} "
                      f"{m_max:>14.2f} {E_rt:>10.4f} {binding:>12}")

    return pd.DataFrame(results)


def compute_max_payloads_all():
    u"""步1: 三种机型 × 15 服务区 — 能量约束下的最大安全质量载荷 (写CSV)"""
    models = load_models()

    print("=" * 70)
    print("步 1：最大安全载荷（等效航程模型）")
    print("=" * 70)

    df_res = build_max_payload_table(models, verbose=True)
    df_res.to_csv(PROJECT / "data" / "Q1_fixed_max_payload.csv",
                  index=False, encoding="utf-8-sig")
    print("\n已保存: data/Q1_fixed_max_payload.csv")
    print("  m_energy: 能量约束下的最大安全质量载荷")
    print("  体积约束 V_g 在组批阶段作为装箱约束检查")
    return df_res, models


def _build_models_with_rho(models, rho):
    u"""基于基准 models, 用指定 ρ 重新构造三类物理模型

    不修改传入的 models, 返回全新的 TransportPhysicsModel dict.
    """
    rho_models = {}
    for g, base_model in models.items():
        u_mod = base_model.u.copy()
        u_mod["ρ_g"] = float(rho)
        rho_models[g] = TransportPhysicsModel(u_mod, base_model.routes)
    return rho_models


# ═══════════════════════════════════════════════════════════════
# 步 2: 组批优化 — 共享工具
# ═══════════════════════════════════════════════════════════════

def generate_feasible_batches(box_masses, box_volumes, m_eff, V_g):
    u"""DFS 剪枝生成所有满足质量+体积约束的货箱子集

    将货箱按质量降序排列，递归尝试添加每个后续货箱。
    一旦累计质量/体积超过上限，剪去该分支。
    复杂度从 O(2^n) 降至实际可行组合数。
    """
    n = len(box_masses)
    items = sorted(
        [(box_masses[i], box_volumes[i], i) for i in range(n)],
        key=lambda x: x[0], reverse=True,
    )
    batches = []

    def dfs(start, cur_indices, cur_mass, cur_vol):
        if cur_indices:
            batches.append({
                "indices": sorted(cur_indices),
                "mass": cur_mass,
                "volume": cur_vol,
                "n_boxes": len(cur_indices),
            })
        for idx in range(start, n):
            mass_i, vol_i, orig_i = items[idx]
            new_m = cur_mass + mass_i
            new_v = cur_vol + vol_i
            if new_m > m_eff + 1e-9:
                continue
            if new_v > V_g + 1e-9:
                continue
            dfs(idx + 1, cur_indices + [orig_i], new_m, new_v)

    dfs(0, [], 0.0, 0.0)
    return batches


def _collect_service_batches(model, df_max, cargo_df, service, g):
    u"""为指定 (g, service) 生成可行组合并计算能耗+时间

    返回: (feasible, n_boxes, energies, times)
    """
    boxes = cargo_df[cargo_df["service"] == service]
    box_masses = []
    box_volumes = []
    for _, row in boxes.iterrows():
        cnt = int(row["total_boxes"])
        box_masses.extend([row["mass_per_box"]] * cnt)
        box_volumes.extend([row["volume_per_box"]] * cnt)

    n_boxes = len(box_masses)
    V_g = float(model.u["V_g"])
    Q_g = float(model.u["Q_g"])

    m_energy = float(
        df_max[(df_max["type"] == g)
               & (df_max["service"] == service)]["m_energy"].values[0]
    )
    m_eff = min(m_energy, Q_g)

    feasible = generate_feasible_batches(box_masses, box_volumes, m_eff, V_g)

    energies = {}
    times = {}
    for b_idx, batch in enumerate(feasible):
        energies[b_idx] = model.round_trip_energy(batch["mass"], service)
        times[b_idx] = model.sortie_total_time(batch["n_boxes"], service)

    return feasible, n_boxes, energies, times


# ═══════════════════════════════════════════════════════════════
# 步 2: 三个单目标最优解
# ═══════════════════════════════════════════════════════════════

def _build_milp(batches, box_count):
    u"""构建覆盖约束 MILP 的公共组件"""
    n = len(batches)
    A = np.zeros((box_count, n))
    for b_idx, batch in enumerate(batches):
        for j in batch["indices"]:
            A[j, b_idx] = 1.0
    con = LinearConstraint(A, np.ones(box_count), np.ones(box_count))
    bounds = Bounds(np.zeros(n), np.ones(n))
    integrality = np.ones(n, dtype=int)
    return n, A, con, bounds, integrality


def _run_milp(c_vec, con, bounds, integrality):
    u"""求解单次 MILP, 返回 selected 索引列表或 None"""
    res = milp(c=c_vec, constraints=con, bounds=bounds,
               integrality=integrality, options={"disp": False})
    if not res.success:
        return None
    return [b for b, x in enumerate(res.x) if x > 0.5]


def solve_min_N(batches, box_count, energies, times):
    u"""单目标: min N_f (最少架次数)"""
    if len(batches) == 0:
        return None
    if len(batches) == 1:
        return [0]
    _, _, con, bounds, integrality = _build_milp(batches, box_count)
    return _run_milp(np.ones(len(batches)), con, bounds, integrality)


def solve_min_E(batches, box_count, energies, times):
    u"""单目标: min ΣE (最低总能耗)"""
    if len(batches) == 0:
        return None
    if len(batches) == 1:
        return [0]
    _, _, con, bounds, integrality = _build_milp(batches, box_count)
    c = np.array([energies[b] for b in range(len(batches))])
    return _run_milp(c, con, bounds, integrality)


def solve_min_T(batches, box_count, energies, times):
    u"""单目标: min ΣT (最短累计时间)"""
    if len(batches) == 0:
        return None
    if len(batches) == 1:
        return [0]
    _, _, con, bounds, integrality = _build_milp(batches, box_count)
    c = np.array([times[b] for b in range(len(batches))])
    return _run_milp(c, con, bounds, integrality)


def run_single_objective(models):
    u"""步 2: 三个单目标最优解

    对每个 (g, service) 分别求解:
      N-opt: min N_f
      E-opt: min ΣE
      T-opt: min ΣT

    输出: Q1_fixed_single_objective.csv (每个 service 的三组极值)
    """
    print("\n" + "=" * 70)
    print("步 2：三个单目标最优解 (N-opt, E-opt, T-opt)")
    print("=" * 70)

    df_max = pd.read_csv(PROJECT / "data" / "Q1_fixed_max_payload.csv")
    cargo_df = load_cargo()
    service_areas = sorted(cargo_df["service"].unique())

    solvers = {
        "N-opt": (solve_min_N, "min N_f"),
        "E-opt": (solve_min_E, "min ΣE"),
        "T-opt": (solve_min_T, "min ΣT"),
    }

    all_records = []

    for g, model in models.items():
        u = model.u
        print(f"\n机型 {g}:  Q_g={u['Q_g']}kg  V_g={u['V_g']}m^3")
        header = (f"  {'服务区':<8} {'箱数':>4} "
                  f"{'N-opt':>5} {'E(N)':>10} {'T(N)':>10}"
                  f"{'E-opt':>5} {'E(E)':>10} {'T(E)':>10}"
                  f"{'T-opt':>5} {'E(T)':>10} {'T(T)':>10}")
        print(header)
        print(f"  {'-'*90}")

        for service in service_areas:
            feasible, n_boxes, energies, times = _collect_service_batches(
                model, df_max, cargo_df, service, g)

            if len(feasible) == 0:
                print(f"  {service:<8} {n_boxes:>4} {'无可行组合':>10}")
                continue

            results = {}
            for strategy, (solver_fn, _) in solvers.items():
                sel = solver_fn(feasible, n_boxes, energies, times)
                if sel is None:
                    results[strategy] = None
                else:
                    results[strategy] = {
                        "N": len(sel),
                        "E": sum(energies[b] for b in sel),
                        "T": sum(times[b] for b in sel),
                        "selected": sel,
                    }

            n_opt   = results.get("N-opt")
            e_opt   = results.get("E-opt")
            t_opt   = results.get("T-opt")

            if None in (n_opt, e_opt, t_opt):
                print(f"  {service:<8} {n_boxes:>4} {'MILP失败':>10}")
                continue

            print(f"  {service:<8} {n_boxes:>4} "
                  f"{n_opt['N']:>5} {n_opt['E']:>10.2f} {n_opt['T']:>10.0f}"
                  f"{e_opt['N']:>5} {e_opt['E']:>10.2f} {e_opt['T']:>10.0f}"
                  f"{t_opt['N']:>5} {t_opt['E']:>10.2f} {t_opt['T']:>10.0f}")

            for strategy, res in results.items():
                all_records.append({
                    "type": g, "service": service,
                    "strategy": strategy,
                    "N_f": res["N"],
                    "E_total_kWh": res["E"],
                    "T_total_s": res["T"],
                })

    df_so = pd.DataFrame(all_records)
    df_so.to_csv(PROJECT / "data" / "Q1_fixed_single_objective.csv",
                 index=False, encoding="utf-8-sig")

    print("\n已保存: data/Q1_fixed_single_objective.csv")
    return df_so


# ═══════════════════════════════════════════════════════════════
# 步 5: ρ 敏感性分析
# ═══════════════════════════════════════════════════════════════

def sensitivity_analysis(models):
    u"""返航安全余量 ρ 敏感性分析 ── 载荷边界 + 混合机型组批双重分析

    对每个 ρ ∈ {10,15,20,25,30}:
      1. 用 rho_models 重新计算 3×15 最大安全载荷
      2. 用 rho_df_max 重新生成候选批次 → 精确 DP 求 N/E/T-opt
      3. 输出 4 个文件:
         - Q1_sensitivity_payload_summary.csv  每 (rho, type) 统计
         - Q1_sensitivity_payload_detail.csv   每 (rho, type, service) 明细
         - Q1_sensitivity_mixed_summary.csv    每 (rho, service, obj) N/E/T
         - Q1_sensitivity_mixed_plan.csv       每 (rho, obj, service) 逐架次方案
    """
    cargo_df = load_cargo()
    rho_values = [10, 15, 20, 25, 30]
    services = sorted(cargo_df["service"].unique())

    payload_summary = []
    payload_detail = []
    mixed_summary = []
    mixed_plans = []

    print("\n" + "=" * 70)
    print("步 5：ρ 敏感性分析 — 载荷边界 + 混合机型组批")
    print("=" * 70)

    for rho in rho_values:
        rho_models = _build_models_with_rho(models, rho)
        df_max_rho = build_max_payload_table(rho_models)

        print(f"\n── ρ = {rho}% ──")

        for _, row in df_max_rho.iterrows():
            payload_detail.append({
                "rho": rho,
                "type": row["type"],
                "service": row["service"],
                "distance": row["distance"],
                "H_up_out": row["H_up_out"],
                "Q_g": row["Q_g"],
                "V_g": row["V_g"],
                "m_energy": row["m_energy"],
                "E_round_kWh": row["E_round_kWh"],
                "E_avail_kWh": row["E_avail_kWh"],
                "binding": row["binding"],
            })

        for g, base_model in models.items():
            Q_g = float(base_model.u["Q_g"])
            subset = df_max_rho[df_max_rho["type"] == g]
            m_vals = subset["m_energy"].values
            mass_bound = int(np.sum(m_vals >= Q_g - 1e-6))
            energy_bound = len(m_vals) - mass_bound
            payload_summary.append({
                "rho": rho, "type": g,
                "mass_limited_areas": mass_bound,
                "energy_limited_areas": energy_bound,
                "m_max_mean": round(float(np.mean(m_vals)), 4),
                "m_max_min": round(float(np.min(m_vals)), 4),
                "m_max_max": round(float(np.max(m_vals)), 4),
            })

        for service in services:
            boxes, candidates = _mixed_candidates(
                rho_models, df_max_rho, cargo_df, service
            )
            for target in ("N", "E", "T"):
                solution = solve_mixed_partition(
                    candidates, len(boxes), target
                )
                records = _mixed_plan(
                    solution, boxes, candidates,
                    rho_models, df_max_rho, service
                )
                for rec in records:
                    rec["rho"] = rho
                    rec["objective"] = f"{target}-opt"
                mixed_plans.extend(records)

                counts = Counter(r["type"] for r in records)
                mixed_summary.append({
                    "rho": rho,
                    "service": service,
                    "objective": f"{target}-opt",
                    "total_boxes": len(boxes),
                    "N_f": solution[1],
                    "E_total_kWh": solution[0],
                    "T_total_s": solution[2],
                    "A_sorties": counts.get("A", 0),
                    "B_sorties": counts.get("B", 0),
                    "C_sorties": counts.get("C", 0),
                })

        msg = (
            f"  N-opt: {sum(r['N_f'] for r in mixed_summary if r['rho']==rho and r['objective']=='N-opt')}, "
            f"E-opt: {sum(r['E_total_kWh'] for r in mixed_summary if r['rho']==rho and r['objective']=='E-opt'):.3f}kWh, "
            f"T-opt: {sum(r['T_total_s'] for r in mixed_summary if r['rho']==rho and r['objective']=='T-opt'):.0f}s"
        )
        print(msg)

    pd.DataFrame(payload_summary).to_csv(
        PROJECT / "data" / "Q1_sensitivity_payload_summary.csv",
        index=False, encoding="utf-8-sig")
    pd.DataFrame(payload_detail).to_csv(
        PROJECT / "data" / "Q1_sensitivity_payload_detail.csv",
        index=False, encoding="utf-8-sig")
    pd.DataFrame(mixed_summary).to_csv(
        PROJECT / "data" / "Q1_sensitivity_mixed_summary.csv",
        index=False, encoding="utf-8-sig")
    pd.DataFrame(mixed_plans).to_csv(
        PROJECT / "data" / "Q1_sensitivity_mixed_plan.csv",
        index=False, encoding="utf-8-sig")

    print("\n已保存:")
    print("  data/Q1_sensitivity_payload_summary.csv  — ρ × type 载荷统计")
    print("  data/Q1_sensitivity_payload_detail.csv   — ρ × type × service 载荷明细")
    print("  data/Q1_sensitivity_mixed_summary.csv    — ρ × obj 混合组批 N/E/T")
    print("  data/Q1_sensitivity_mixed_plan.csv       — ρ × obj × service 逐架次方案")

    return payload_summary, mixed_summary, mixed_plans


# ═══════════════════════════════════════════════════════════════
# 混合机型组批: 每架次自由选择 A/B/C，精确单目标求解
# ═══════════════════════════════════════════════════════════════

def _mixed_boxes(cargo_df, service):
    boxes = []
    for _, row in cargo_df[cargo_df["service"] == service].iterrows():
        for _ in range(int(row["total_boxes"])):
            boxes.append({
                "id": f"{service}-B{len(boxes) + 1:03d}",
                "cargo_type": row["cargo_type"],
            })
    if not boxes or len(boxes) > 20:
        raise ValueError(f"{service}: 精确混合组批要求 1 至 20 箱")
    return boxes


def _mixed_candidates(models, df_max, cargo_df, service):
    """复用固定机型的候选生成与能耗/时间公式。"""
    boxes = _mixed_boxes(cargo_df, service)
    candidates = {g: {} for g in models}
    for g, model in models.items():
        batches, n_boxes, energies, times = _collect_service_batches(
            model, df_max, cargo_df, service, g
        )
        if n_boxes != len(boxes):
            raise AssertionError(f"{service}/{g}: 货箱展开顺序不一致")
        for b_idx, batch in enumerate(batches):
            energy = energies[b_idx]
            if energy > model.available_energy + 1e-9:
                continue
            mask = sum(1 << i for i in batch["indices"])
            candidates[g][mask] = {
                "E": energy, "T": times[b_idx],
                "mass": batch["mass"], "volume": batch["volume"],
                "n_boxes": batch["n_boxes"],
            }
    return boxes, candidates


def solve_mixed_partition(candidates, box_count, objective):
    """精确集合划分 DP；目标分别为 N→E→T、E→N→T、T→E→N。"""
    if objective not in ("N", "E", "T"):
        raise ValueError(f"未知目标: {objective}")
    full = (1 << box_count) - 1
    best_batch = [None] * (full + 1)
    for g in sorted(candidates):
        for mask, metrics in candidates[g].items():
            option = (g, metrics)
            incumbent = best_batch[mask]
            if objective == "T":
                key = (metrics["T"], metrics["E"], g)
                old_key = (
                    incumbent[1]["T"], incumbent[1]["E"], incumbent[0]
                ) if incumbent else None
            else:
                key = (metrics["E"], metrics["T"], g)
                old_key = (
                    incumbent[1]["E"], incumbent[1]["T"], incumbent[0]
                ) if incumbent else None
            if old_key is None or key < old_key:
                best_batch[mask] = option

    def score(result):
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
            option = best_batch[subset]
            if subset & first and option is not None:
                g, metrics = option
                tail = solve(remaining ^ subset)
                candidate = (
                    metrics["E"] + tail[0],
                    1 + tail[1],
                    metrics["T"] + tail[2],
                    ((subset, g),) + tail[3],
                )
                if best is None or score(candidate) < score(best):
                    best = candidate
            subset = (subset - 1) & remaining
        if best is None:
            raise ValueError("混合机型组批无可行解")
        return best

    return solve(full)


def _mixed_plan(solution, boxes, candidates, models, df_max, service):
    covered = 0
    rows = []
    for sortie_no, (mask, g) in enumerate(solution[3], start=1):
        if covered & mask:
            raise AssertionError(f"{service}: 货箱重复配送")
        covered |= mask
        batch = candidates[g][mask]
        model = models[g]
        safe_mass = float(df_max.loc[
            (df_max["type"] == g) & (df_max["service"] == service),
            "m_energy",
        ].iloc[0])
        if (batch["mass"] > min(safe_mass, model.u["Q_g"]) + 1e-9
                or batch["volume"] > model.u["V_g"] + 1e-9
                or batch["E"] > model.available_energy + 1e-9):
            raise AssertionError(f"{service}/{g}: 批次违反安全约束")
        selected = [box for i, box in enumerate(boxes) if mask & (1 << i)]
        breakdown = Counter(box["cargo_type"] for box in selected)
        rows.append({
            "service": service,
            "sortie_id": f"{service}-{sortie_no:02d}",
            "type": g,
            "box_ids": "|".join(box["id"] for box in selected),
            "cargo_counts": "|".join(
                f"{name}:{count}" for name, count in sorted(breakdown.items())
            ),
            "n_boxes": batch["n_boxes"],
            "mass_kg": batch["mass"],
            "volume_m3": batch["volume"],
            "energy_kWh": batch["E"],
            "time_s": batch["T"],
            "safe_payload_kg": safe_mass,
            "available_energy_kWh": model.available_energy,
        })
    if covered != (1 << len(boxes)) - 1:
        raise AssertionError(f"{service}: 货箱未全部配送")
    if (not math.isclose(sum(r["energy_kWh"] for r in rows), solution[0], abs_tol=1e-8)
            or not math.isclose(sum(r["time_s"] for r in rows), solution[2], abs_tol=1e-8)):
        raise AssertionError(f"{service}: 方案指标与逐架次记录不一致")
    return rows


def run_mixed(models, df_max, df_fixed):
    """三个混合机型单目标实验；文件统一写为 Q1_mixed_*。"""
    print("\n" + "=" * 70)
    print("混合机型组批：N-opt / E-opt / T-opt")
    print("=" * 70)
    cargo_df = load_cargo()
    services = sorted(cargo_df["service"].unique())
    fixed_e = df_fixed[df_fixed["strategy"] == "E-opt"]
    fixed_lookup = fixed_e.set_index(["type", "service"])
    objective_rows = []
    comparison_rows = []
    plans = {target: [] for target in ("N", "E", "T")}
    totals = {target: {"N": 0, "E": 0.0, "T": 0.0, "boxes": 0,
                       "types": Counter()} for target in plans}
    fixed_totals = {g: {"N": 0, "E": 0.0, "T": 0.0} for g in models}

    for service in services:
        boxes, candidates = _mixed_candidates(models, df_max, cargo_df, service)
        solutions = {
            target: solve_mixed_partition(candidates, len(boxes), target)
            for target in plans
        }
        for g in models:
            row = fixed_lookup.loc[(g, service)]
            fixed_totals[g]["N"] += int(row["N_f"])
            fixed_totals[g]["E"] += float(row["E_total_kWh"])
            fixed_totals[g]["T"] += float(row["T_total_s"])
            comparison_rows.append({
                "service": service, "strategy": f"fixed-{g}-E-opt",
                "N_f": row["N_f"], "E_total_kWh": row["E_total_kWh"],
                "T_total_s": row["T_total_s"],
            })
        for target, solution in solutions.items():
            records = _mixed_plan(
                solution, boxes, candidates, models, df_max, service
            )
            plans[target].extend(records)
            counts = Counter(r["type"] for r in records)
            total = totals[target]
            total["N"] += solution[1]
            total["E"] += solution[0]
            total["T"] += solution[2]
            total["boxes"] += len(boxes)
            total["types"].update(counts)
            objective_rows.append({
                "service": service, "objective": f"{target}-opt",
                "total_boxes": len(boxes), "N_f": solution[1],
                "E_total_kWh": solution[0], "T_total_s": solution[2],
                "A_sorties": counts["A"], "B_sorties": counts["B"],
                "C_sorties": counts["C"],
            })
            comparison_rows.append({
                "service": service, "strategy": f"mixed-{target}-opt",
                "N_f": solution[1], "E_total_kWh": solution[0],
                "T_total_s": solution[2],
            })
        if solutions["E"][0] > min(
            float(fixed_lookup.loc[(g, service), "E_total_kWh"]) for g in models
        ) + 1e-7:
            raise AssertionError(f"{service}: 混合能耗结果高于固定机型最优")
        print(
            f"{service}: N*={solutions['N'][1]}, "
            f"E*={solutions['E'][0]:.6f} kWh, "
            f"T*={solutions['T'][2]:.3f} s"
        )

    for target, total in totals.items():
        counts = total["types"]
        objective_rows.append({
            "service": "all", "objective": f"{target}-opt",
            "total_boxes": total["boxes"], "N_f": total["N"],
            "E_total_kWh": total["E"], "T_total_s": total["T"],
            "A_sorties": counts["A"], "B_sorties": counts["B"],
            "C_sorties": counts["C"],
        })
        comparison_rows.append({
            "service": "all", "strategy": f"mixed-{target}-opt",
            "N_f": total["N"], "E_total_kWh": total["E"],
            "T_total_s": total["T"],
        })
    for g, total in fixed_totals.items():
        comparison_rows.append({
            "service": "all", "strategy": f"fixed-{g}-E-opt",
            "N_f": total["N"], "E_total_kWh": total["E"],
            "T_total_s": total["T"],
        })
    if totals["N"]["N"] > totals["E"]["N"] or totals["N"]["N"] > totals["T"]["N"]:
        raise AssertionError("混合架次极值不一致")
    if totals["E"]["E"] > totals["N"]["E"] + 1e-7 or totals["E"]["E"] > totals["T"]["E"] + 1e-7:
        raise AssertionError("混合能耗极值不一致")
    if totals["T"]["T"] > totals["N"]["T"] + 1e-7 or totals["T"]["T"] > totals["E"]["T"] + 1e-7:
        raise AssertionError("混合时间极值不一致")

    for target, rows in plans.items():
        pd.DataFrame(rows).to_csv(
            PROJECT / "data" / f"Q1_mixed_{target}_opt_plan.csv",
            index=False, encoding="utf-8-sig",
        )
    pd.DataFrame(objective_rows).to_csv(
        PROJECT / "data" / "Q1_mixed_objectives.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(comparison_rows).to_csv(
        PROJECT / "data" / "Q1_mixed_comparison.csv",
        index=False, encoding="utf-8-sig",
    )
    input_names = (
        "物资需求.csv", "运输无人机_机型参数.csv",
        "Q1_fixed_max_payload.csv", "Q1_fixed_single_objective.csv",
        "distance_matrix.csv", "climb_height_matrix.csv", "descent_height_matrix.csv",
    )
    manifest = {
        "run_command": "python code/Q1.py",
        "solver": "exact subset dynamic programming; no random seed",
        "objectives": {
            "N-opt": "min N, then E, then cumulative T",
            "E-opt": "min E, then N, then cumulative T",
            "T-opt": "min cumulative T, then E, then N",
        },
        "scope": (
            "O01-service-O01; indivisible boxes; A/B/C per sortie; "
            "safe payload, volume and available energy enforced; "
            "fleet scheduling, shared batteries and deadlines excluded"
        ),
        "python_version": sys.version.split()[0],
        "input_sha256": {
            f"data/{name}": hashlib.sha256(
                (PROJECT / "data" / name).read_bytes()
            ).hexdigest() for name in input_names
        },
    }
    with (PROJECT / "data" / "Q1_mixed_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for target, total in totals.items():
        print(
            f"混合 {target}-opt: N={total['N']}, E={total['E']:.9f} kWh, "
            f"T={total['T']:.3f} s"
        )
    return pd.DataFrame(objective_rows)


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def run_mixed_only():
    """兼容旧混合实验入口；先生成安全载荷与固定 E-opt 对照。"""
    df_max, models = compute_max_payloads_all()
    df_so = run_single_objective(models)
    return run_mixed(models, df_max, df_so)


def main():
    t0 = time()

    # 1. 最大安全载荷
    df_max, models = compute_max_payloads_all()

    # 2. 固定机型 A/B/C 的 N/E/T 极值
    df_so = run_single_objective(models)

    # 3. A/B/C 混合机型 N/E/T 极值
    run_mixed(models, df_max, df_so)

    # 4. 安全余量敏感性分析
    sensitivity_analysis(models)

    print(f"\n总用时: {time() - t0:.1f}s")


if __name__ == "__main__":
    main()