import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))

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

最大安全载荷
------------
  m_max = max q  s.t.  E_round(q) ≤ (1-ρ_g)*E_use
                ∧      q ≤ Q_g
                ∧      (体积约束在组批阶段检查)

组批优化（集合划分 MILP）
-------------------------
  1. DFS 剪枝生成所有可行货箱组合（质量+体积）
  2. 三阶段字典序 MILP 求解集合划分:
        Stage 1: min N_f  (最少架次数)
        Stage 2: min ΣE   (最低能耗, 固定 N_f)
        Stage 3: min ΣT   (最短累计时间, 固定 N_f, E)
"""

PROJECT = Path(__file__).resolve().parent
G = 9.81


def load_cargo():
    return pd.read_csv(PROJECT / "data" / "物资需求.csv")


# ═══════════════════════════════════════════════════════════════
# 第一步：最大安全载荷
# ═══════════════════════════════════════════════════════════════

def compute_max_payloads_all():
    u"""三种机型 × 15 服务区 — 能量约束下的最大安全质量载荷"""
    models = load_models()

    print("=" * 70)
    print("第一步：最大安全载荷（等效航程模型）")
    print("=" * 70)

    results = []
    for g, model in models.items():
        u = model.u
        E_avail = model.available_energy
        print(f"\n机型 {g}:")
        print(f"  M_g0={u['M_g0']}kg  Q_g={u['Q_g']}kg  V_g={u['V_g']}m³")
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

            print(f"  {si:<8} {d:>10.0f} {H_up:>10.1f} "
                  f"{m_max:>14.2f} {E_rt:>10.4f} {binding:>12}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(PROJECT / "data" / "Q1_max_payload.csv",
                  index=False, encoding="utf-8-sig")
    print("\n已保存: data/Q1_max_payload.csv")
    print("  m_energy: 能量约束下的最大安全质量载荷")
    print("  体积约束 V_g 在组批阶段作为装箱约束检查")
    return df_res, models


# ═══════════════════════════════════════════════════════════════
# 第二步：货箱组批 — DFS 剪枝 + 三阶段字典序 MILP
# ═══════════════════════════════════════════════════════════════

def generate_feasible_batches(box_masses, box_volumes, m_eff, V_g):
    u"""DFS 剪枝生成所有满足质量+体积约束的货箱子集

    将货箱按质量降序排列，递归尝试添加每个后续货箱。
    一旦累计质量/体积超过上限，剪去该分支（后续更重的也不可能加入）。
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


def solve_set_partition_lex(batches, box_count, energies, times):
    u"""三阶段字典序 MILP 求解集合划分

    Stage 1: min Σ x_b        → N*
    Stage 2: min Σ E_b x_b   s.t. Σ x_b = N*   → E*
    Stage 3: min Σ T_b x_b   s.t. Σ x_b = N*, Σ E_b x_b = E*

    短路: N*=1 时直接枚举, 无需后两阶段 MILP.
    """
    n_batches = len(batches)
    if n_batches == 0:
        return None

    # ── 单一可行组合: 无需 MILP ──
    if n_batches == 1:
        return [0]

    # ── 构建约束矩阵 ──
    A = np.zeros((box_count, n_batches))
    for b_idx, batch in enumerate(batches):
        for j in batch["indices"]:
            A[j, b_idx] = 1.0

    eq_constraint = LinearConstraint(A, np.ones(box_count), np.ones(box_count))
    bounds = Bounds(np.zeros(n_batches), np.ones(n_batches))
    integrality = np.ones(n_batches, dtype=int)
    opts = {"disp": False}

    # ── Stage 1: 最少架次数 ──
    c_N = np.ones(n_batches)
    res1 = milp(c=c_N, constraints=eq_constraint, bounds=bounds,
                integrality=integrality, options=opts)
    if not res1.success:
        return None
    N_star = int(round(res1.fun))

    # ── N*=1 短路: 直接找全覆盖 batch ──
    if N_star == 1:
        full_batches = [b for b in range(n_batches)
                        if batches[b]["n_boxes"] == box_count]
        if full_batches:
            best = min(full_batches,
                       key=lambda b: (energies[b], times[b]))
            return [best]

    # ── Stage 2: 最低能耗（固定架次数）──
    c_E = np.array([energies[b] for b in range(n_batches)])
    A_N = np.ones((1, n_batches))
    A2 = np.vstack([A, A_N])
    lb2 = np.concatenate([np.ones(box_count), [N_star - 1e-9]])
    ub2 = np.concatenate([np.ones(box_count), [N_star + 1e-9]])
    con2 = LinearConstraint(A2, lb2, ub2)
    res2 = milp(c=c_E, constraints=con2, bounds=bounds,
                integrality=integrality, options=opts)
    if not res2.success:
        return [b for b, x in enumerate(res1.x) if x > 0.5]
    E_star = res2.fun

    # ── Stage 3: 最短累计时间（固定架次数+能耗）──
    c_T = np.array([times[b] for b in range(n_batches)])
    A3 = np.vstack([A, A_N, c_E.reshape(1, -1)])
    lb3 = np.concatenate([np.ones(box_count), [N_star - 1e-9], [E_star - 1e-6]])
    ub3 = np.concatenate([np.ones(box_count), [N_star + 1e-9], [E_star + 1e-6]])
    con3 = LinearConstraint(A3, lb3, ub3)
    res3 = milp(c=c_T, constraints=con3, bounds=bounds,
                integrality=integrality, options=opts)
    if not res3.success:
        return [b for b, x in enumerate(res2.x) if x > 0.5]

    return [b_idx for b_idx, x_val in enumerate(res3.x) if x_val > 0.5]


def run_batching(models):
    u"""第二步: 货箱组批优化"""
    print("\n" + "=" * 70)
    print("第二步：货箱组批优化（DFS剪枝 + 三阶段字典序MILP）")
    print("=" * 70)

    df_max = pd.read_csv(PROJECT / "data" / "Q1_max_payload.csv")
    cargo_df = load_cargo()
    service_areas = sorted(cargo_df["service"].unique())

    all_batches = []

    for g, model in models.items():
        u = model.u
        V_g = float(u["V_g"])
        Q_g = float(u["Q_g"])

        print(f"\n机型 {g}:")
        print(f"  Q_g={Q_g}kg  V_g={V_g}m³")
        print(f"  {'服务区':<8} {'箱数':>4} {'可行组合':>8} "
              f"{'架次':>4} {'总质量':>8} {'总体积':>8} "
              f"{'能耗(kWh)':>10} {'时间(s)':>10}")
        print(f"  {'-'*70}")

        for service in service_areas:
            boxes = cargo_df[cargo_df["service"] == service]
            box_masses = []
            box_volumes = []
            for _, row in boxes.iterrows():
                count = int(row["total_boxes"])
                box_masses.extend([row["mass_per_box"]] * count)
                box_volumes.extend([row["volume_per_box"]] * count)

            n_boxes = len(box_masses)

            m_energy = float(
                df_max[(df_max["type"] == g)
                       & (df_max["service"] == service)]["m_energy"].values[0]
            )
            m_eff = min(m_energy, Q_g)

            feasible = generate_feasible_batches(box_masses, box_volumes, m_eff, V_g)

            if len(feasible) == 0:
                print(f"  {service:<8} {n_boxes:>4} {'无解':>8}")
                continue

            energies = {}
            times = {}
            for b_idx, batch in enumerate(feasible):
                energies[b_idx] = model.round_trip_energy(batch["mass"], service)
                times[b_idx] = model.sortie_total_time(batch["n_boxes"], service)

            selected = solve_set_partition_lex(feasible, n_boxes, energies, times)

            if selected is None:
                print(f"  {service:<8} {n_boxes:>4} {'MILP失败':>8}")
                continue

            total_m = sum(feasible[b]["mass"] for b in selected)
            total_v = sum(feasible[b]["volume"] for b in selected)
            total_E = sum(energies[b] for b in selected)
            total_T = sum(times[b] for b in selected)

            print(f"  {service:<8} {n_boxes:>4} {len(feasible):>8} "
                  f"{len(selected):>4} {total_m:>8.1f} {total_v:>8.3f} "
                  f"{total_E:>10.4f} {total_T:>10.0f}")

            for b_idx in selected:
                batch = feasible[b_idx]
                all_batches.append({
                    "type": g, "service": service,
                    "n_boxes": batch["n_boxes"],
                    "mass": batch["mass"],
                    "volume": batch["volume"],
                    "energy_kWh": energies[b_idx],
                    "time_s": times[b_idx],
                })

    df_batches = pd.DataFrame(all_batches)
    df_batches.to_csv(PROJECT / "data" / "Q1_batch_plan.csv",
                      index=False, encoding="utf-8-sig")

    summary = df_batches.groupby(["type", "service"]).agg(
        sorties=("n_boxes", "count"),
        total_boxes=("n_boxes", "sum"),
        total_mass=("mass", "sum"),
        total_volume=("volume", "sum"),
        total_energy=("energy_kWh", "sum"),
        total_time=("time_s", "sum"),
    ).reset_index()
    summary.to_csv(PROJECT / "data" / "Q1_batch_summary.csv",
                   index=False, encoding="utf-8-sig")

    print("\n已保存: data/Q1_batch_plan.csv, data/Q1_batch_summary.csv")

    print("\n" + "=" * 70)
    print("各机型总汇总")
    print("=" * 70)
    overall = summary.groupby("type").agg(
        sorties=("sorties", "sum"),
        boxes=("total_boxes", "sum"),
        energy_kWh=("total_energy", "sum"),
        time_s=("total_time", "sum"),
    )
    print(overall.to_string())

    return df_batches


# ═══════════════════════════════════════════════════════════════
# 第三步：ρ 敏感性分析
# ═══════════════════════════════════════════════════════════════

def sensitivity_analysis(models):
    u"""返航安全余量 ρ 敏感性分析"""
    print("\n" + "=" * 70)
    print("第三步：ρ 敏感性分析（等效航程模型）")
    print("=" * 70)

    service_areas = sorted(models["A"].routes["distance"].columns)
    service_areas = [s for s in service_areas if s.startswith("S")]
    rho_values = [10, 15, 20, 25, 30]

    all_sens = []

    for g, base_model in models.items():
        Q_g = float(base_model.u["Q_g"])
        print(f"\n机型 {g}:")
        print(f"  {'ρ(%)':<8} {'质量受限区':>8} {'能量受限区':>8} "
              f"{'m_max均值':>12} {'m_max最小':>12}")

        for rho in rho_values:
            u_mod = base_model.u.copy()
            u_mod["ρ_g"] = float(rho)

            from src.physics import TransportPhysicsModel
            m = TransportPhysicsModel(u_mod, base_model.routes)

            m_vals = []
            energy_bound = 0
            mass_bound = 0
            for si in service_areas:
                m_e = m.max_safe_payload(si)
                m_vals.append(m_e)
                if m_e >= Q_g - 1e-6:
                    mass_bound += 1
                else:
                    energy_bound += 1

            print(f"  {rho:<8} {mass_bound:>10} {energy_bound:>10} "
                  f"{np.mean(m_vals):>12.1f} {np.min(m_vals):>12.1f}")

            all_sens.append({
                "type": g, "rho": rho,
                "mass_limited_areas": mass_bound,
                "energy_limited_areas": energy_bound,
                "m_max_mean": np.mean(m_vals),
                "m_max_min": np.min(m_vals),
                "m_max_max": np.max(m_vals),
            })

    df_sens = pd.DataFrame(all_sens)
    df_sens.to_csv(PROJECT / "data" / "Q1_sensitivity.csv",
                   index=False, encoding="utf-8-sig")
    print("\n已保存: data/Q1_sensitivity.csv")
    return df_sens


def main():
    t0 = time()
    df_max, models = compute_max_payloads_all()
    run_batching(models)
    sensitivity_analysis(models)
    print(f"\n总用时: {time() - t0:.1f}s")


if __name__ == "__main__":
    main()