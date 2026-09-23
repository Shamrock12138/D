import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
from time import time

from src.physics import TransportPhysicsModel, load_models
from src.optimization import solve_pareto_frontier

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

步 1 ─ 最大安全载荷
--------------------
  m_max = max q  s.t.  E_round(q) ≤ (1-ρ_g)*E_use  ∧  q ≤ Q_g
  (体积约束 V_g 在组批阶段检查)

步 2 ─ 组批优化 (两方案对比)
----------------------------
  Baseline (资源节约策略 MSS):
    三阶段字典序 MILP:  N_f ≻ ΣE ≻ ΣT

  Proposed (多目标协同 MOS):
    Pareto 多目标 MILP (ε-约束 + 加权法)
    目标:  min (N_f, ΣE, ΣT) → 非支配集

步 3 ─ 结果对比
---------------
  两方案在架次数、能耗、时间三个维度的对比分析

步 4 ─ ρ 敏感性分析
--------------------
  返航安全余量 ρ 对最大安全载荷的影响
"""

PROJECT = Path(__file__).resolve().parent
G = 9.81


def load_cargo():
    return pd.read_csv(PROJECT / "data" / "物资需求.csv")


# ═══════════════════════════════════════════════════════════════
# 步 1: 最大安全载荷
# ═══════════════════════════════════════════════════════════════

def compute_max_payloads_all():
    u"""三种机型 × 15 服务区 — 能量约束下的最大安全质量载荷"""
    models = load_models()

    print("=" * 70)
    print("步 1：最大安全载荷（等效航程模型）")
    print("=" * 70)

    results = []
    for g, model in models.items():
        u = model.u
        E_avail = model.available_energy
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
# 步 2(a): Baseline — 资源节约策略 (MSS)
# ═══════════════════════════════════════════════════════════════

def solve_baseline_partition(batches, box_count, energies, times):
    u"""三阶段字典序 MILP:  N_f ≻ ΣE ≻ ΣT

    Stage 1: min Σ x_b        → N*
    Stage 2: min Σ E_b x_b   s.t. Σ x_b = N*   → E*
    Stage 3: min Σ T_b x_b   s.t. Σ x_b = N*, Σ E_b x_b = E*

    短路: N*=1 时直接枚举最优全覆盖 batch.
    """
    n_batches = len(batches)
    if n_batches == 0:
        return None
    if n_batches == 1:
        return [0]

    A = np.zeros((box_count, n_batches))
    for b_idx, batch in enumerate(batches):
        for j in batch["indices"]:
            A[j, b_idx] = 1.0

    eq_constraint = LinearConstraint(A, np.ones(box_count), np.ones(box_count))
    bounds = Bounds(np.zeros(n_batches), np.ones(n_batches))
    integrality = np.ones(n_batches, dtype=int)
    opts = {"disp": False}

    c_N = np.ones(n_batches)
    res1 = milp(c=c_N, constraints=eq_constraint, bounds=bounds,
                integrality=integrality, options=opts)
    if not res1.success:
        return None
    N_star = int(round(res1.fun))

    if N_star == 1:
        full = [b for b in range(n_batches)
                if batches[b]["n_boxes"] == box_count]
        if full:
            return [min(full, key=lambda b: (energies[b], times[b]))]

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

    c_T = np.array([times[b] for b in range(n_batches)])
    A3 = np.vstack([A, A_N, c_E.reshape(1, -1)])
    lb3 = np.concatenate([np.ones(box_count), [N_star - 1e-9], [E_star - 1e-6]])
    ub3 = np.concatenate([np.ones(box_count), [N_star + 1e-9], [E_star + 1e-6]])
    con3 = LinearConstraint(A3, lb3, ub3)
    res3 = milp(c=c_T, constraints=con3, bounds=bounds,
                integrality=integrality, options=opts)
    if not res3.success:
        return [b for b, x in enumerate(res2.x) if x > 0.5]

    return [b for b, x_val in enumerate(res3.x) if x_val > 0.5]


def run_baseline(models):
    u"""步 2(a): Baseline — 资源节约型组批策略 (最小架次优先)"""
    print("\n" + "=" * 70)
    print("步 2(a)：Baseline — 资源节约策略 MSS (N_f ≻ E ≻ T)")
    print("=" * 70)

    df_max = pd.read_csv(PROJECT / "data" / "Q1_max_payload.csv")
    cargo_df = load_cargo()
    service_areas = sorted(cargo_df["service"].unique())

    all_records = []

    for g, model in models.items():
        u = model.u
        print(f"\n机型 {g}:  Q_g={u['Q_g']}kg  V_g={u['V_g']}m^3")
        print(f"  {'服务区':<8} {'箱数':>4} {'可行组合':>8} "
              f"{'架次':>4} {'能耗(kWh)':>10} {'时间(s)':>10}")
        print(f"  {'-'*54}")

        for service in service_areas:
            feasible, n_boxes, energies, times = _collect_service_batches(
                model, df_max, cargo_df, service, g)

            if len(feasible) == 0:
                print(f"  {service:<8} {n_boxes:>4} {'无解':>8}")
                continue

            selected = solve_baseline_partition(feasible, n_boxes, energies, times)
            if selected is None:
                print(f"  {service:<8} {n_boxes:>4} {'MILP失败':>8}")
                continue

            total_E = sum(energies[b] for b in selected)
            total_T = sum(times[b] for b in selected)

            print(f"  {service:<8} {n_boxes:>4} {len(feasible):>8} "
                  f"{len(selected):>4} {total_E:>10.4f} {total_T:>10.0f}")

            for b_idx in selected:
                batch = feasible[b_idx]
                all_records.append({
                    "type": g, "service": service,
                    "n_boxes": batch["n_boxes"],
                    "mass": batch["mass"], "volume": batch["volume"],
                    "energy_kWh": energies[b_idx], "time_s": times[b_idx],
                })

    df_batches = pd.DataFrame(all_records)
    df_batches.to_csv(PROJECT / "data" / "Q1_baseline_plan.csv",
                      index=False, encoding="utf-8-sig")

    summary = df_batches.groupby(["type", "service"]).agg(
        sorties=("n_boxes", "count"),
        total_boxes=("n_boxes", "sum"),
        total_mass=("mass", "sum"),
        total_volume=("volume", "sum"),
        total_energy=("energy_kWh", "sum"),
        total_time=("time_s", "sum"),
    ).reset_index()
    summary.to_csv(PROJECT / "data" / "Q1_baseline_summary.csv",
                   index=False, encoding="utf-8-sig")

    print("\n已保存: data/Q1_baseline_plan.csv, data/Q1_baseline_summary.csv")

    overall = summary.groupby("type").agg(
        sorties=("sorties", "sum"),
        boxes=("total_boxes", "sum"),
        energy_kWh=("total_energy", "sum"),
        time_s=("total_time", "sum"),
    )

    print("\n  Baseline 各机型总汇总")
    print("  " + "-" * 40)
    print(overall.to_string())

    return df_batches


# ═══════════════════════════════════════════════════════════════
# 步 2(b): Proposed — Pareto 多目标协同优化 (MOS)
# ═══════════════════════════════════════════════════════════════

def run_pareto(models):
    u"""步 2(b): Proposed — Pareto 多目标协同优化策略

    对每个 (g, service) 生成 Pareto 前沿 (N_f, ΣE, ΣT) 的非支配解集,
    保存所有前沿点到 Q1_pareto_frontier.csv.
    """
    print("\n" + "=" * 70)
    print("步 2(b)：Proposed — Pareto 多目标优化 MOS")
    print("=" * 70)

    df_max = pd.read_csv(PROJECT / "data" / "Q1_max_payload.csv")
    cargo_df = load_cargo()
    service_areas = sorted(cargo_df["service"].unique())

    all_frontier = []

    for g, model in models.items():
        u = model.u
        print(f"\n机型 {g}:  Q_g={u['Q_g']}kg  V_g={u['V_g']}m^3")
        print(f"  {'服务区':<8} {'箱数':>4} {'非支配解':>10} "
              f"{'N范围':>8} {'E范围(kWh)':>14} {'T范围(s)':>14}")
        print(f"  {'-'*64}")

        for service in service_areas:
            feasible, n_boxes, energies, times = _collect_service_batches(
                model, df_max, cargo_df, service, g)

            if len(feasible) == 0:
                print(f"  {service:<8} {n_boxes:>4} {'无解':>10}")
                continue

            frontier = solve_pareto_frontier(
                feasible, n_boxes, energies, times,
                n_extra_E=3, n_extra_T=3)

            if not frontier:
                print(f"  {service:<8} {n_boxes:>4} {'未找到':>10}")
                continue

            N_vals = [p["N"] for p in frontier]
            E_vals = [p["E"] for p in frontier]
            T_vals = [p["T"] for p in frontier]

            print(f"  {service:<8} {n_boxes:>4} {len(frontier):>10} "
                  f"[{min(N_vals)},{max(N_vals)}]  "
                  f"[{min(E_vals):.2f},{max(E_vals):.2f}]  "
                  f"[{min(T_vals):.0f},{max(T_vals):.0f}]")

            for p in frontier:
                for b_idx in p["selected"]:
                    batch = feasible[b_idx]
                    all_frontier.append({
                        "type": g, "service": service,
                        "N_f": p["N"],
                        "E_total": p["E"],
                        "T_total": p["T"],
                        "batch_idx": b_idx,
                        "n_boxes": batch["n_boxes"],
                        "mass": batch["mass"],
                        "volume": batch["volume"],
                        "energy_kWh": energies[b_idx],
                        "time_s": times[b_idx],
                    })

    df_fr = pd.DataFrame(all_frontier)
    df_fr.to_csv(PROJECT / "data" / "Q1_pareto_frontier.csv",
                 index=False, encoding="utf-8-sig")
    print("\n已保存: data/Q1_pareto_frontier.csv")
    return df_fr


# ═══════════════════════════════════════════════════════════════
# 步 3: 两方案对比
# ═══════════════════════════════════════════════════════════════

def _agg_service(df):
    u"""按 (type, service) 汇总统, E, T"""
    return df.groupby(["type", "service"]).agg(
        N_f=("N_f", "first") if "N_f" in df.columns else ("n_boxes", "count"),
        sorties=("n_boxes", "count") if "n_boxes" in df.columns else ("N_f", "first"),
        E_total=("energy_kWh", "sum"),
        T_total=("time_s", "sum"),
    ).reset_index()


def compare_results(df_baseline_plan, df_pareto_fr):
    u"""步 3: Baseline vs Pareto 对比分析

    对每个 (g, service):
      - Baseline: 单一解
      - Pareto: 展示前沿上三个代表性解 (min-N, min-E, min-T)
    """
    print("\n" + "=" * 70)
    print("步 3：Baseline (MSS) vs Proposed (MOS) 对比")
    print("=" * 70)

    # ── 汇总 Baseline ──
    bl_agg = df_baseline_plan.groupby(["type", "service"]).agg(
        N_bl=("n_boxes", "count"),
        E_bl=("energy_kWh", "sum"),
        T_bl=("time_s", "sum"),
    ).reset_index()

    # ── 汇总 Pareto: 按 solution 提取三个代表点 ──
    pq = df_pareto_fr.groupby(["type", "service", "N_f", "E_total", "T_total"]).agg(
        total_E=("energy_kWh", "sum"),
        total_T=("time_s", "sum"),
    ).reset_index()

    comparison_rows = []

    for g in ["A", "B", "C"]:
        print(f"\n{'='*60}")
        print(f"机型 {g}")
        print(f"{'='*60}")
        print(f"  {'服务区':<8} {'方案':>12} {'N_f':>5} "
              f"{'E(kWh)':>10} {'T(s)':>10} {'ΔN':>5} {'ΔE%':>8} {'ΔT%':>8}")
        print(f"  {'-'*58}")

        for service in sorted(bl_agg["service"].unique()):
            bl_row = bl_agg[(bl_agg["type"] == g)
                            & (bl_agg["service"] == service)]
            if bl_row.empty:
                continue
            bl = bl_row.iloc[0]
            N_bl, E_bl, T_bl = bl["N_bl"], bl["E_bl"], bl["T_bl"]

            # Baseline
            print(f"  {service:<8} {'Baseline':>12} {N_bl:>5} "
                  f"{E_bl:>10.2f} {T_bl:>10.0f}")

            # Pareto 三个代表点
            pq_svc = pq[(pq["type"] == g) & (pq["service"] == service)]
            if pq_svc.empty:
                continue

            idx_min_N = pq_svc["N_f"].idxmin()
            idx_min_E = pq_svc["total_E"].idxmin()
            idx_min_T = pq_svc["total_T"].idxmin()

            rep_points = [
                ("Pareto-N↓", idx_min_N),
                ("Pareto-E↓", idx_min_E),
                ("Pareto-T↓", idx_min_T),
            ]

            for label, idx in rep_points:
                row = pq_svc.loc[idx]
                dN = row["N_f"] - N_bl
                dE = (row["total_E"] - E_bl) / E_bl * 100 if E_bl else 0
                dT = (row["total_T"] - T_bl) / T_bl * 100 if T_bl else 0
                print(f"  {'':<8} {label:>12} {row['N_f']:>5.0f} "
                      f"{row['total_E']:>10.2f} {row['total_T']:>10.0f} "
                      f"{dN:>+5.0f} {dE:>+7.1f}% {dT:>+7.1f}%")

                comparison_rows.append({
                    "type": g, "service": service,
                    "strategy": label,
                    "N_f": row["N_f"],
                    "E_kWh": row["total_E"],
                    "T_s": row["total_T"],
                })

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(PROJECT / "data" / "Q1_comparison.csv",
                   index=False, encoding="utf-8-sig")
    print("\n已保存: data/Q1_comparison.csv")
    return df_comp


# ═══════════════════════════════════════════════════════════════
# 步 4: ρ 敏感性分析
# ═══════════════════════════════════════════════════════════════

def sensitivity_analysis(models):
    u"""返航安全余量 ρ 敏感性分析"""
    print("\n" + "=" * 70)
    print("步 4：ρ 敏感性分析（等效航程模型）")
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


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time()

    # 步 1
    df_max, models = compute_max_payloads_all()

    # 步 2(a): Baseline
    df_bl = run_baseline(models)

    # 步 2(b): Pareto
    df_pq = run_pareto(models)

    # 步 3: 对比
    compare_results(df_bl, df_pq)

    # 步 4: 敏感性
    sensitivity_analysis(models)

    print(f"\n总用时: {time() - t0:.1f}s")


if __name__ == "__main__":
    main()