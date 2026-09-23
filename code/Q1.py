import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
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
  水平能耗: E_hor = E_use * d / L_g(q)   [kWh]
  爬升能耗: E_up  = (M_g0+q)*g*H_up / (3.6e6*η_up)   [kWh]
  下降能耗: 0
  总能耗:   E_leg = E_hor + E_up

最大安全载荷
------------
  m_max = max q s.t. E_round(q) ≤ (1-ρ_g)*E_use  ∧  q ≤ Q_g

组批优化（集合划分 MILP）
-------------------------
  生成所有可行货箱组合 → MILP 求解集合划分
  目标（字典序）: 最少架次 ≻ 最低能耗 ≻ 最短时间
"""

PROJECT = Path(__file__).resolve().parent


def load_cargo():
    return pd.read_csv(PROJECT / "data" / "物资需求.csv")


def compute_max_payloads_all():
    u"""第一步：三种机型 × 15 服务区最大安全载荷"""
    models = load_models()

    print("=" * 70)
    print("第一步：最大安全载荷（基于等效航程模型）")
    print("=" * 70)

    results = []
    for g, model in models.items():
        u = model.u
        E_avail = model.available_energy
        print(f"\n机型 {g}:")
        print(f"  M_g0={u['M_g0']}kg  Q_g={u['Q_g']}kg  V_g={u['V_g']}m³")
        print(f"  L_0={u['L_0']}m  L_F={u['L_F']}m  E_use={u['E_use']}kWh  "
              f"ρ={u['ρ_g']}%  E_avail={E_avail:.4f}kWh")
        print(f"  {'服务区':<8} {'距离(m)':>10} {'H_up(m)':>10} "
              f"{'m_max(kg)':>12} {'E_round(kWh)':>14} {'绑定':>6}")
        print(f"  {'-'*56}")

        for si in model.routes["distance"].columns:
            if not si.startswith("S"):
                continue
            m_max = model.max_safe_payload(si)
            H_up = float(model.routes["climb_height"].loc["O01", si])
            d = float(model.routes["distance"].loc["O01", si])
            E_rt = model.round_trip_energy(m_max, si)
            binding = "Q_g" if m_max >= u["Q_g"] - 1e-6 else "能量"

            results.append({
                "type": g, "service": si,
                "distance": d, "H_up_out": H_up,
                "m_energy": round(m_max, 4),
                "E_round_kWh": round(E_rt, 6),
                "binding": binding,
            })

            print(f"  {si:<8} {d:>10.0f} {H_up:>10.1f} "
                  f"{m_max:>12.2f} {E_rt:>14.6f} {binding:>6}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(PROJECT / "data" / "Q1_max_payload.csv",
                  index=False, encoding="utf-8-sig")
    print("\n已保存: data/Q1_max_payload.csv")
    return df_res, models


def generate_feasible_batches(box_masses, box_volumes, m_eff, V_g):
    u"""枚举所有满足质量+体积约束的货箱子集"""
    n = len(box_masses)
    batches = []
    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            total_m = sum(box_masses[i] for i in combo)
            total_v = sum(box_volumes[i] for i in combo)
            if total_m <= m_eff + 1e-9 and total_v <= V_g + 1e-9:
                batches.append({
                    "indices": list(combo),
                    "mass": total_m,
                    "volume": total_v,
                    "n_boxes": r,
                })
    return batches


def solve_set_partition(batches, box_count, energies, times):
    u"""MILP 集合划分: 字典序 min (架次数, 能耗, 时间)"""
    n_batches = len(batches)
    if n_batches == 0:
        return None

    w_N, w_E, w_T = 1e12, 1.0, 1e-6

    c = np.array([w_N + w_E * energies[b] + w_T * times[b]
                  for b in range(n_batches)])

    A = np.zeros((box_count, n_batches))
    for b_idx, batch in enumerate(batches):
        for j in batch["indices"]:
            A[j, b_idx] = 1.0

    constraints = LinearConstraint(A, np.ones(box_count), np.ones(box_count))
    bounds = Bounds(np.zeros(n_batches), np.ones(n_batches))

    res = milp(
        c=c,
        constraints=constraints,
        bounds=bounds,
        integrality=np.ones(n_batches, dtype=int),
        options={"disp": False},
    )

    if not res.success:
        return None

    return [b_idx for b_idx, x_val in enumerate(res.x) if x_val > 0.5]


def run_batching(models):
    u"""第二步: 货箱组批优化（集合划分）"""
    print("\n" + "=" * 70)
    print("第二步：货箱组批优化（集合划分 MILP）")
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

            selected = solve_set_partition(feasible, n_boxes, energies, times)

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


def sensitivity_analysis(models):
    u"""第三步: 返航安全余量 ρ 敏感性分析"""
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
        print(f"  {'ρ(%)':<8} {'能量受限区数':>12} "
              f"{'m_max均值':>12} {'m_max最小':>12}")

        for rho in rho_values:
            u_mod = base_model.u.copy()
            u_mod["ρ_g"] = float(rho)

            from src.physics import TransportPhysicsModel
            m = TransportPhysicsModel(u_mod, base_model.routes)

            m_vals = []
            bound_count = 0
            for si in service_areas:
                m_e = m.max_safe_payload(si)
                m_vals.append(m_e)
                if m_e < Q_g - 1e-6:
                    bound_count += 1

            print(f"  {rho:<8} {bound_count:>12} "
                  f"{np.mean(m_vals):>12.1f} {np.min(m_vals):>12.1f}")

            all_sens.append({
                "type": g, "rho": rho,
                "energy_limited_areas": bound_count,
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