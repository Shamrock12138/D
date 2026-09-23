u"""
Q2 基础一致性验证
=================

自动检查:
  1. 数据完整性 (节点、货箱、无人机、电池、航段矩阵)
  2. 单点退化一致性 (evaluate_route 退化到 O01→Si→O01 == Q1 往返结果)
  3. 多点路线动态载荷逻辑 (载荷逐段递减、返航空载)
  4. 电池充电模型边界值

所有检查通过后 printed "Q2 FOUNDATION CHECK PASSED".
"""

import numpy as np
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent

TOL = 1e-8


def _load_route_matrices():
    u"""加载航段矩阵"""
    matrices = {}
    for key in ["distance", "climb_height", "descent_height"]:
        matrices[key] = pd.read_csv(
            PROJECT / "data" / f"{key}_matrix.csv", index_col=0
        )
    return matrices


def run_foundation_check(verbose=True):
    u"""运行全部基础一致性检查。

    Returns
    -------
    dict
        passed : bool
        checks : list[(name, status, detail)]
    """
    results = []
    all_pass = True

    def log(name, ok, detail=""):
        nonlocal all_pass
        if not ok:
            all_pass = False
        results.append((name, ok, detail))
        if verbose:
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))

    # ── 数据完整性 ────────────────────────────────────────
    matrices = _load_route_matrices()
    nodes = list(matrices["distance"].index)

    log("服务节点数", len(nodes) == 16, f"got {len(nodes)}")
    log("服务区数", len(nodes) - 1 == 15, f"got {len(nodes)-1}")
    log("O01 存在", "O01" in nodes)

    from .data_model import load_q2_data
    data = load_q2_data()

    log("货箱总数", len(data["boxes"]) == 80, f"got {len(data['boxes'])}")
    log("无人机数", len(data["uavs"]) == 8, f"got {len(data['uavs'])}")
    log("电池数", len(data["batteries"]) == 14, f"got {len(data['batteries'])}")

    uav_types = data["uavs"]["type"].value_counts()
    for g, expected in [("A", 4), ("B", 2), ("C", 2)]:
        log(f"{g}型无人机", uav_types.get(g, 0) == expected,
            f"count={uav_types.get(g, 0)} expected {expected}")

    bat_types = data["batteries"]["type"].value_counts()
    for g, expected in [("A", 6), ("B", 4), ("C", 4)]:
        log(f"{g}型电池", bat_types.get(g, 0) == expected,
            f"got {bat_types.get(g, 0)} expected {expected}")

    for key, shape in [("distance", (16, 16)), ("climb_height", (16, 16)),
                       ("descent_height", (16, 16))]:
        log(f"{key} matrix", matrices[key].shape == shape,
            f"got {matrices[key].shape}")

    # ── 物理模型加载 ──────────────────────────────────────
    from src.physics import load_models
    models = load_models()

    # ── 桥接测试: 单点退化一致性 ─────────────────────────
    from .route_evaluator import evaluate_route
    boxes_df = data["boxes"]

    test_svs = ["S001", "S005", "S009", "S012"]

    for g_name, model in models.items():
        for sv in test_svs:
            sv_boxes = boxes_df[boxes_df["service"] == sv]
            if len(sv_boxes) == 0:
                continue

            subset = sv_boxes.head(min(3, len(sv_boxes)))
            total_vol = float(subset["volume"].sum())
            while total_vol > float(model.u["V_g"]) + TOL and len(subset) > 1:
                subset = subset.iloc[:-1]
                total_vol = float(subset["volume"].sum())

            box_ids = list(subset["box_id"])
            total_mass = float(subset["mass"].sum())
            total_n = len(box_ids)

            if total_mass > float(model.u["Q_g"]) or len(subset) == 0:
                continue

            result = evaluate_route(
                model, [sv], {sv: box_ids}, boxes_df=boxes_df
            )

            if not result["feasible"]:
                log(f"桥接-{g_name}-{sv} 可行性",
                    False, f"evaluate_route 标记为不可行: {result['reason']}")
                continue

            # Q1 往返能耗
            q1_energy = model.round_trip_energy(total_mass, sv)
            q2_energy = result["energy_kwh"]
            energy_ok = abs(q2_energy - q1_energy) < TOL

            log(f"桥接-能耗-{g_name}-{sv}", energy_ok,
                f"Q1={q1_energy:.6f} Q2={q2_energy:.6f} diff={abs(q2_energy-q1_energy):.2e}")

            # Q1 往返时间
            q1_time = model.sortie_total_time(total_n, sv)
            q2_time = result["duration_s"]
            time_ok = abs(q2_time - q1_time) < TOL

            log(f"桥接-时间-{g_name}-{sv}", time_ok,
                f"Q1={q1_time:.6f} Q2={q2_time:.6f} diff={abs(q2_time-q1_time):.2e}")

    # ── 多点路线载荷逻辑 ──────────────────────────────────
    if len(test_svs) >= 2:
        sv_a, sv_b = test_svs[0], test_svs[1]
        boxes_a = boxes_df[boxes_df["service"] == sv_a]
        boxes_b = boxes_df[boxes_df["service"] == sv_b]
        if len(boxes_a) > 0 and len(boxes_b) > 0:
            bid_a = [boxes_a.iloc[0]["box_id"]]
            bid_b = [boxes_b.iloc[0]["box_id"]]
            mass_a = float(boxes_a.iloc[0]["mass"])
            mass_b = float(boxes_b.iloc[0]["mass"])
            m = models["A"]

            if (mass_a + mass_b) <= float(m.u["Q_g"]):
                r2 = evaluate_route(m, [sv_a, sv_b],
                                    {sv_a: bid_a, sv_b: bid_b},
                                    boxes_df=boxes_df)
                if r2["feasible"] and len(r2["legs"]) == 3:
                    p0, p1, p2 = [lg["payload_kg"] for lg in r2["legs"]]

                    descending = (
                        p0 > p1 + TOL
                        and p1 > p2 + TOL
                        and abs(p2) < TOL
                    )
                    log("多点-载荷逐段下降", descending,
                        f"payloads={[f'{p:.1f}' for p in (p0, p1, p2)]}")

                    log("多点-返航载荷=0", abs(r2["legs"][-1]["payload_kg"]) < TOL)

    # ── 电池充电模型 ──────────────────────────────────────
    from .battery import soc_after_task, charge_time_to_full

    for g_name, model in models.items():
        E_cap = float(model.u["E_use"])
        soc0 = soc_after_task(0.0, E_cap)
        log(f"SOC-{g_name}-空任务", abs(soc0 - 1.0) < TOL,
            f"SOC={soc0:.6f}")

    bat_times = dict(zip(
        data["batteries"]["type"], data["batteries"]["full_charge_time"]
    ))
    for g, Tf in bat_times.items():
        t0 = charge_time_to_full(0.0, Tf)
        log(f"充电-{g}-SOC=0", abs(t0 - Tf) < TOL,
            f"T_chg={t0:.1f} T_full={Tf}")
        t09 = charge_time_to_full(0.9, Tf)
        expected_09 = 0.35 * Tf
        log(f"充电-{g}-SOC=0.9", abs(t09 - expected_09) < TOL,
            f"T_chg={t09:.1f} expected {expected_09:.1f}")
        t1 = charge_time_to_full(1.0, Tf)
        log(f"充电-{g}-SOC=1", abs(t1) < TOL,
            f"T_chg={t1:.1f}")

    # ── 汇总 ──────────────────────────────────────────────
    if verbose:
        n_pass = sum(1 for _, ok, _ in results if ok)
        n_fail = len(results) - n_pass
        print()
        if all_pass:
            print("Q2 FOUNDATION CHECK PASSED")
        else:
            print(f"Q2 FOUNDATION CHECK: {n_fail} FAILURES / {len(results)} total")

    return {"passed": all_pass, "checks": results}


if __name__ == "__main__":
    run_foundation_check()