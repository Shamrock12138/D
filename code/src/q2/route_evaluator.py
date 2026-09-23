u"""
Q2 多点路线物理计算
====================

统一接口: evaluate_route(model, visit_order, deliveries)

核心能力
--------
1. 动态载荷 — 每航段载荷按已投送质量递减
2. 多点能耗 — 逐航段调用 model.energy(origin, dest, q)
3. 多点时间 — 逐站累计飞行/交接时间, 输出逐箱送达偏移
4. 与 Q1 退化一致 — 单点路线结果 == Q1 往返结果

使用示例
--------
>>> from src.q2.route_evaluator import evaluate_route
>>> from src.physics import load_models
>>> models = load_models()
>>> result = evaluate_route(
...     models["A"],
...     ["S003", "S007", "S012"],
...     {"S003": ["B021","B022"], "S007": ["B041"], "S012": ["B063","B064"]}
... )
>>> print(result["energy_kwh"], result["duration_s"], result["feasible"])
"""

import numpy as np


def _box_data(boxes_df):
    u"""将 boxes_df 转换为 {box_id: {mass, volume, service}} 字典"""
    lookup = {}
    for _, row in boxes_df.iterrows():
        lookup[row["box_id"]] = {
            "mass": float(row["mass"]),
            "volume": float(row["volume"]),
            "service": row["service"],
        }
    return lookup


def evaluate_route(model, visit_order, deliveries, boxes_df=None, box_lookup=None):
    u"""评估一条多点往返路线的物理可行性、能耗和时间。

    Parameters
    ----------
    model : TransportPhysicsModel
        已实例化的机型物理模型
    visit_order : list[str]
        访问的服务区顺序, e.g. ["S003","S007","S012"]
    deliveries : dict[str, list[str]]
        每个服务区投放的货箱编号列表, 如 {"S003":["B021","B022"], ...}
    boxes_df : pd.DataFrame, optional
        货箱数据 (load_boxes() 输出), 用于获取每箱 mass/volume/service
    box_lookup : dict, optional
        预构建的 box_id → {mass, volume, service} 字典, 优先于 boxes_df

    Returns
    -------
    dict
        feasible       — 是否满足质量和体积约束
        total_mass     — 起飞总质量 (kg)
        total_volume   — 起飞总体积 (m^3)
        energy_kwh     — 全路线总能耗 (kWh)
        duration_s     — 架次累计时间 (s)
        end_soc        — 任务结束时 SOC (0~1)
        delivery_offsets — {box_id: 相对送达时间 (s)}
        legs           — 逐航段明细列表
        reason         — 不可行时的原因描述
    """
    u = model.u
    if box_lookup is None:
        box_lookup = _box_data(boxes_df) if boxes_df is not None else {}

    if len(visit_order) == 0:
        return {
            "feasible": True, "total_mass": 0.0, "total_volume": 0.0,
            "energy_kwh": 0.0, "duration_s": 0.0, "end_soc": 1.0,
            "delivery_offsets": {}, "legs": [], "reason": None,
        }

    total_boxes = sum(len(v) for v in deliveries.values())
    total_mass = 0.0
    total_volume = 0.0

    for sv in visit_order:
        for box_id in deliveries.get(sv, []):
            if box_id in box_lookup:
                total_mass += box_lookup[box_id]["mass"]
                total_volume += box_lookup[box_id]["volume"]

    if total_mass > float(u["Q_g"]):
        return {
            "feasible": False, "total_mass": total_mass, "total_volume": total_volume,
            "energy_kwh": np.nan, "duration_s": np.nan, "end_soc": np.nan,
            "delivery_offsets": {}, "legs": [],
            "reason": f"总质量 {total_mass:.1f}kg 超过最大载荷 {u['Q_g']}kg",
        }

    if total_volume > float(u["V_g"]):
        return {
            "feasible": False, "total_mass": total_mass, "total_volume": total_volume,
            "energy_kwh": np.nan, "duration_s": np.nan, "end_soc": np.nan,
            "delivery_offsets": {}, "legs": [],
            "reason": f"总体积 {total_volume:.4f}m^3 超过可用容积 {u['V_g']}m^3",
        }

    remainder = dict(total_mass=total_mass, total_volume=total_volume)

    # ── 累积时间 ──────────────────────────────────────────
    elapsed = u["T_setup"] + u["T_load"] * total_boxes
    delivery_offsets = {}

    legs = []
    energy_total = 0.0
    current_q = total_mass

    nodes = ["O01"] + visit_order + ["O01"]

    for k in range(len(nodes) - 1):
        origin = nodes[k]
        dest = nodes[k + 1]

        if dest != "O01":
            sv = dest
            n_deliver = len(deliveries.get(sv, []))
        else:
            n_deliver = 0

        t_fly = model.time(origin, dest)
        e_leg = model.energy(origin, dest, current_q)

        leg_info = {
            "origin": origin, "destination": dest,
            "payload_kg": current_q, "energy_kwh": e_leg,
            "time_s": t_fly, "boxes_delivered": n_deliver,
        }
        legs.append(leg_info)

        energy_total += e_leg
        elapsed += t_fly

        if dest != "O01":
            sv = dest
            boxes_here = deliveries.get(sv, [])
            if boxes_here:
                elapsed += u["T_handover"] + u["T_handover_p"] * len(boxes_here)
                for bid in boxes_here:
                    delivery_offsets[bid] = elapsed

            mass_dropped = sum(
                box_lookup[bid]["mass"] if bid in box_lookup else 0.0
                for bid in boxes_here
            )
            current_q -= mass_dropped

    end_soc = 1.0 - energy_total / float(u["E_use"])
    e_avail = (1.0 - float(u["ρ_g"]) / 100.0) * float(u["E_use"])

    if energy_total > e_avail:
        return {
            "feasible": False, "total_mass": total_mass, "total_volume": total_volume,
            "energy_kwh": energy_total, "duration_s": elapsed, "end_soc": end_soc,
            "delivery_offsets": delivery_offsets, "legs": legs,
            "reason": (
                f"能耗 {energy_total:.4f}kWh 超过可用电能 {e_avail:.4f}kWh "
                f"(SOC={end_soc:.4f})"
            ),
        }

    return {
        "feasible": True,
        "total_mass": total_mass,
        "total_volume": total_volume,
        "energy_kwh": energy_total,
        "duration_s": elapsed,
        "end_soc": end_soc,
        "delivery_offsets": delivery_offsets,
        "legs": legs,
        "reason": None,
    }