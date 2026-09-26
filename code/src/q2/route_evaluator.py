

























import numpy as np


def _box_data(boxes_df):

    lookup = {}
    for _, row in boxes_df.iterrows():
        lookup[row["box_id"]] = {
            "mass": float(row["mass"]),
            "volume": float(row["volume"]),
            "service": row["service"],
        }
    return lookup


def evaluate_route(model, visit_order, deliveries, boxes_df=None, box_lookup=None):
    



























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