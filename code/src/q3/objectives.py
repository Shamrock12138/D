"""Q3's four objective definitions and independent schedule evaluation.

F1 is measured in priority-weighted seconds.  A box is included only when it
has an expected delivery time and no hard deadline.  F3 uses kWh and F4 counts
transport plus relay sorties.
"""

import math


OBJECTIVE_NAMES = ("F1_timeliness", "F2_joint_cmax_s", "F3_total_energy_kWh", "F4_total_sorties")


def soft_box_targets(boxes, deadlines):
    """Return box_id -> (expected seconds, priority weight) for F1."""
    targets = {}
    for row in boxes.itertuples(index=False):
        box_id = str(row.box_id)
        if math.isfinite(deadlines[box_id]) or math.isnan(float(row.expected_time)):
            continue
        weight = float(row.priority)
        if weight < 0:
            raise ValueError(f"Negative priority weight for {box_id}")
        targets[box_id] = (float(row.expected_time), weight)
    return targets


def evaluate_objectives(problem, transport, relay, delivery):
    """Evaluate the actual, unrounded output schedules."""
    targets = soft_box_targets(problem["boxes"], problem["deadlines"])
    delivered = delivery.set_index("box_id")
    missing = set(targets) - set(delivered.index.astype(str))
    if missing:
        raise ValueError(f"Missing soft-deadline deliveries: {sorted(missing)[:10]}")
    f1 = sum(
        weight * max(0.0, float(delivered.loc[box_id, "delivery_time_s"]) - expected)
        for box_id, (expected, weight) in targets.items()
    )
    relay_cmax = float(relay["return_time_s"].max()) if len(relay) else 0.0
    return {
        "F1_timeliness": f1,
        "F2_joint_cmax_s": max(float(transport["end_time_s"].max()), relay_cmax),
        "F3_total_energy_kWh": float(transport["energy_kWh"].sum() + relay["relay_energy_kWh"].sum()),
        "F4_total_sorties": int(len(transport) + len(relay)),
    }
