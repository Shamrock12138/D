"""Q3's four objective definitions and independent schedule evaluation.

F1 is measured in priority-weighted seconds.  A box is included only when it
has an expected delivery time and no hard deadline.  F3 uses kWh and F4 counts
transport plus relay sorties.
"""

import math


OBJECTIVE_NAMES = ("F1_timeliness", "F2_joint_cmax_s", "F3_total_energy_kWh", "F4_total_sorties")
F1_TIME_SCALE = 10
ENERGY_SCALE = 1_000_000
Q3_MULTI_OBJECTIVE_TIER = "all"


def time_units(seconds):
    """Round actual output time to the 0.1-second reporting grid."""
    scaled = float(seconds) * F1_TIME_SCALE
    return int(round(scaled))


def energy_units(kwh):
    """Round one sortie's energy to the shared micro-kWh objective grid."""
    return int(round(float(kwh) * ENERGY_SCALE))


def soft_box_targets(boxes, deadlines):
    """Return box_id -> (expected seconds, priority weight) for F1."""
    targets = {}
    for row in boxes.itertuples(index=False):
        box_id = str(row.box_id)
        if math.isfinite(deadlines[box_id]) or math.isnan(float(row.expected_time)):
            continue
        weight = float(row.priority)
        if weight < 0 or not weight.is_integer():
            raise ValueError(f"Priority weight must be a nonnegative integer for {box_id}")
        targets[box_id] = (float(row.expected_time), weight)
    return targets


def relay_session_energy_kwh(relay, relay_options=None):
    """Count each shared session's outbound/return energy once plus gap service.

    Gap-level relay rows remain the communication coverage certificates. When
    legacy frozen rows lack component energies, Step7 options are joined back
    by their certified (gap, pattern, candidate) key.
    """
    if relay is None or relay.empty:
        return 0.0
    from src.q3.relay.operation_profile import load_relay_flight_parameters
    from src.q3.session_resources import session_energy_by_id

    return float(sum(session_energy_by_id(
        relay, load_relay_flight_parameters(), relay_options
    ).values()))


def evaluate_objectives(problem, transport, relay, delivery):
    """Evaluate the actual, unrounded output schedules."""
    targets = soft_box_targets(problem["boxes"], problem["deadlines"])
    delivered = delivery.set_index("box_id")
    missing = set(targets) - set(delivered.index.astype(str))
    if missing:
        raise ValueError(f"Missing soft-deadline deliveries: {sorted(missing)[:10]}")
    f1_units = sum(
        int(weight) * max(
            0, time_units(delivered.loc[box_id, "delivery_time_s"]) - time_units(expected)
        )
        for box_id, (expected, weight) in targets.items()
    )
    relay_cmax = float(relay["return_time_s"].max()) if len(relay) else 0.0
    return {
        "F1_timeliness": f1_units / F1_TIME_SCALE,
        "F2_joint_cmax_s": max(float(transport["end_time_s"].max()), relay_cmax),
        "F3_total_energy_kWh": (
            sum(energy_units(value) for value in transport["energy_kWh"])
            / ENERGY_SCALE + relay_session_energy_kwh(relay, problem.get("relay"))
        ),
        "F4_total_sorties": int(len(transport) + (
            relay["relay_session_id"].nunique()
            if len(relay) and "relay_session_id" in relay
            else len(relay)
        )),
    }
