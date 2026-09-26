"""Physical relay-session energy accounting and shared energy-component use."""

import pandas as pd

from src.q2.battery import charge_time_to_full, soc_after_task


SESSION_COLUMNS = (
    "relay_session_id", "relay_uav_id", "candidate_id", "dispatch_time_s",
    "arrival_time_s", "service_end_time_s", "service_duration_s",
    "return_time_s", "uav_release_time_s", "outbound_energy_kWh",
    "return_energy_kWh", "service_energy_kWh", "relay_session_energy_kWh",
    "end_soc", "charge_duration_s", "energy_release_time_s",
    "energy_component_id",
)


def _with_energy_components(relay, relay_options=None):
    required = {"outbound_energy_kWh", "return_energy_kWh", "service_energy_kWh"}
    if required <= set(relay.columns):
        return relay.copy()
    if relay_options is None or not required <= set(relay_options.columns):
        raise ValueError("Relay session energy components are missing and cannot be recovered")
    keys = ["gap_id", "pattern_id", "candidate_id"]
    merged = relay.merge(
        relay_options[keys + sorted(required)].drop_duplicates(keys),
        on=keys, how="left", validate="many_to_one", suffixes=("", "_option"))
    for column in required:
        option_column = f"{column}_option"
        if option_column in merged:
            merged[column] = merged[column].fillna(merged[option_column])
            merged.drop(columns=[option_column], inplace=True)
    if merged[list(required)].isna().any().any():
        raise ValueError("Could not recover relay session energy components from Step7 options")
    return merged


def session_energy_by_id(relay, relay_params, relay_options=None):
    """Compute physical session energy from one flight and continuous dwell."""
    if relay is None or relay.empty:
        return {}
    required = {"outbound_energy_kWh", "return_energy_kWh", "service_energy_kWh"}
    option_has_components = (
        relay_options is not None and required <= set(relay_options.columns)
    )
    if not required <= set(relay.columns) and not option_has_components:
        if "relay_session_id" in relay:
            raise ValueError(
                "Cannot recompute shared-session energy without outbound, return, "
                "and service components"
            )
        if "relay_energy_kWh" not in relay:
            raise ValueError("Legacy relay schedule has no recoverable energy fields")
        # Legacy gap-only schedules had no shared physical-session accounting.
        return {str(index): float(value)
                for index, value in relay["relay_energy_kWh"].items()}
    frame = _with_energy_components(relay, relay_options)
    required_times = {"arrival_time_s", "service_end_s"}
    if not required_times <= set(frame.columns):
        raise ValueError("Relay rows need arrival_time_s and service_end_s for session energy")
    session_col = "relay_session_id" if "relay_session_id" in frame else "gap_id"
    energy_scale = 1_000_000
    result = {}
    for session_id, group in frame.groupby(session_col, sort=False):
        dwell_s = max(
            0.0,
            float(group["service_end_s"].max())
            - float(group["arrival_time_s"].min()),
        )
        service_kwh = float(relay_params.service_power_kw) * dwell_s / 3600.0
        total = (
            round(float(group["outbound_energy_kWh"].max()) * energy_scale)
            + round(float(group["return_energy_kWh"].max()) * energy_scale)
            + round(service_kwh * energy_scale)
        ) / energy_scale
        result[str(session_id)] = total
    return result


def build_relay_session_table(relay, relay_params, session_capacity_kwh=None,
                              relay_options=None):
    """Collapse gap certificates into physical sessions and compute session energy.

    A session incurs outbound and return energy once (the maximum certified
    leg energy within the session) and continuous hover/communication dwell
    from the first arrival to the last covered service end.
    """
    if relay is None or relay.empty:
        return pd.DataFrame(columns=SESSION_COLUMNS)
    frame = _with_energy_components(relay, relay_options)
    energy_by_session = session_energy_by_id(frame, relay_params)
    if "relay_session_id" not in frame:
        frame["relay_session_id"] = frame["gap_id"].astype(str)
    capacity = float(session_capacity_kwh or relay_params.energy_capacity_kwh)
    rows = []
    for session_id, group in frame.groupby("relay_session_id", sort=True):
        candidates = group["candidate_id"].astype(str).unique()
        relay_ids = group["relay_uav_id"].astype(str).unique()
        if len(candidates) != 1 or len(relay_ids) != 1:
            raise ValueError(f"Session {session_id} spans relay UAVs or sites")
        outbound = float(group["outbound_energy_kWh"].max())
        returning = float(group["return_energy_kWh"].max())
        arrival = float(group["arrival_time_s"].min())
        service_end = float(group["service_end_s"].max())
        dwell_s = max(0.0, service_end - arrival)
        service = float(relay_params.service_power_kw) * dwell_s / 3600.0
        total = energy_by_session[str(session_id)]
        end_soc = soc_after_task(total, capacity)
        charge_s = float(charge_time_to_full(end_soc, relay_params.full_charge_time_s))
        dispatch = float(group["dispatch_time_s"].min())
        return_time = float(group["return_time_s"].max())
        rows.append({
            "relay_session_id": str(session_id),
            "relay_uav_id": str(relay_ids[0]),
            "candidate_id": str(candidates[0]),
            "dispatch_time_s": dispatch,
            "arrival_time_s": arrival,
            "service_end_time_s": service_end,
            "service_duration_s": dwell_s,
            "return_time_s": return_time,
            "uav_release_time_s": float(group["uav_release_time_s"].max()),
            "outbound_energy_kWh": outbound,
            "return_energy_kWh": returning,
            "service_energy_kWh": service,
            "relay_session_energy_kWh": total,
            "end_soc": end_soc,
            "charge_duration_s": charge_s,
            "energy_release_time_s": return_time + charge_s,
            "energy_component_id": "",
        })
    return pd.DataFrame(rows, columns=SESSION_COLUMNS)


def assign_session_energy_components(sessions, component_count=6):
    """Greedily color charging intervals; return (assigned table, all_fit)."""
    result = sessions.copy()
    if result.empty:
        if "energy_component_id" not in result:
            result["energy_component_id"] = pd.Series(dtype=str)
        return result, True
    component_count = int(component_count)
    ready = {f"E{i:02d}": 0.0 for i in range(1, component_count + 1)}
    assigned = {}
    ordered = result.sort_values(
        ["dispatch_time_s", "energy_release_time_s", "relay_session_id"]
    )
    for row in ordered.itertuples(index=False):
        start = float(row.dispatch_time_s)
        available = [component for component, free_at in ready.items()
                     if free_at <= start + 1e-9]
        if not available:
            result["energy_component_id"] = result["relay_session_id"].map(assigned).fillna("")
            return result, False
        component = min(available, key=lambda item: (ready[item], item))
        assigned[str(row.relay_session_id)] = component
        ready[component] = float(row.energy_release_time_s)
    result["energy_component_id"] = result["relay_session_id"].astype(str).map(assigned)
    return result, True


def attach_session_resources(relay, relay_params, component_count=6, relay_options=None):
    """Return gap rows annotated with session energy/component data and sessions."""
    sessions = build_relay_session_table(relay, relay_params, relay_options=relay_options)
    sessions, feasible = assign_session_energy_components(sessions, component_count)
    if relay is None or relay.empty:
        return relay.copy() if relay is not None else pd.DataFrame(), sessions, feasible
    lookup = sessions.set_index("relay_session_id")
    result = relay.copy()
    for session_column, gap_column in (
        ("relay_session_energy_kWh", "relay_session_energy_kWh"),
        ("end_soc", "relay_session_end_soc"),
        ("energy_release_time_s", "relay_session_energy_release_time_s"),
        ("energy_component_id", "energy_component_id"),
    ):
        result[gap_column] = result["relay_session_id"].astype(str).map(
            lookup[session_column]
        )
    return result, sessions, feasible
