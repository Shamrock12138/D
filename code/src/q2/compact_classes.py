"""Lossless box symmetry classes and count-based transport patterns.

Box IDs are retained only as class members for final deterministic decoding.
All attributes that affect constraints or objectives participate in the key.
"""

from collections import defaultdict
from itertools import combinations, product
import math

import pandas as pd
from ortools.sat.python import cp_model

from src.q2.battery import charge_time_to_full, soc_after_task
from src.q2.route_evaluator import _box_data, evaluate_route


def _number(value, default=math.inf):
    """Normalize absent numeric values without merging them with finite values."""
    return default if pd.isna(value) else float(value)


def build_box_classes(boxes):
    """Return class table and physical-box to class mapping."""
    grouped = defaultdict(list)
    for row in boxes.itertuples(index=False):
        first = bool(row.is_first_batch)
        hard_deadline = math.inf
        if first:
            hard_deadline = min(hard_deadline, _number(row.first_deadline))
        if str(row.cargo_type) == "医疗物资":
            hard_deadline = min(hard_deadline, _number(row.expected_time))
        key = (str(row.service), str(row.cargo_type), float(row.mass),
               float(row.volume), hard_deadline, _number(row.expected_time),
               int(row.priority), first)
        grouped[key].append(str(row.box_id))
    records = []
    box_to_class = {}
    for index, (key, members) in enumerate(sorted(grouped.items()), start=1):
        class_id = f"C{index:03d}"
        members.sort()
        for box_id in members:
            box_to_class[box_id] = class_id
        records.append(dict(zip(
            ("service", "cargo_type", "mass", "volume", "hard_deadline_s",
             "expected_time_s", "priority", "is_first_batch"), key
        )) | {"class_id": class_id, "count": len(members),
              "box_ids": tuple(members)})
    return pd.DataFrame(records), box_to_class


def enumerate_service_loads(classes, service, max_mass, max_volume):
    """Enumerate class-count vectors within one service and UAV payload limits."""
    local = classes.loc[classes["service"] == service]
    rows = list(local.itertuples(index=False))
    for amounts in product(*(range(int(row.count) + 1) for row in rows)):
        if not any(amounts):
            continue
        mass = sum(n * row.mass for n, row in zip(amounts, rows))
        volume = sum(n * row.volume for n, row in zip(amounts, rows))
        if mass <= max_mass and volume <= max_volume:
            yield {row.class_id: n for n, row in zip(amounts, rows) if n}, mass, volume


def generate_compact_patterns(boxes, models, max_stops=2, services=None,
                              progress=None):
    """Generate one/two-stop physical patterns with class counts, never box IDs.

    Representative member IDs are used only transiently by the established
    route physics evaluator; they are absent from the returned pattern tables.
    """
    if max_stops not in (1, 2):
        raise ValueError("Only one- and two-stop pattern generation is implemented")
    classes, _ = build_box_classes(boxes)
    class_rows = classes.set_index("class_id")
    service_ids = sorted(services if services is not None else classes["service"].unique())
    lookup = _box_data(boxes)
    pattern_rows = []
    count_rows = []
    next_id = 0
    for uav_type, model in sorted(models.items()):
        route_cache = {}
        loads = {service: list(enumerate_service_loads(
            classes, service, float(model.u["Q_g"]), float(model.u["V_g"])))
            for service in service_ids}
        for stop_count in range(1, max_stops + 1):
            for site_index, sites in enumerate(combinations(service_ids, stop_count), 1):
                for selected in product(*(loads[site] for site in sites)):
                    counts = {}
                    total_mass = total_volume = 0.0
                    for class_counts, mass, volume in selected:
                        counts.update(class_counts)
                        total_mass += mass
                        total_volume += volume
                    if total_mass > float(model.u["Q_g"]) or total_volume > float(model.u["V_g"]):
                        continue
                    deliveries = {site: [box_id
                                           for class_id, amount in selected[j][0].items()
                                           for box_id in class_rows.loc[class_id].box_ids[:amount]]
                                  for j, site in enumerate(sites)}
                    orders = [sites] if stop_count == 1 else [sites, sites[::-1]]
                    for visit_order in orders:
                        physical_key = (tuple(visit_order), tuple(
                            (site, len(deliveries[site]),
                             round(sum(lookup[bid]["mass"] for bid in deliveries[site]), 9),
                             round(sum(lookup[bid]["volume"] for bid in deliveries[site]), 9))
                            for site in visit_order))
                        if physical_key not in route_cache:
                            route = evaluate_route(model, list(visit_order), deliveries,
                                                   box_lookup=lookup)
                            site_offsets = {
                                site: float(route["delivery_offsets"][deliveries[site][0]])
                                for site in visit_order if route["feasible"]
                            }
                            route_cache[physical_key] = (route, site_offsets)
                        route, site_offsets = route_cache[physical_key]
                        if not route["feasible"]:
                            continue
                        latest = math.inf
                        offsets = {}
                        for class_id, amount in counts.items():
                            row = class_rows.loc[class_id]
                            offset = site_offsets[row.service]
                            offsets[class_id] = offset
                            if math.isfinite(float(row.hard_deadline_s)):
                                latest = min(latest, float(row.hard_deadline_s) - offset)
                        if latest < 0:
                            continue
                        next_id += 1
                        pattern_id = f"P{next_id:06d}"
                        pattern_rows.append({
                            "pattern_id": pattern_id, "uav_type": uav_type,
                            "n_stops": stop_count, "visit_order": ">".join(visit_order),
                            "n_boxes": sum(counts.values()),
                            "total_mass_kg": total_mass, "total_volume_m3": total_volume,
                            "energy_kWh": float(route["energy_kwh"]),
                            "duration_s": float(route["duration_s"]),
                            "end_SOC": float(route["end_soc"]),
                            "has_hard_deadline": math.isfinite(latest),
                            "latest_start_s": latest,
                        })
                        count_rows.extend({"pattern_id": pattern_id, "class_id": class_id,
                                           "count": amount,
                                           "delivery_offset_s": offsets[class_id]}
                                          for class_id, amount in counts.items())
                if progress is not None and site_index % 20 == 0:
                    progress(uav_type, stop_count, site_index, len(pattern_rows))
    return classes, pd.DataFrame(pattern_rows), pd.DataFrame(count_rows)


def pattern_multiplicity(counts, class_supply):
    """Safe upper bound on repeat sorties for a count-based pattern."""
    if not counts:
        raise ValueError("A transport pattern must carry at least one box")
    return min(int(class_supply[class_id]) // int(amount)
               for class_id, amount in counts.items())


def select_compact_patterns(patterns, pattern_counts, classes, top_k=8):
    """Keep diverse patterns per class/UAV plus all feasible unit-class patterns."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    task_rows = patterns.set_index("pattern_id", drop=False)
    chosen = set()
    for class_id, group in pattern_counts.groupby("class_id"):
        pattern_ids = group["pattern_id"].astype(str).unique()
        local = task_rows.loc[pattern_ids]
        for _, typed in local.groupby("uav_type"):
            chosen.update(typed.nsmallest(top_k, "energy_kWh")["pattern_id"])
            chosen.update(typed.nsmallest(top_k, "duration_s")["pattern_id"])
            finite = typed.loc[typed["latest_start_s"].map(math.isfinite)]
            chosen.update(finite.nlargest(top_k, "latest_start_s")["pattern_id"])
        unit = pattern_counts.loc[
            (pattern_counts["class_id"] == class_id)
            & (pattern_counts["count"] == 1), "pattern_id"]
        sizes = pattern_counts.groupby("pattern_id").size()
        chosen.update(pid for pid in unit if sizes[pid] == 1)
    selected = patterns.loc[patterns["pattern_id"].isin(chosen)].reset_index(drop=True)
    selected_counts = pattern_counts.loc[
        pattern_counts["pattern_id"].isin(chosen)
    ].reset_index(drop=True)
    supplied = set(classes["class_id"])
    covered = set(selected_counts["class_id"])
    if supplied - covered:
        raise ValueError(f"No retained pattern for classes {sorted(supplied - covered)}")
    return selected, selected_counts


def expand_pattern_counts(pattern_counts, class_supply):
    """Create distinguishable sortie slots; a pattern may appear repeatedly."""
    slots = []
    for pattern_id, counts in pattern_counts.items():
        maximum = pattern_multiplicity(counts, class_supply)
        for copy in range(1, maximum + 1):
            slots.append({"sortie_id": f"{pattern_id}-{copy}",
                          "pattern_id": pattern_id, "copy": copy,
                          "class_counts": dict(counts)})
    return slots


def add_class_conservation(model, slots, chosen, class_supply):
    """Enforce exact quantity, not ExactlyOne on representative box IDs."""
    for class_id, supply in class_supply.items():
        terms = [slot["class_counts"].get(class_id, 0) * chosen[i]
                 for i, slot in enumerate(slots)
                 if slot["class_counts"].get(class_id, 0)]
        if not terms and supply:
            raise ValueError(f"Class {class_id} has no transport pattern")
        model.Add(sum(terms) == int(supply))


def assign_box_ids(selected_sorties, classes):
    """Decode selected class counts to each original box exactly once."""
    available = {row.class_id: list(row.box_ids)
                 for row in classes.itertuples(index=False)}
    assignments = []
    for sortie in selected_sorties:
        for class_id, amount in sorted(sortie["class_counts"].items()):
            pool = available[class_id]
            if amount > len(pool):
                raise ValueError(f"Over-assigned class {class_id}")
            for box_id in pool[:amount]:
                assignments.append({"sortie_id": sortie["sortie_id"],
                                    "class_id": class_id, "box_id": box_id})
            available[class_id] = pool[amount:]
    if any(available.values()):
        raise ValueError("Selected sorties leave boxes undelivered")
    return pd.DataFrame(assignments)


def decode_box_deliveries(selected_sorties, classes, pattern_counts):
    """Produce the required per-box delivery times after scheduling sorties."""
    assigned = assign_box_ids(selected_sorties, classes)
    if assigned.empty:
        return assigned
    sortie_rows = pd.DataFrame([
        {"sortie_id": sortie["sortie_id"], "pattern_id": sortie["pattern_id"],
         "start_time_s": float(sortie["start_time_s"])}
        for sortie in selected_sorties
    ])
    offsets = pattern_counts[["pattern_id", "class_id", "delivery_offset_s"]]
    result = assigned.merge(sortie_rows, on="sortie_id", validate="many_to_one")
    result = result.merge(offsets, on=["pattern_id", "class_id"],
                          validate="many_to_one")
    result["delivery_time_s"] = result["start_time_s"] + result["delivery_offset_s"]
    if result["box_id"].duplicated().any():
        raise AssertionError("A physical box was delivered more than once")
    return result.sort_values("box_id").reset_index(drop=True)


def class_timeliness(selected_sorties, classes, pattern_counts):
    """Return F1 = sum(count * priority * lateness) without box-ID variables."""
    class_rows = classes.set_index("class_id")
    offsets = {(row.pattern_id, row.class_id): float(row.delivery_offset_s)
               for row in pattern_counts.itertuples(index=False)}
    total = 0.0
    for sortie in selected_sorties:
        for class_id, amount in sortie["class_counts"].items():
            row = class_rows.loc[class_id]
            if (not math.isfinite(float(row.hard_deadline_s))
                    and math.isfinite(float(row.expected_time_s))):
                delivery_time = (float(sortie["start_time_s"])
                                 + offsets[(sortie["pattern_id"], class_id)])
                total += (int(amount) * float(row.priority)
                          * max(0.0, delivery_time - float(row.expected_time_s)))
    return total


def materialize_selected_sorties(selected_sorties, patterns, pattern_counts, classes):
    """Bridge scheduled compact sorties to legacy Q2/Q3 task and delivery tables.

    This is called only after category conservation is solved; box IDs are
    assigned once here, never enumerated as candidate decision variables.
    """
    pattern_rows = patterns.set_index("pattern_id")
    task_rows = []
    for sortie in selected_sorties:
        row = pattern_rows.loc[sortie["pattern_id"]].to_dict()
        row["task_id"] = sortie["sortie_id"]
        row["cargo_pattern"] = "|".join(
            f"{class_id}:{amount}" for class_id, amount in
            sorted(sortie["class_counts"].items()))
        row["charge_time_s"] = 0.0
        row["start_hint_s"] = float(sortie["start_time_s"])
        task_rows.append(row)
    delivery = decode_box_deliveries(selected_sorties, classes, pattern_counts)
    delivery = delivery.rename(columns={"sortie_id": "task_id"})
    hard = dict(zip(classes["class_id"], classes["hard_deadline_s"]))
    delivery["deadline_s"] = delivery["class_id"].map(hard)
    return pd.DataFrame(task_rows), delivery


def build_compact_master(patterns, pattern_counts, classes, uav_ids,
                         battery_ids, energy_capacity, charge_full, horizon_s,
                         objective="N"):
    """Build transport-feasible class-count master with a Q2 single objective."""
    if objective not in {"F1", "Cmax", "E", "N"}:
        raise ValueError(f"Unknown compact Q2 objective: {objective}")
    supply = dict(zip(classes["class_id"], classes["count"]))
    counts = {pid: dict(zip(group["class_id"], group["count"]))
              for pid, group in pattern_counts.groupby("pattern_id")}
    count_rows = {pid: tuple(group.itertuples(index=False))
                  for pid, group in pattern_counts.groupby("pattern_id")}
    class_rows = classes.set_index("class_id")
    class_deadlines = dict(zip(classes["class_id"], classes["hard_deadline_s"]))
    pattern_deadlines = {}
    for pattern_id, group in pattern_counts.groupby("pattern_id"):
        pattern_deadlines[pattern_id] = min(
            (float(class_deadlines[row.class_id]) - float(row.delivery_offset_s)
             for row in group.itertuples(index=False)
             if math.isfinite(float(class_deadlines[row.class_id]))),
            default=math.inf,
        )
    slots = expand_pattern_counts(counts, supply)
    pattern_rows = patterns.set_index("pattern_id")
    model = cp_model.CpModel()
    chosen = []
    starts = []
    active_ends = []
    energy_terms = []
    lateness_terms = []
    flight_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)
    battery_horizon = horizon_s + math.ceil(max(
        charge_time_to_full(0.0, full) for full in charge_full.values()))
    previous = {}
    for i, slot in enumerate(slots):
        row = pattern_rows.loc[slot["pattern_id"]]
        typ = str(row.uav_type)
        flight = max(1, math.ceil(float(row.duration_s)))
        soc = soc_after_task(float(row.energy_kWh), energy_capacity[typ])
        charge = charge_time_to_full(soc, charge_full[typ])
        battery = max(flight, math.ceil(float(row.duration_s) + charge))
        latest = horizon_s - flight
        if math.isfinite(pattern_deadlines[slot["pattern_id"]]):
            latest = min(latest, math.floor(pattern_deadlines[slot["pattern_id"]]))
        x = model.NewBoolVar(f"choose_{i}")
        s = model.NewIntVar(0, max(0, latest), f"start_{i}")
        if latest < 0:
            model.Add(x == 0)
        f_end = model.NewIntVar(flight, horizon_s, f"flight_end_{i}")
        b_end = model.NewIntVar(battery, battery_horizon, f"battery_end_{i}")
        flight_intervals[typ].append(model.NewOptionalIntervalVar(
            s, flight, f_end, x, f"flight_{i}"))
        battery_intervals[typ].append(model.NewOptionalIntervalVar(
            s, battery, b_end, x, f"battery_{i}"))
        prior = previous.get(slot["pattern_id"])
        if prior is not None:
            model.Add(prior >= x)
        previous[slot["pattern_id"]] = x
        chosen.append(x)
        starts.append(s)
        if objective == "Cmax":
            active_end = model.NewIntVar(0, horizon_s, f"active_end_{i}")
            model.Add(active_end == f_end).OnlyEnforceIf(x)
            model.Add(active_end == 0).OnlyEnforceIf(x.Not())
            active_ends.append(active_end)
        elif objective == "E":
            energy_terms.append(int(round(float(row.energy_kWh) * 1_000_000)) * x)
        elif objective == "F1":
            for item in count_rows[slot["pattern_id"]]:
                cls = class_rows.loc[item.class_id]
                if (math.isfinite(float(cls.hard_deadline_s))
                        or not math.isfinite(float(cls.expected_time_s))):
                    continue
                diff_ms = math.ceil(1000 * (float(item.delivery_offset_s)
                                            - float(cls.expected_time_s)))
                upper = max(0, 1000 * horizon_s + diff_ms)
                late = model.NewIntVar(0, upper, f"late_ms_{i}_{item.class_id}")
                model.Add(late >= 1000 * s + diff_ms).OnlyEnforceIf(x)
                model.Add(late == 0).OnlyEnforceIf(x.Not())
                lateness_terms.append(int(item.count) * int(cls.priority) * late)
    add_class_conservation(model, slots, chosen, supply)
    for typ, intervals in flight_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(uav_ids[typ]))
    for typ, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(battery_ids[typ]))
    if objective == "N":
        model.Minimize(sum(chosen))
    elif objective == "E":
        model.Minimize(sum(energy_terms))
    elif objective == "Cmax":
        cmax = model.NewIntVar(0, horizon_s, "cmax")
        model.AddMaxEquality(cmax, active_ends)
        model.Minimize(cmax)
    else:
        model.Minimize(sum(lateness_terms))
    return model, slots, chosen, starts
