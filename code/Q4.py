"""Q4: exact partition and minimum resource pools for a frozen Q3 schedule.

Run after Step13: python code/Q4.py
The input contract and mathematical definitions are in code/Q4_model.md.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "data" / "q3_final_frozen"
OUT = ROOT / "data" / "q4"
SERVICES = tuple(f"S{i:03d}" for i in range(1, 16))
KINDS = ("TUAV_A", "TUAV_B", "TUAV_C", "TBAT_A", "TBAT_B", "TBAT_C", "RUAV", "REC")
# From the current attachment; Step13 must freeze and check these inventory values.
INVENTORY = (4, 2, 2, 6, 4, 4, 2, 6)
FILES = (
    "q3_joint_transport_schedule.csv", "q3_joint_relay_schedule.csv",
    "q3_joint_delivery_schedule.csv", "q3_joint_resource_summary.csv",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required(row: dict, field: str, source: str) -> str:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{source}: missing {field}; see Q4_model.md Step13 contract")
    return str(value).strip()


def number(row: dict, field: str, source: str) -> float:
    value = float(required(row, field, source))
    if not math.isfinite(value):
        raise ValueError(f"{source}: {field} must be finite")
    return value


def normalize_sortie(row: dict, source: str) -> str:
    ids = [str(row[name]).strip() for name in ("sortie_id", "task_id") if row.get(name)]
    if not ids or len(set(ids)) != 1:
        raise ValueError(f"{source}: missing or conflicting sortie_id/task_id")
    return ids[0]


def load_final_q3(folder: Path = FROZEN) -> dict:
    needed = [folder / name for name in (*FILES, "acceptance.json")]
    absent = [path.name for path in needed if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Q4 cannot run: final Q3 solution is not frozen; missing {absent}")
    acceptance = json.loads((folder / "acceptance.json").read_text(encoding="utf-8"))
    if acceptance.get("status") not in ("FEASIBLE", "OPTIMAL") or acceptance.get("validation", {}).get("all_pass") is not True:
        raise ValueError("Q4 cannot run: final Q3 acceptance failed")
    checksums = acceptance.get("outputs_sha256") or acceptance.get("input_sha256", {})
    for name in FILES:
        if name not in checksums or checksums[name] != digest(folder / name):
            raise ValueError(f"Q3 frozen file is not acceptance-hash verified: {name}")
    transport = read_csv(folder / FILES[0])
    relay = read_csv(folder / FILES[1])
    delivery = read_csv(folder / FILES[2])
    summary = read_csv(folder / FILES[3])
    if not transport or len(summary) != 1:
        raise ValueError("Q3 transport table must be nonempty and summary must have one row")
    seen = set()
    for row in transport:
        sid = normalize_sortie(row, "transport")
        if sid in seen:
            raise ValueError(f"duplicate transport sortie: {sid}")
        seen.add(sid)
        row["sortie_id"] = sid
        visits = required(row, "visit_order", f"transport {sid}").replace("→", ">").replace(",", ">")
        sites = tuple(s.strip() for s in visits.split(">") if s.strip() and s.strip() != "O01")
        if not sites or len(set(sites)) != len(sites) or not set(sites) <= set(SERVICES):
            raise ValueError(f"invalid visit_order for {sid}: {visits}")
        row["_sites"] = sites
        if required(row, "uav_type", sid) not in ("A", "B", "C"):
            raise ValueError(f"invalid uav_type for {sid}")
        a, b, c = (number(row, name, sid) for name in ("start_time_s", "end_time_s", "charge_end_s"))
        if not a < b <= c:
            raise ValueError(f"invalid transport/battery interval for {sid}")
        if row.get("uav_release_time_s"):
            release = number(row, "uav_release_time_s", sid)
            if release < b:
                raise ValueError(f"transport UAV release precedes flight end for {sid}")
    if set().union(*(set(row["_sites"]) for row in transport)) != set(SERVICES):
        raise ValueError("frozen Q3 transport routes do not cover all 15 services")
    for i, row in enumerate(relay):
        source = f"relay row {i + 1}"
        links = row.get("sortie_ids") or row.get("sortie_id") or row.get("task_id")
        if not links:
            raise ValueError(f"{source}: missing protected sortie IDs")
        ids = tuple(s.strip() for s in str(links).replace(";", ",").split(",") if s.strip())
        if not ids or not set(ids) <= seen:
            raise ValueError(f"{source}: unknown protected sortie ID")
        row["_sorties"] = ids
        a, b, c = (number(row, name, source) for name in
                   ("dispatch_time_s", "uav_release_time_s", "energy_release_time_s"))
        if not a < b or not a < c:
            raise ValueError(f"{source}: invalid relay interval")
    seen_boxes = set()
    visits_by_sortie = {row["sortie_id"]: set(row["_sites"]) for row in transport}
    for i, row in enumerate(delivery):
        source = f"delivery row {i + 1}"
        box = required(row, "box_id", source)
        if box in seen_boxes:
            raise ValueError(f"duplicate delivered box: {box}")
        seen_boxes.add(box)
        sid = normalize_sortie(row, source)
        service = required(row, "service", source)
        if sid not in visits_by_sortie or service not in visits_by_sortie[sid]:
            raise ValueError(f"{source}: box service not on frozen sortie route")
        row["sortie_id"] = sid
        row["_mass_kg"] = number(row, "mass_kg", source)
        if row["_mass_kg"] < 0:
            raise ValueError(f"{source}: negative cargo mass")
        number(row, "delivery_time_s", source)
    if len(seen_boxes) != 80:
        raise ValueError(f"expected 80 unique delivered boxes, got {len(seen_boxes)}")
    return {"transport": transport, "relay": relay, "delivery": delivery,
            "summary": summary[0], "sha256": {name: digest(folder / name) for name in FILES}}


def dependency_blocks(q3: dict) -> tuple[tuple[str, ...], ...]:
    parent = {site: site for site in SERVICES}

    def root(site):
        while parent[site] != site:
            parent[site] = parent[parent[site]]
            site = parent[site]
        return site

    def union(sites):
        sites = list(sites)
        for site in sites[1:]:
            parent[root(site)] = root(sites[0])

    by_sortie = {row["sortie_id"]: row for row in q3["transport"]}
    for row in q3["transport"]:
        union(row["_sites"])
    for row in q3["relay"]:
        union(site for sid in row["_sorties"] for site in by_sortie[sid]["_sites"])
    blocks = {}
    for site in SERVICES:
        blocks.setdefault(root(site), []).append(site)
    return tuple(sorted((tuple(sorted(v)) for v in blocks.values()), key=lambda x: x[0]))


def intervals(q3: dict, blocks: tuple) -> tuple[list[tuple[int, int, float, float]], list[float]]:
    site_block = {site: i for i, block in enumerate(blocks) for site in block}
    jobs = []
    weights = [0.0] * len(blocks)
    for row in q3["transport"]:
        block = site_block[row["_sites"][0]]
        typ = row["uav_type"]
        start, end, charge = (float(row[x]) for x in ("start_time_s", "end_time_s", "charge_end_s"))
        release = float(row.get("uav_release_time_s") or end)
        jobs.extend(((block, KINDS.index("TUAV_" + typ), start, release),
                     (block, KINDS.index("TBAT_" + typ), start, charge)))
        weights[block] += end - start
    by_sortie = {row["sortie_id"]: row for row in q3["transport"]}
    for row in q3["relay"]:
        block = site_block[by_sortie[row["_sorties"][0]]["_sites"][0]]
        start = float(row["dispatch_time_s"])
        end = float(row["uav_release_time_s"])
        energy = float(row["energy_release_time_s"])
        jobs.extend(((block, 6, start, end), (block, 7, start, energy)))
        weights[block] += end - start
    return jobs, weights


def minimum_pool(intervals_: list[tuple[float, float]]) -> tuple[int, list[int]]:
    """Return exact pool size and a valid zero-based interval coloring."""
    busy, free, colors = [], [], []
    next_color = 0
    for original, (start, end) in sorted(enumerate(intervals_), key=lambda x: (x[1][0], x[1][1])):
        while busy and busy[0][0] <= start:
            _, color = heapq.heappop(busy)
            heapq.heappush(free, color)
        if free:
            color = heapq.heappop(free)
        else:
            color = next_color
            next_color += 1
        colors.append((original, color))
        heapq.heappush(busy, (end, color))
    by_original = [0] * len(intervals_)
    for original, color in colors:
        by_original[original] = color
    return next_color, by_original


def peak(intervals_: list[tuple[float, float]]) -> int:
    events = sorted([(a, 1) for a, _ in intervals_] + [(b, -1) for _, b in intervals_])
    active = answer = 0
    for _, change in events:
        active += change
        answer = max(answer, active)
    return answer


def cached_groups(jobs: list, weights: list[float]) -> dict[int, tuple[tuple[int, ...], float]]:
    cache = {}
    for mask in range(1, 1 << len(weights)):
        chosen = [i for i in range(len(weights)) if mask & (1 << i)]
        counts = tuple(peak([(a, b) for block, kind, a, b in jobs
                             if kind == r and block in chosen]) for r in range(8))
        cache[mask] = (counts, sum(weights[i] for i in chosen))
    return cache


def partitions(n: int, k: int):
    """Restricted-growth strings enumerate each unlabeled nonempty partition once."""
    if n < k:
        return
    groups = [1] + [0] * (n - 1)

    def visit(index: int, used: int):
        if index == n:
            if used == k:
                yield tuple(groups)
            return
        if used + n - index < k:
            return
        for label in range(1, min(used + 1, k) + 1):
            groups[index] = label
            yield from visit(index + 1, max(used, label))

    yield from visit(1, 1)


@dataclass(frozen=True)
class Result:
    groups: tuple[int, ...]
    counts: tuple[int, ...]
    cv: float
    shortage: int

    @property
    def total(self):
        return sum(self.counts)


def evaluate(groups: tuple[int, ...], cache: dict, k: int) -> Result:
    masks = [sum(1 << i for i, label in enumerate(groups) if label == g) for g in range(1, k + 1)]
    if any(not mask for mask in masks):
        raise ValueError("empty partition group")
    vectors = [cache[mask][0] for mask in masks]
    work = [cache[mask][1] for mask in masks]
    counts = tuple(sum(v[r] for v in vectors) for r in range(8))
    mean = sum(work) / k
    cv = math.sqrt(sum((w - mean) ** 2 for w in work) / k) / mean if mean else 0.0
    shortage = sum(max(0, counts[r] - INVENTORY[r]) for r in range(8))
    return Result(groups, counts, cv, shortage)


def dominates(a: Result, b: Result) -> bool:
    return all(x <= y for x, y in zip(a.counts, b.counts)) and a.cv <= b.cv + 1e-12 and (
        a.counts != b.counts or a.cv < b.cv - 1e-12)


def pareto(results: list[Result]) -> list[Result]:
    front = []
    for result in sorted(results, key=lambda x: (x.total, x.counts, x.cv)):
        if not any(dominates(other, result) for other in front):
            front = [other for other in front if not dominates(result, other)]
            front.append(result)
    return front


def stirling(n: int, k: int) -> int:
    table = [[0] * (k + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            table[i][j] = j * table[i - 1][j] + table[i - 1][j - 1]
    return table[n][k]


def describe(result: Result, blocks: tuple, cache: dict, baseline: tuple, label: str) -> dict:
    k = max(result.groups)
    rows = []
    for group in range(1, k + 1):
        mask = sum(1 << i for i, value in enumerate(result.groups) if value == group)
        rows.append({"group": f"G{group}",
                     "services": [site for i in range(len(blocks)) if mask & (1 << i) for site in blocks[i]],
                     "resources": dict(zip(KINDS, cache[mask][0])), "work_s": cache[mask][1]})
    return {"K": k, "scheme": label, "groups": rows, "counts": dict(zip(KINDS, result.counts)),
            "redundancy": dict(zip(KINDS, (result.counts[i] - baseline[i] for i in range(8)))),
            "shortage": dict(zip(KINDS, (max(0, result.counts[i] - INVENTORY[i]) for i in range(8)))),
            "inventory_surplus": dict(zip(KINDS, (max(0, INVENTORY[i] - result.counts[i]) for i in range(8)))),
            "total_resources": result.total, "total_shortage": result.shortage, "work_cv": result.cv}


def attach_group_details(description: dict, q3: dict) -> None:
    for group in description["groups"]:
        services = set(group["services"])
        sorties = {row["sortie_id"] for row in q3["transport"] if row["_sites"][0] in services}
        boxes = [row for row in q3["delivery"] if row["sortie_id"] in sorties]
        group["transport_sorties"] = len(sorties)
        group["relay_tasks"] = sum(row["_sorties"][0] in sorties for row in q3["relay"])
        group["boxes"] = len(boxes)
        group["mass_kg"] = sum(row["_mass_kg"] for row in boxes)


def solve(q3: dict) -> dict:
    blocks = dependency_blocks(q3)
    jobs, weights = intervals(q3, blocks)
    cache = cached_groups(jobs, weights)
    baseline = cache[(1 << len(blocks)) - 1][0]
    results = {"blocks": [list(block) for block in blocks],
               "baseline": dict(zip(KINDS, baseline)), "inventory": dict(zip(KINDS, INVENTORY)),
               "q3_sha256": q3["sha256"], "solutions": {}}
    for k in (2, 3):
        if len(blocks) < k:
            results["solutions"][str(k)] = {"status": "INFEASIBLE", "reason": "fewer dependency blocks than groups"}
            continue
        checked = 0
        main = None
        # Keep only the lowest CV for each integer resource vector. A dominated
        # configuration can never be Pareto, and ties need no duplicate row.
        by_resources = {}
        for grouping in partitions(len(blocks), k):
            current = evaluate(grouping, cache, k)
            checked += 1
            if main is None or (current.shortage, current.total, current.cv, current.groups) < (
                    main.shortage, main.total, main.cv, main.groups):
                main = current
            old = by_resources.get(current.counts)
            if old is None or (current.cv, current.groups) < (old.cv, old.groups):
                by_resources[current.counts] = current
        expected = stirling(len(blocks), k)
        if checked != expected:
            raise AssertionError(f"K={k}: partition count {checked} != {expected}")
        front = pareto(list(by_resources.values()))
        balanced = min(front, key=lambda x: (x.cv, x.shortage, x.total, x.groups))
        if any(x < b for x, b in zip(main.counts, baseline)):
            raise AssertionError("partition resource demand below unpartitioned baseline")
        results["solutions"][str(k)] = {
            "status": "EXACT", "partitions_checked": checked, "pareto_count": len(front),
            "shortage_first": describe(main, blocks, cache, baseline, "shortage_first"),
            "balance_first": describe(balanced, blocks, cache, baseline, "balance_first"),
            "pareto": [describe(x, blocks, cache, baseline, "pareto") for x in front],
        }
        for key in ("shortage_first", "balance_first"):
            attach_group_details(results["solutions"][str(k)][key], q3)
    return results


def validate_q4(q3: dict, result: dict) -> dict:
    checks = {"frozen_q3_accepted": True, "service_coverage": True,
              "transport_dependencies": True, "relay_dependencies": True,
              "resource_minima_verified": True, "inventory_gap_verified": True,
              "partition_enumeration_complete": True, "work_cv_verified": True}
    blocks = result["blocks"]
    jobs, _ = intervals(q3, tuple(tuple(block) for block in blocks))
    for k in (2, 3):
        item = result["solutions"][str(k)]
        if item["status"] != "EXACT":
            checks["partition_enumeration_complete"] &= len(blocks) < k
            continue
        checks["partition_enumeration_complete"] &= item["partitions_checked"] == stirling(len(blocks), k)
        for scheme in ("shortage_first", "balance_first"):
            solution = item[scheme]
            groups = solution["groups"]
            flat = [site for group in groups for site in group["services"]]
            checks["service_coverage"] &= len(groups) == k and sorted(flat) == list(SERVICES)
            service_group = {site: g for g, group in enumerate(groups) for site in group["services"]}
            for row in q3["transport"]:
                checks["transport_dependencies"] &= len({service_group[site] for site in row["_sites"]}) == 1
            by_sortie = {row["sortie_id"]: row for row in q3["transport"]}
            for row in q3["relay"]:
                sites = [site for sid in row["_sorties"] for site in by_sortie[sid]["_sites"]]
                checks["relay_dependencies"] &= len({service_group[site] for site in sites}) == 1
            site_block = {site: i for i, block in enumerate(blocks) for site in block}
            block_group = {site_block[site]: g for site, g in service_group.items()}
            for g, group in enumerate(groups):
                for r, kind in enumerate(KINDS):
                    actual = minimum_pool([(a, b) for block, category, a, b in jobs
                                           if category == r and block_group[block] == g])[0]
                    checks["resource_minima_verified"] &= actual == group["resources"][kind]
            for kind in KINDS:
                count = solution["counts"][kind]
                checks["inventory_gap_verified"] &= (
                    solution["shortage"][kind] == max(0, count - result["inventory"][kind])
                    and solution["redundancy"][kind] == count - result["baseline"][kind])
            work = [group["work_s"] for group in groups]
            mean = sum(work) / k
            actual_cv = math.sqrt(sum((w - mean) ** 2 for w in work) / k) / mean if mean else 0.0
            checks["work_cv_verified"] &= abs(actual_cv - solution["work_cv"]) <= 1e-10
    return {"all_pass": all(checks.values()), "checks": checks}


def main():
    q3 = load_final_q3()
    result = solve(q3)
    validation = validate_q4(q3, result)
    if not validation["all_pass"]:
        raise AssertionError(f"Q4 validation failed: {validation['checks']}")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "q4_result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "q4_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "q4_blocks.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("block", "services"))
        for i, block in enumerate(result["blocks"], 1):
            writer.writerow((f"B{i}", ">".join(block)))
    with (OUT / "Q4_partition_groups.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("K", "scheme", "group", "services", "boxes", "mass_kg",
                         "transport_sorties", "relay_tasks", "work_s", *KINDS))
        for item in result["solutions"].values():
            if item["status"] != "EXACT":
                continue
            for scheme in ("shortage_first", "balance_first"):
                for group in item[scheme]["groups"]:
                    writer.writerow((item[scheme]["K"], scheme, group["group"],
                                     ">".join(group["services"]), group["boxes"], group["mass_kg"],
                                     group["transport_sorties"], group["relay_tasks"], group["work_s"],
                                     *(group["resources"][kind] for kind in KINDS)))
    with (OUT / "Q4_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("K", "scheme", "total_resources", "total_redundancy",
                         "total_shortage", "work_cv", *KINDS))
        for item in result["solutions"].values():
            if item["status"] != "EXACT":
                continue
            for scheme in ("shortage_first", "balance_first"):
                d = item[scheme]
                writer.writerow((d["K"], scheme, d["total_resources"], sum(d["redundancy"].values()),
                                 d["total_shortage"], d["work_cv"],
                                 *(d["counts"][kind] for kind in KINDS)))
    for k in (2, 3):
        with (OUT / f"q4_pareto_K{k}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("K", "group_services", "total_resources", "total_shortage", "work_cv", *KINDS))
            item = result["solutions"][str(k)]
            if item["status"] == "EXACT":
                for solution in item["pareto"]:
                    writer.writerow((k, "|".join(">".join(g["services"]) for g in solution["groups"]),
                                     solution["total_resources"], solution["total_shortage"],
                                     solution["work_cv"], *(solution["counts"][kind] for kind in KINDS)))
    output_names = ("q4_result.json", "q4_validation.json", "q4_blocks.csv",
                    "Q4_partition_groups.csv", "Q4_comparison.csv",
                    "q4_pareto_K2.csv", "q4_pareto_K3.csv")
    manifest = {"q3_input_sha256": q3["sha256"],
                "outputs_sha256": {name: digest(OUT / name) for name in output_names},
                "blocks": len(result["blocks"]),
                "validation_all_pass": validation["all_pass"]}
    (OUT / "q4_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Q4 exact results: {target}")


if __name__ == "__main__":
    main()
