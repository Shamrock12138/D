

import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Tuple

from .checker import DATA, STATUS_PATH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REQUIREMENT_PATH = DATA / "relay_requirement.json"
SUMMARY_PATH = DATA / "communication_gap_summary.csv"


def _point(row: dict) -> dict:

    return {
        "time": float(row["time"]),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "z": float(row["z"]),
        "phase": row["phase"],
        "direct": int(row["direct"]),
    }


def _load_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {"flight_id", "uav_id", "time", "x", "y", "z", "phase", "direct"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"通信状态文件缺少字段: {sorted(needed - set(reader.fieldnames or []))}")
        return list(reader)


def detect_gaps(rows: Iterable[dict]) -> Tuple[List[dict], List[dict]]:
    





    grouped = OrderedDict()
    for row in rows:
        flight_id = int(row["flight_id"])
        uav_id = str(row["uav_id"]).strip()
        if flight_id <= 0 or not uav_id:
            raise ValueError("通信状态包含无效架次编号或无人机编号")
        point = _point(row)
        if point["direct"] not in (0, 1):
            raise ValueError(f"架次 {flight_id} 的 direct 只能为 0 或 1")
        if not all(math.isfinite(point[k]) for k in ("time", "x", "y", "z")):
            raise ValueError(f"架次 {flight_id} 包含非有限的时间或坐标")
        group = grouped.setdefault(flight_id, {"uav_id": uav_id, "points": []})
        if group["uav_id"] != uav_id:
            raise ValueError(f"架次 {flight_id} 对应了多个无人机编号")
        group["points"].append(point)

    if not grouped:
        raise ValueError("通信状态为空，无法提取断连时段")

    requirements = []
    summaries = []
    for flight_id, group in grouped.items():
        points = sorted(group["points"], key=lambda point: point["time"])
        if any(b["time"] <= a["time"] for a, b in zip(points, points[1:])):
            raise ValueError(f"架次 {flight_id} 存在重复或非递增采样时刻")
        n_gaps = 0
        n_unavailable = 0
        coverage_s = 0.0
        index = 0
        while index < len(points):
            if points[index]["direct"] == 1:
                index += 1
                continue
            start_idx = index
            while index < len(points) and points[index]["direct"] == 0:
                index += 1
            end_idx = index - 1
            n_gaps += 1
            n_unavailable += end_idx - start_idx + 1
            before = points[start_idx - 1] if start_idx > 0 else None
            after = points[index] if index < len(points) else None
            coverage_start = before["time"] if before else points[start_idx]["time"]
            coverage_end = after["time"] if after else points[end_idx]["time"]
            coverage_s += coverage_end - coverage_start
            requirements.append({
                "requirement_id": f"F{flight_id:03d}_G{n_gaps:02d}",
                "flight_id": flight_id,
                "uav_id": group["uav_id"],
                "gap_start": points[start_idx]["time"],
                "gap_end": points[end_idx]["time"],
                "coverage_start": coverage_start,
                "coverage_end": coverage_end,
                "unavailable_samples": end_idx - start_idx + 1,
                "trajectory_points": points[start_idx:index],
                "boundary_points": {"before": before, "after": after},
            })

        summaries.append({
            "flight_id": flight_id,
            "uav_id": group["uav_id"],
            "first_time": points[0]["time"],
            "last_time": points[-1]["time"],
            "sample_count": len(points),
            "unavailable_samples": n_unavailable,
            "gap_count": n_gaps,
            "sampled_direct_continuity": int(n_gaps == 0),
            "relay_coverage_window_s": coverage_s,
        })
    return requirements, summaries


def run_gap_detector(
    status_path: Path = STATUS_PATH,
    requirement_path: Path = REQUIREMENT_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> Tuple[List[dict], List[dict]]:
    requirements, summaries = detect_gaps(_load_rows(status_path))
    with requirement_path.open("w", encoding="utf-8") as stream:
        json.dump(requirements, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return requirements, summaries


if __name__ == "__main__":
    gaps, flights = run_gap_detector()
    print(f"通信断连区间: {len(gaps)}，受影响架次: {sum(x['gap_count'] > 0 for x in flights)}/{len(flights)}")
    print(f"中继需求: {REQUIREMENT_PATH.relative_to(DATA.parent)}")
    print(f"架次汇总: {SUMMARY_PATH.relative_to(DATA.parent)}")
