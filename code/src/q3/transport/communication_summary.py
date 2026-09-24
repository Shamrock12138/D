u"""由航段/节点直连缓存生成 Q3 全候选运输任务的通信摘要。"""

import csv
import hashlib
import json
import math
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy
import PIL
import openpyxl

from src.q3.communication.checker import check_direct_link
from src.q3.communication.direct_profile import (
    DirectProfileCache, DirectState, required_segment_keys,
)
from src.q3.communication.link_budget import PARAMETER_PATH
from src.q3.communication.terrain_block import DEM_PATH
from src.q3.trajectory_generator import load_box_services

from .candidate_loader import TransportTaskTemplate, load_candidate_tasks
from .relative_trajectory import generate_relative_trajectories

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
SUMMARY_PATH = DATA / "q3_transport_comm_summary.csv"
VALIDATION_PATH = DATA / "q3_direct_profile_validation.csv"
MANIFEST_PATH = DATA / "q3_transport_comm_manifest.json"


def _extend_samples(samples: List[Tuple[float, bool, float]], fragment):
    if not fragment:
        return
    if samples and math.isclose(samples[-1][0], fragment[0][0], abs_tol=1e-8):
        samples.pop()
    samples.extend(fragment)


def _static_fragment(cache: DirectProfileCache, node: str, start: float,
                     duration: float, phase: str):
    position = cache.nodes[node]
    points = cache.generator._sample_phase(
        start, duration, phase,
        lambda ratio: (position["x"], position["y"], position["operation_height"]),
        node,
    )
    state = cache.node_states[node]
    return [(point.time, state.direct, state.margin_db) for point in points]


def assemble_task_profile(template: TransportTaskTemplate, cache: DirectProfileCache,
                          box_services):
    """完全沿用 TrajectoryGenerator 的阶段采样/边界替换规则。"""
    params = cache.uav_params[template.uav_type]
    counts = Counter(box_services[box_id] for box_id in template.boxes)
    if set(counts) != set(template.visit_order):
        raise ValueError(f"{template.task_id} 停靠点与货箱服务区不一致")

    samples: List[Tuple[float, bool, float]] = []
    time_s = 0.0
    setup = params["setup_time"] + params["load_time_per_box"] * len(template.boxes)
    _extend_samples(samples, _static_fragment(cache, "O01", time_s, setup, "setup"))
    time_s += setup

    for origin, destination in zip(template.route, template.route[1:]):
        segment = cache.segments[(template.uav_type, origin, destination)]
        _extend_samples(samples, [
            (time_s + tau, direct, margin)
            for tau, direct, margin in zip(segment.times, segment.direct, segment.margin_db)
        ])
        time_s += segment.duration_s
        if destination != "O01":
            handover = (
                params["handover_time"]
                + params["handover_time_per_box"] * counts[destination]
            )
            _extend_samples(samples, _static_fragment(
                cache, destination, time_s, handover, "handover"
            ))
            time_s += handover

    if not samples or abs(time_s - template.duration_s) >= 0.1:
        raise ValueError(
            f"{template.task_id} 通信 profile 时长 {time_s:.6f}s "
            f"与 Q2 候选时长 {template.duration_s:.6f}s 不一致"
        )
    if not math.isclose(samples[0][0], 0.0, abs_tol=1e-8):
        raise ValueError(f"{template.task_id} 通信 profile 未从 τ=0 开始")
    if abs(samples[-1][0] - time_s) > 1e-8:
        raise ValueError(f"{template.task_id} 通信 profile 未覆盖至终点")
    if any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
        raise ValueError(f"{template.task_id} 通信 profile 时刻不严格递增")
    return samples


def summarize_profile(template: TransportTaskTemplate, samples):
    outage = sum(
        next_sample[0] - sample[0]
        for sample, next_sample in zip(samples, samples[1:])
        if not sample[1]
    )
    direct_time = template.duration_s - outage
    gap_starts = []
    gap_lengths = []
    active_start = None
    for time_s, direct, _ in samples:
        if not direct and active_start is None:
            active_start = time_s
            gap_starts.append(time_s)
        elif direct and active_start is not None:
            gap_lengths.append(time_s - active_start)
            active_start = None
    if active_start is not None:
        gap_lengths.append(template.duration_s - active_start)
    return {
        "task_id": template.task_id,
        "uav_type": template.uav_type,
        "n_stops": template.n_stops,
        "visit_order": ">".join(template.visit_order),
        "duration_s": template.duration_s,
        "energy_kWh": template.energy_kWh,
        "latest_start_s": template.latest_start_s if math.isfinite(template.latest_start_s) else "",
        "direct_time_s": direct_time,
        "outage_time_s": outage,
        "outage_ratio": outage / template.duration_s,
        "gap_count": len(gap_starts),
        "max_gap_s": max(gap_lengths, default=0.0),
        "min_margin_db": min(sample[2] for sample in samples),
        "needs_relay": int(any(not sample[1] for sample in samples)),
    }


def _validation_templates(templates: Sequence[TransportTaskTemplate], seed: int = 2026):
    groups = defaultdict(list)
    for template in templates:
        groups[(template.n_stops, template.uav_type)].append(template)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        group = groups[key]
        if len(group) < 5:
            raise ValueError(f"验证分组 {key} 不足 5 个候选任务")
        selected.extend(rng.sample(group, 5))
    if len(selected) != 30:
        raise ValueError("预期抽取 30 个验证任务")
    return selected


def validate_against_full_trajectories(templates, cache, box_services,
                                       path: Path = VALIDATION_PATH):
    """30 个任务：完整轨迹+原始 checker 与航段缓存逐时刻对账。"""
    selected = _validation_templates(templates)
    full = generate_relative_trajectories(
        selected, dt=cache.generator.dt, nodes=cache.nodes,
        route_params=cache.route_params, uav_params_all=cache.uav_params,
        box_services=box_services, strict=True, verbose=False,
    )
    rows = []
    for template, trajectory in zip(selected, full):
        cached = assemble_task_profile(template, cache, box_services)
        if len(cached) != trajectory.n_points:
            raise AssertionError(
                f"{template.task_id} 采样数不一致: {len(cached)} != {trajectory.n_points}"
            )
        max_time_diff = max(
            abs(sample[0] - tau)
            for sample, tau in zip(cached, trajectory.times)
        )
        if max_time_diff > 1e-7:
            raise AssertionError(f"{template.task_id} 采样时刻偏差 {max_time_diff:.3g}s")
        direct_full = [
            bool(check_direct_link(
                (x, y, z), cache.gateway, cache.terrain, cache.parameters
            )["direct"])
            for x, y, z in zip(trajectory.x, trajectory.y, trajectory.z)
        ]
        mismatches = sum(
            cached_point[1] != actual
            for cached_point, actual in zip(cached, direct_full)
        )
        if mismatches:
            raise AssertionError(f"{template.task_id} 直连状态不一致: {mismatches} 点")
        duration_diff = abs(trajectory.times[-1] - template.duration_s)
        if duration_diff >= 0.1:
            raise AssertionError(f"{template.task_id} 轨迹时长偏差 {duration_diff:.3g}s")
        rows.append({
            "task_id": template.task_id, "uav_type": template.uav_type,
            "n_stops": template.n_stops, "n_points": trajectory.n_points,
            "max_time_diff_s": max_time_diff, "duration_diff_s": duration_diff,
            "direct_mismatches": mismatches,
        })
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_communication_summary(dt: float = 10.0, verbose: bool = True):
    templates = load_candidate_tasks()
    box_services = load_box_services()
    cache = DirectProfileCache(dt=dt)
    try:
        cache.build_nodes()
        keys = required_segment_keys(templates)
        cache.build_segments(keys, verbose=verbose)
        fields = (
            "task_id", "uav_type", "n_stops", "visit_order", "duration_s",
            "energy_kWh", "latest_start_s", "direct_time_s", "outage_time_s",
            "outage_ratio", "gap_count", "max_gap_s", "min_margin_db", "needs_relay",
        )
        relay_count = 0
        with SUMMARY_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, template in enumerate(templates, 1):
                row = summarize_profile(
                    template, assemble_task_profile(template, cache, box_services)
                )
                writer.writerow(row)
                relay_count += row["needs_relay"]
                if verbose and index % 10000 == 0:
                    print(f"  候选任务通信摘要: {index}/{len(templates)}", flush=True)
        validation = validate_against_full_trajectories(
            templates, cache, box_services
        )
        input_paths = (
            DATA / "Q2_candidate_tasks.csv",
            DATA / "Q2_candidate_deliveries.csv",
            DATA / "物资需求.csv",
            DATA / "服务区数据.csv",
            DATA / "运输无人机_机型参数.csv",
            DATA / "route_parameter_all.csv",
            DEM_PATH, PARAMETER_PATH,
        )
        manifest = {
            "step": "Q3 candidate transport direct communication profiles",
            "run_command": "python code/Q3_step3.py",
            "runtime": {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "Pillow": PIL.__version__,
                "openpyxl": openpyxl.__version__,
            },
            "dt_s": dt,
            "task_count": len(templates),
            "segment_profile_count": len(cache.segments),
            "node_profile_count": len(cache.node_states),
            "needs_relay_count": relay_count,
            "validation_seed": 2026,
            "validation_tasks": len(validation),
            "validation_mismatches": sum(row["direct_mismatches"] for row in validation),
            "input_sha256": {path.name: _sha256(path) for path in input_paths},
            "output_sha256": {
                SUMMARY_PATH.name: _sha256(SUMMARY_PATH),
                VALIDATION_PATH.name: _sha256(VALIDATION_PATH),
            },
        }
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Q3 Step3: {len(templates)} 任务、{len(cache.segments)} 航段缓存、"
            f"{len(cache.node_states)} 节点状态；需中继 {relay_count} 任务；"
            f"抽检 {len(validation)} 任务全部一致。",
            flush=True,
        )
        return manifest
    finally:
        cache.close()


if __name__ == "__main__":
    run_communication_summary()
