u"""由航段/节点直连缓存生成 Q3 全量 compact pattern 的通信摘要。"""

import csv as _csv
import hashlib
import json
import math
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy
import pandas as pd
import PIL
import openpyxl

from src.q3.communication.direct_profile import (
    DirectProfileCache, required_segment_keys,
)
from src.q3.communication.link_budget import PARAMETER_PATH
from src.q3.communication.terrain_block import DEM_PATH

from .compact_loader import (
    CompactPatternTemplate,
    load_all_compact_pattern_templates,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
SUMMARY_PATH = DATA / "q3_pattern_comm_summary.csv"
VALIDATION_PATH = DATA / "q3_pattern_direct_profile_validation.csv"
MANIFEST_PATH = DATA / "q3_pattern_comm_manifest.json"


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


def assemble_pattern_profile(
    template: CompactPatternTemplate,
    cache: DirectProfileCache,
):
    """根据 compact pattern 重建完整相对通信 profile。"""

    params = cache.uav_params[template.uav_type]

    samples = []
    time_s = 0.0

    setup_duration = (
        params["setup_time"]
        + params["load_time_per_box"] * template.n_boxes
    )

    _extend_samples(
        samples,
        _static_fragment(
            cache,
            "O01",
            time_s,
            setup_duration,
            "setup",
        ),
    )

    time_s += setup_duration

    for origin, destination in zip(
        template.route,
        template.route[1:],
    ):

        segment = cache.segments[
            (
                template.uav_type,
                origin,
                destination,
            )
        ]

        _extend_samples(
            samples,
            [
                (
                    time_s + tau,
                    direct,
                    margin,
                )
                for tau, direct, margin
                in zip(
                    segment.times,
                    segment.direct,
                    segment.margin_db,
                )
            ],
        )

        time_s += segment.duration_s

        if destination != "O01":

            n_boxes_here = template.service_counts[
                destination
            ]

            handover_duration = (
                params["handover_time"]
                + params["handover_time_per_box"]
                * n_boxes_here
            )

            _extend_samples(
                samples,
                _static_fragment(
                    cache,
                    destination,
                    time_s,
                    handover_duration,
                    "handover",
                ),
            )

            time_s += handover_duration

    if not samples:
        raise RuntimeError(
            f"{template.pattern_id}: empty profile"
        )

    if abs(time_s - template.duration_s) >= 0.1:
        raise RuntimeError(
            f"{template.pattern_id}: "
            f"profile duration={time_s:.6f}, "
            f"Q2 duration={template.duration_s:.6f}"
        )

    return samples


def summarize_pattern_profile(
    template: CompactPatternTemplate,
    samples,
):

    outage_time = sum(
        b[0] - a[0]
        for a, b in zip(
            samples,
            samples[1:],
        )
        if not a[1]
    )

    gap_starts = []
    gap_lengths = []

    active_start = None

    for time_s, direct, _ in samples:

        if not direct and active_start is None:
            active_start = time_s
            gap_starts.append(time_s)

        elif direct and active_start is not None:
            gap_lengths.append(
                time_s - active_start
            )
            active_start = None

    if active_start is not None:
        gap_lengths.append(
            template.duration_s - active_start
        )

    return {
        "pattern_id": template.pattern_id,
        "uav_type": template.uav_type,
        "n_stops": template.n_stops,
        "visit_order": ">".join(
            template.visit_order
        ),

        "n_boxes": template.n_boxes,

        "duration_s": template.duration_s,
        "energy_kWh": template.energy_kWh,
        "latest_start_s": (
            template.latest_start_s
            if math.isfinite(template.latest_start_s)
            else ""
        ),

        "direct_time_s":
            template.duration_s - outage_time,

        "outage_time_s": outage_time,

        "outage_ratio":
            outage_time / template.duration_s,

        "gap_count": len(gap_starts),

        "max_gap_s":
            max(gap_lengths, default=0.0),

        "min_margin_db":
            min(x[2] for x in samples),

        "needs_relay":
            int(any(not x[1] for x in samples)),
    }


def _validation_templates(templates: Sequence[CompactPatternTemplate], seed: int = 2026):
    groups = defaultdict(list)
    for template in templates:
        groups[(template.n_stops, template.uav_type)].append(template)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        group = groups[key]
        if len(group) < 5:
            raise ValueError(f"验证分组 {key} 不足 5 个候选 pattern")
        selected.extend(rng.sample(group, 5))
    if len(selected) != 30:
        raise ValueError("预期抽取 30 个验证 pattern")
    return selected


def validate_pattern_profiles(templates, cache, path=None):
    """30 个 pattern：全轨迹 direct check 与航段缓存逐时刻对账。"""
    out_path = path if path is not None else VALIDATION_PATH
    selected = _validation_templates(templates)
    from src.q3.communication.checker import check_direct_link
    from src.q3.communication.link_budget import load_direct_parameters
    from src.q3.communication.terrain_block import DemTerrain
    from src.q3.trajectory_generator import TrajectoryGenerator

    parameters = load_direct_parameters()
    terrain = DemTerrain()
    gateway = (cache.nodes["O01"]["x"], cache.nodes["O01"]["y"],
               cache.nodes["O01"]["h"] + parameters.gateway_height_m)

    rows = []
    try:
        for template in selected:
            params = cache.uav_params[template.uav_type]
            gen = TrajectoryGenerator(cache.nodes, cache.route_params, dt=cache.generator.dt)
            services_list = []
            for service, count in sorted(template.service_counts.items()):
                services_list.extend([service] * count)

            trajectory = gen.generate_relative(
                template.uav_type, list(template.route),
                services_list, params, strict=False, verbose=False,
            )

            cached = assemble_pattern_profile(template, cache)

            if len(cached) != trajectory.n_points:
                raise AssertionError(
                    f"{template.pattern_id} 采样数不一致: {len(cached)} != {trajectory.n_points}"
                )

            max_time_diff = max(
                abs(sample[0] - tau)
                for sample, tau in zip(cached, trajectory.times)
            )
            if max_time_diff > 1e-6:
                raise AssertionError(f"{template.pattern_id} 采样时刻偏差 {max_time_diff:.3g}s")

            direct_full = [
                bool(check_direct_link(
                    (x, y, z), gateway, terrain, parameters
                )["direct"])
                for x, y, z in zip(trajectory.x, trajectory.y, trajectory.z)
            ]
            mismatches = sum(
                cached_point[1] != actual
                for cached_point, actual in zip(cached, direct_full)
            )
            if mismatches:
                raise AssertionError(f"{template.pattern_id} 直连状态不一致: {mismatches} 点")

            rows.append({
                "pattern_id": template.pattern_id, "uav_type": template.uav_type,
                "n_stops": template.n_stops, "n_points": trajectory.n_points,
                "max_time_diff_s": max_time_diff,
                "direct_mismatches": mismatches,
            })
    finally:
        terrain.close()

    if rows:
        with out_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = _csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return rows


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_pattern_communication_summary(
    dt: float = 10.0,
    verbose: bool = True,
):

    classes, patterns, counts, templates = (
        load_all_compact_pattern_templates()
    )

    cache = DirectProfileCache(dt=dt)

    try:
        cache.build_nodes()

        keys = required_segment_keys(templates)

        cache.build_segments(
            keys,
            verbose=verbose,
        )

        rows = []

        for index, template in enumerate(
            templates,
            start=1,
        ):

            samples = assemble_pattern_profile(
                template,
                cache,
            )

            rows.append(
                summarize_pattern_profile(
                    template,
                    samples,
                )
            )

            if verbose and index % 10000 == 0:
                print(
                    f"Pattern communication: "
                    f"{index}/{len(templates)}",
                    flush=True,
                )

        result = pd.DataFrame(rows)

        if set(result["pattern_id"]) != set(
            patterns["pattern_id"].astype(str)
        ):
            raise AssertionError(
                "通信摘要未覆盖全部 Q2 compact patterns"
            )

        result.to_csv(
            SUMMARY_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Pattern 数: {len(result)}"
        )

        print(
            "需中继 Pattern: "
            f"{int(result['needs_relay'].sum())}"
        )

        return result

    finally:
        cache.close()


def run_communication_summary(
    dt: float = 10.0,
    verbose: bool = True,
):
    return run_pattern_communication_summary(
        dt=dt,
        verbose=verbose,
    )


if __name__ == "__main__":
    run_pattern_communication_summary()