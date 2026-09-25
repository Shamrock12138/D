u"""Q3 Step5: Pattern 级通信缺口模板化。

对 Step4 筛选后的 compact patterns，从航段/节点直连缓存重建每个
pattern 的 gap 时间线，并建立唯一断连状态库。

关键设计:
  - 断连状态库按 (uav_type, phase, origin, destination, local_tau) 去重
  - 航段上的同一切片只存一次 (x,y,z,margin_db,terrain_blocked)
  - 节点固定位置只存 (x,y,z) + 通信状态
  - gap 映射表记录每个 gap 内每个采样点的 state_id
  - 主键从 task_id 改为 pattern_id
"""

import csv as _csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.q3.communication.direct_profile import DirectProfileCache
from src.q3.communication.link_budget import PARAMETER_PATH
from src.q3.communication.terrain_block import DEM_PATH

from src.q3.transport.compact_loader import (
    CompactPatternTemplate,
    load_q2_compact_artifacts,
    build_compact_pattern_templates,
)
from src.q3.transport.communication_summary import (
    required_segment_keys,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"

CANDIDATE_PATTERNS = DATA / "q3_compact_patterns.csv"
CANDIDATE_COUNTS = DATA / "q3_compact_pattern_counts.csv"

OUTAGE_STATES = DATA / "q3_outage_states.csv"
PATTERN_GAPS = DATA / "q3_pattern_comm_gaps.csv"
PATTERN_GAP_STATES = DATA / "q3_pattern_gap_states.csv"
GAP_MANIFEST = DATA / "q3_pattern_gap_manifest.json"

ROUTE_PARAMETERS = DATA / "route_parameter_all.csv"
NODE_PARAMETERS = DATA / "服务区数据.csv"
UAV_PARAMETERS = DATA / "运输无人机_机型参数.csv"


@dataclass(frozen=True)
class OutageStateKey:
    u"""唯一断连状态的联合键，按 (机型, 阶段, 起点, 终点, 段内时刻) 去重。"""
    uav_type: str
    phase: str
    origin: str
    destination: str
    local_tau: float

    def __hash__(self):
        raw = (
            self.uav_type + "\x00" + self.phase + "\x00"
            + self.origin + "\x00" + self.destination
        )
        raw += f"\x00{self.local_tau:.6f}"
        return hash(raw)

    def __eq__(self, other):
        if not isinstance(other, OutageStateKey):
            return False
        return (
            self.uav_type == other.uav_type
            and self.phase == other.phase
            and self.origin == other.origin
            and self.destination == other.destination
            and abs(self.local_tau - other.local_tau) < 1e-6
        )


@dataclass(frozen=True)
class ProfileSample:
    tau: float
    direct: bool
    margin_db: float
    state_key: OutageStateKey
    x: float
    y: float
    z: float
    phase: str
    node: Optional[str]


def _round_tau(t: float) -> float:
    return round(t, 6)


def _build_outage_state_library(
    cache: DirectProfileCache,
    verbose: bool = True,
):
    u"""遍历所有航段和节点，提取断连位置生成唯一状态库。"""
    state_rows: Dict[OutageStateKey, dict] = {}
    next_id = 1
    total_seg = 0
    total_node = 0

    for key, seg in cache.segments.items():
        uav_type, origin, dest = key
        for i in range(len(seg.times)):
            if seg.direct[i]:
                continue
            t = _round_tau(seg.times[i])
            phase = seg.phase[i]
            sk = OutageStateKey(
                uav_type=uav_type, phase=phase,
                origin=origin, destination=dest, local_tau=t,
            )
            if sk not in state_rows:
                sid = f"OS{next_id:06d}"
                next_id += 1
                state_rows[sk] = {
                    "state_id": sid,
                    "uav_type": uav_type,
                    "phase": phase,
                    "origin": origin,
                    "destination": dest,
                    "local_tau": t,
                    "x": seg.x[i],
                    "y": seg.y[i],
                    "z": seg.z[i],
                    "margin_db": seg.margin_db[i],
                    "terrain_blocked": int(seg.terrain_blocked[i]),
                }
                total_seg += 1

    for node_name, state in cache.node_states.items():
        if state.direct:
            continue
        node = cache.nodes[node_name]
        x, y, z = node["x"], node["y"], node["operation_height"]
        for phase in ("setup", "handover"):
            sk = OutageStateKey(
                uav_type="ANY", phase=phase,
                origin=node_name, destination=node_name, local_tau=0.0,
            )
            if sk not in state_rows:
                sid = f"OS{next_id:06d}"
                next_id += 1
                state_rows[sk] = {
                    "state_id": sid,
                    "uav_type": "ANY",
                    "phase": phase,
                    "origin": node_name,
                    "destination": node_name,
                    "local_tau": 0.0,
                    "x": x,
                    "y": y,
                    "z": z,
                    "margin_db": state.margin_db,
                    "terrain_blocked": int(state.terrain_blocked),
                }
                total_node += 1

    df = pd.DataFrame(list(state_rows.values()))
    state_map: Dict[OutageStateKey, str] = {
        sk: row["state_id"] for sk, row in state_rows.items()
    }

    if verbose:
        print(
            f"  唯一断连状态: {len(df)} (航段 {total_seg}, 节点 {total_node})",
            flush=True,
        )
    return state_map, df


def _assemble_pattern_profile_with_sources(
    template: CompactPatternTemplate,
    cache: DirectProfileCache,
):

    params = cache.uav_params[
        template.uav_type
    ]

    samples = []

    def _extend(items):
        if not items:
            return

        if (
            samples
            and math.isclose(
                samples[-1].tau,
                items[0].tau,
                abs_tol=1e-8,
            )
        ):
            samples.pop()

        samples.extend(items)

    time_s = 0.0

    duration = (
        params["setup_time"]
        + params["load_time_per_box"]
        * template.n_boxes
    )

    node = cache.nodes["O01"]
    state = cache.node_states["O01"]

    key = OutageStateKey(
        uav_type="ANY",
        phase="setup",
        origin="O01",
        destination="O01",
        local_tau=0.0,
    )

    points = cache.generator._sample_phase(
        time_s,
        duration,
        "setup",
        lambda ratio: (
            node["x"],
            node["y"],
            node["operation_height"],
        ),
        "O01",
    )

    _extend([
        ProfileSample(
            p.time,
            state.direct,
            state.margin_db,
            key,
            p.x,
            p.y,
            p.z,
            p.phase,
            p.node,
        )
        for p in points
    ])

    time_s += duration

    for origin, destination in zip(
        template.route,
        template.route[1:],
    ):

        seg = cache.segments[
            (
                template.uav_type,
                origin,
                destination,
            )
        ]

        _extend([
            ProfileSample(
                time_s + seg.times[i],
                seg.direct[i],
                seg.margin_db[i],

                OutageStateKey(
                    uav_type=
                        template.uav_type,

                    phase=
                        seg.phase[i],

                    origin=origin,

                    destination=
                        destination,

                    local_tau=
                        _round_tau(
                            seg.times[i]
                        ),
                ),

                seg.x[i],
                seg.y[i],
                seg.z[i],
                seg.phase[i],
                seg.node[i],
            )
            for i in range(
                len(seg.times)
            )
        ])

        time_s += seg.duration_s

        if destination != "O01":

            n_boxes_here = (
                template.service_counts[
                    destination
                ]
            )

            duration = (
                params["handover_time"]
                + params[
                    "handover_time_per_box"
                ]
                * n_boxes_here
            )

            node = cache.nodes[
                destination
            ]

            state = cache.node_states[
                destination
            ]

            key = OutageStateKey(
                uav_type="ANY",
                phase="handover",
                origin=destination,
                destination=destination,
                local_tau=0.0,
            )

            points = (
                cache.generator
                ._sample_phase(
                    time_s,
                    duration,
                    "handover",
                    lambda ratio: (
                        node["x"],
                        node["y"],
                        node[
                            "operation_height"
                        ],
                    ),
                    destination,
                )
            )

            _extend([
                ProfileSample(
                    p.time,
                    state.direct,
                    state.margin_db,
                    key,
                    p.x,
                    p.y,
                    p.z,
                    p.phase,
                    p.node,
                )
                for p in points
            ])

            time_s += duration

    if abs(
        time_s
        - template.duration_s
    ) >= 0.1:

        raise RuntimeError(
            f"{template.pattern_id}: "
            "gap profile duration mismatch"
        )

    return samples


def _extract_gaps_with_states(
    templates,
    cache,
    state_map,
    verbose=True,
    needs_relay_pids=None,
):
    """遍历 pattern，提取 gap + 逐采样 state 映射。"""
    gap_rows = []
    gap_state_rows = []
    total_gaps = 0
    n_direct = 0
    n_relay = 0
    pattern_gap_counts = {}
    gap_index = 0
    skipped = 0

    for ti, template in enumerate(templates):
        if verbose and (ti + 1) % 500 == 0:
            print(f"  处理 pattern: {ti + 1}/{len(templates)}", flush=True)

        try:
            source_samples = _assemble_pattern_profile_with_sources(
                template, cache,
            )
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"  ⚠ 跳过 {template.pattern_id}: {exc}", flush=True)
            skipped += 1
            continue

        has_outage = any(not sample.direct for sample in source_samples)
        if not has_outage:
            n_direct += 1
            continue

        n_relay += 1
        pattern_gap_counts[template.pattern_id] = 0

        in_gap = False
        gap_start = 0.0
        gap_samples = []
        gap_before = None
        previous = None

        def boundary_fields(prefix, sample):
            if sample is None:
                return {
                    f"{prefix}_tau": "", f"{prefix}_direct": "",
                    f"{prefix}_x": "", f"{prefix}_y": "", f"{prefix}_z": "",
                    f"{prefix}_phase": "", f"{prefix}_node": "",
                    f"{prefix}_margin_db": "",
                }
            return {
                f"{prefix}_tau": round(sample.tau, 3),
                f"{prefix}_direct": int(sample.direct),
                f"{prefix}_x": sample.x, f"{prefix}_y": sample.y,
                f"{prefix}_z": sample.z, f"{prefix}_phase": sample.phase,
                f"{prefix}_node": sample.node or "",
                f"{prefix}_margin_db": sample.margin_db,
            }

        def append_gap(gap_after):
            nonlocal gap_index, total_gaps, gap_samples
            gap_index += 1
            gap_id = f"PG{gap_index:06d}"
            total_gaps += 1
            pattern_gap_counts[template.pattern_id] += 1
            gap_end = gap_after.tau if gap_after is not None else source_samples[-1].tau
            coverage_start = gap_before.tau if gap_before is not None else gap_start
            coverage_end = gap_after.tau if gap_after is not None else gap_end
            row = {
                "pattern_id": template.pattern_id,
                "gap_id": gap_id,
                "gap_index": pattern_gap_counts[template.pattern_id],
                "tau_start": round(gap_start, 3),
                "tau_end": round(gap_end, 3),
                "duration_s": round(gap_end - gap_start, 3),
                "coverage_start": round(coverage_start, 3),
                "coverage_end": round(coverage_end, 3),
                "coverage_duration_s": round(coverage_end - coverage_start, 3),
                "n_outage_samples": len(gap_samples),
            }
            row.update(boundary_fields("before", gap_before))
            row.update(boundary_fields("after", gap_after))
            gap_rows.append(row)
            for gidx, sample in enumerate(gap_samples):
                if sample.state_key not in state_map:
                    raise AssertionError(
                        f"{template.pattern_id}/{gap_id} 的断连采样没有 state_id"
                    )
                gap_state_rows.append({
                    "pattern_id": template.pattern_id,
                    "gap_id": gap_id,
                    "sample_idx": gidx,
                    "tau": round(sample.tau, 3),
                    "state_id": state_map[sample.state_key],
                })
            gap_samples = []

        for sample in source_samples:
            if not sample.direct and not in_gap:
                in_gap = True
                gap_start = sample.tau
                gap_before = previous if previous is not None and previous.direct else None
                gap_samples = [sample]
            elif not sample.direct and in_gap:
                gap_samples.append(sample)
            elif sample.direct and in_gap:
                in_gap = False
                append_gap(sample)
            previous = sample

        if in_gap:
            append_gap(None)

    gaps_df = pd.DataFrame(gap_rows)
    gap_states_df = pd.DataFrame(gap_state_rows)

    stats = {
        "n_templates_total": len(templates),
        "n_skipped": skipped,
        "n_direct_tasks": n_direct,
        "n_relay_tasks": n_relay,
        "n_total_gaps": total_gaps,
        "n_unique_outage_states": len(state_map),
    }

    if verbose:
        print(f"\n{'─' * 40}", flush=True)
        print(f"  全直连 pattern: {n_direct}", flush=True)
        print(f"  有断连 pattern: {n_relay}", flush=True)
        print(f"  总 gap 数:   {total_gaps}", flush=True)
        if n_relay > 0:
            print(f"  avg gaps/pattern: {total_gaps / n_relay:.1f}", flush=True)
            if len(gaps_df) > 0:
                print(
                    f"  avg gap dur:   {gaps_df['duration_s'].mean():.1f}s",
                    flush=True,
                )

    return gaps_df, gap_states_df, stats


def _prune_outage_states(outage_df, gap_states_df):
    u"""仅保留 pattern gap 真正引用的状态，并重新连续编号。"""
    if gap_states_df.empty:
        return outage_df.iloc[0:0].copy(), gap_states_df.copy(), len(outage_df)
    if gap_states_df["state_id"].isna().any() or (gap_states_df["state_id"] == "").any():
        raise AssertionError("gap-state 映射存在空 state_id")
    used = set(gap_states_df["state_id"])
    pruned = outage_df[outage_df["state_id"].isin(used)].copy()
    missing = used - set(pruned["state_id"])
    if missing:
        raise AssertionError(f"gap-state 引用了不存在的状态: {sorted(missing)[:5]}")
    orphan_count = len(outage_df) - len(pruned)
    old_ids = pruned["state_id"].tolist()
    remap = {old: f"OS{index:06d}" for index, old in enumerate(old_ids, 1)}
    pruned["state_id"] = pruned["state_id"].map(remap)
    remapped_gaps = gap_states_df.copy()
    remapped_gaps["state_id"] = remapped_gaps["state_id"].map(remap)
    if remapped_gaps["state_id"].isna().any():
        raise AssertionError("状态压缩后出现空 state_id")
    return pruned.reset_index(drop=True), remapped_gaps, orphan_count


def _safe_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def extract_pattern_gap_templates(
    dt: float = 10.0,
    verbose: bool = True,
):

    classes, _, _ = (
        load_q2_compact_artifacts()
    )

    patterns = pd.read_csv(
        CANDIDATE_PATTERNS,
        encoding="utf-8-sig",
    )

    counts = pd.read_csv(
        CANDIDATE_COUNTS,
        encoding="utf-8-sig",
    )

    templates = (
        build_compact_pattern_templates(
            patterns,
            counts,
            classes,
        )
    )

    needs_relay_pids = set(
        patterns.loc[
            patterns["needs_relay"] == 1,
            "pattern_id",
        ].astype(str)
    )

    cache = DirectProfileCache(
        dt=dt
    )

    try:

        cache.build_nodes()

        cache.build_segments(
            required_segment_keys(
                templates
            ),
            verbose=verbose,
        )

        state_map, outage_states = (
            _build_outage_state_library(
                cache,
                verbose=verbose,
            )
        )

        gaps, gap_states, stats = (
            _extract_gaps_with_states(
                templates,
                cache,
                state_map,
                verbose=verbose,
                needs_relay_pids=
                    needs_relay_pids,
            )
        )

    finally:
        cache.close()

    outage_states, gap_states, orphan = (
        _prune_outage_states(
            outage_states,
            gap_states,
        )
    )

    stats["n_outage_states_before_prune"] = len(state_map)
    stats["n_orphan_states_pruned"] = orphan
    stats["n_unique_outage_states"] = len(outage_states)
    stats["dt_s"] = dt
    if verbose:
        print(
            f"  状态压缩: {len(state_map)} → {len(outage_states)} "
            f"(裁掉 {orphan})",
            flush=True,
        )

    _safe_csv(
        outage_states,
        OUTAGE_STATES,
    )

    _safe_csv(
        gaps,
        PATTERN_GAPS,
    )

    _safe_csv(
        gap_states,
        PATTERN_GAP_STATES,
    )

    print(f"\n输出: {OUTAGE_STATES.name} ({len(outage_states)} 行)", flush=True)
    print(f"输出: {PATTERN_GAPS.name} ({len(gaps)} 行)", flush=True)
    print(f"输出: {PATTERN_GAP_STATES.name} ({len(gap_states)} 行)", flush=True)

    return (
        outage_states,
        gaps,
        gap_states,
        stats,
    )


def extract_gap_templates(*args, **kwargs):
    return extract_pattern_gap_templates(
        dt=kwargs.get("dt", 10.0),
        verbose=kwargs.get(
            "verbose",
            True,
        ),
    )


if __name__ == "__main__":
    extract_pattern_gap_templates()