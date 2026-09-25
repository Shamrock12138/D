u"""Q3 Step5: 通信缺口模板化。

对 Step4 筛选后的候选任务，从 Step3 的航段/节点直连缓存重建每个
任务的 gap 时间线，并建立唯一断连状态库（避免后续中继计算重复进行
DEM/LOS 判断）。

关键设计:
  - 断连状态库按 (uav_type, phase, origin, destination, local_tau) 去重
  - 航段上的同一切片只存一次 (x,y,z,margin_db,terrain_blocked)
  - 节点固定位置只存 (x,y,z) + 通信状态
  - gap 映射表记录每个 gap 内每个采样点的 state_id

输出:
  q3_outage_states.csv   — 唯一断连空间状态
  q3_task_comm_gaps.csv  — 任务级 gap 摘要
  q3_task_gap_states.csv — gap → state_id 逐采样映射
  q3_comm_gap_manifest.json
"""

import csv as _csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.q3.communication.direct_profile import DirectProfileCache
from src.q3.communication.link_budget import PARAMETER_PATH
from src.q3.communication.terrain_block import DEM_PATH
from src.q3.trajectory_generator import load_box_services
from src.q3.transport.candidate_loader import (
    TransportTaskTemplate, load_candidate_tasks, load_box_deadlines,
)
from src.q3.transport.communication_summary import required_segment_keys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"

CANDIDATE_TASKS = DATA / "q3_candidate_tasks.csv"
CANDIDATE_DELIVERIES = DATA / "q3_candidate_deliveries.csv"

OUTAGE_STATES = DATA / "q3_outage_states.csv"
TASK_GAPS = DATA / "q3_task_comm_gaps.csv"
TASK_GAP_STATES = DATA / "q3_task_gap_states.csv"
GAP_MANIFEST = DATA / "q3_comm_gap_manifest.json"
STEP3_MANIFEST = DATA / "q3_transport_comm_manifest.json"
STEP4_MANIFEST = DATA / "q3_candidate_filter_manifest.json"
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


def _assemble_profile_with_sources(
    template: TransportTaskTemplate,
    cache: DirectProfileCache,
    box_services: dict,
) -> List[ProfileSample]:
    u"""完全对等 assemble_task_profile，同时记录每个采样点的 state_key。

    重要: 必须与 communication_summary.assemble_task_profile 使用完全相同的
    _extend_samples 去重逻辑、_sample_phase 采样规则、以及阶段时长参数。
    """
    params = cache.uav_params[template.uav_type]
    counts = Counter(box_services[box_id] for box_id in template.boxes)
    if set(counts) != set(template.visit_order):
        raise ValueError(f"{template.task_id} 停靠点与货箱服务区不一致")

    source_samples: List[ProfileSample] = []

    def _extend(new_items: List[ProfileSample]):
        if not new_items:
            return
        if source_samples and math.isclose(
            source_samples[-1].tau, new_items[0].tau, abs_tol=1e-8,
        ):
            source_samples.pop()
        source_samples.extend(new_items)

    uav_name = template.uav_type
    time_s = 0.0

    # --- setup at O01 ---
    setup_dur = params["setup_time"] + params["load_time_per_box"] * len(template.boxes)
    node_O01 = cache.nodes["O01"]
    state_O01 = cache.node_states["O01"]
    setup_sk = OutageStateKey(
        uav_type="ANY", phase="setup",
        origin="O01", destination="O01", local_tau=0.0,
    )
    setup_points = cache.generator._sample_phase(
        time_s, setup_dur, "setup",
        lambda ratio: (node_O01["x"], node_O01["y"], node_O01["operation_height"]),
        "O01",
    )
    _extend([
        ProfileSample(
            p.time, state_O01.direct, state_O01.margin_db, setup_sk,
            p.x, p.y, p.z, p.phase, p.node,
        )
        for p in setup_points
    ])
    time_s += setup_dur

    # --- segments ---
    seg_cache = cache.segments
    for origin, dest in zip(template.route, template.route[1:]):
        seg_key = (uav_name, origin, dest)
        seg = seg_cache[seg_key]
        _extend([
            ProfileSample(
                time_s + seg.times[i], seg.direct[i], seg.margin_db[i],
                OutageStateKey(
                    uav_type=uav_name,
                    phase=seg.phase[i],
                    origin=origin, destination=dest,
                    local_tau=_round_tau(seg.times[i]),
                ),
                seg.x[i], seg.y[i], seg.z[i], seg.phase[i], seg.node[i],
            )
            for i in range(len(seg.times))
        ])
        time_s += seg.duration_s

        # --- handover at destination ---
        if dest != "O01":
            hov_dur = (
                params["handover_time"]
                + params["handover_time_per_box"] * counts[dest]
            )
            dest_node = cache.nodes[dest]
            dest_state = cache.node_states[dest]
            hov_sk = OutageStateKey(
                uav_type="ANY", phase="handover",
                origin=dest, destination=dest, local_tau=0.0,
            )
            hov_points = cache.generator._sample_phase(
                time_s, hov_dur, "handover",
                lambda ratio: (dest_node["x"], dest_node["y"],
                               dest_node["operation_height"]),
                dest,
            )
            _extend([
                ProfileSample(
                    p.time, dest_state.direct, dest_state.margin_db, hov_sk,
                    p.x, p.y, p.z, p.phase, p.node,
                )
                for p in hov_points
            ])
            time_s += hov_dur

    return source_samples


def _extract_gaps_with_states(
    templates: List[TransportTaskTemplate],
    cache: DirectProfileCache,
    state_map: Dict[OutageStateKey, str],
    box_services: dict = None,
    verbose: bool = True,
    needs_relay_tids: Set[str] = None,
):
    u"""遍历任务，提取 gap + 逐采样 state 映射。"""
    gap_rows: List[dict] = []
    gap_state_rows: List[dict] = []
    total_gaps = 0
    n_direct = 0
    n_relay = 0
    task_gap_counts: Dict[str, int] = {}
    gap_index = 0
    skipped = 0

    if needs_relay_tids is None:
        needs_relay_tids = set()
        with CANDIDATE_TASKS.open("r", encoding="utf-8-sig", newline="") as f:
            for row in _csv.DictReader(f):
                if row.get("needs_relay", "0") == "1":
                    needs_relay_tids.add(row["task_id"].strip())

    for ti, template in enumerate(templates):
        if verbose and (ti + 1) % 500 == 0:
            print(f"  处理任务: {ti + 1}/{len(templates)}", flush=True)

        try:
            source_samples = _assemble_profile_with_sources(
                template, cache, box_services,
            )
        except (ValueError, KeyError) as exc:
            print(f"  ⚠ 跳过 {template.task_id}: {exc}", flush=True)
            skipped += 1
            continue

        has_outage = any(not sample.direct for sample in source_samples)
        if not has_outage:
            n_direct += 1
            continue

        n_relay += 1
        task_gap_counts[template.task_id] = 0

        in_gap = False
        gap_start = 0.0
        gap_samples: List[ProfileSample] = []
        gap_before: Optional[ProfileSample] = None
        previous: Optional[ProfileSample] = None

        def boundary_fields(prefix: str, sample: Optional[ProfileSample]):
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

        def append_gap(gap_after: Optional[ProfileSample]):
            nonlocal gap_index, total_gaps, gap_samples
            gap_index += 1
            gap_id = f"G{gap_index:06d}"
            total_gaps += 1
            task_gap_counts[template.task_id] += 1
            gap_end = gap_after.tau if gap_after is not None else source_samples[-1].tau
            coverage_start = gap_before.tau if gap_before is not None else gap_start
            coverage_end = gap_after.tau if gap_after is not None else gap_end
            row = {
                "task_id": template.task_id, "gap_id": gap_id,
                "gap_index": task_gap_counts[template.task_id],
                "tau_start": round(gap_start, 3), "tau_end": round(gap_end, 3),
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
                        f"{template.task_id}/{gap_id} 的断连采样没有 state_id"
                    )
                gap_state_rows.append({
                    "task_id": template.task_id, "gap_id": gap_id,
                    "sample_idx": gidx, "tau": round(sample.tau, 3),
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

    step4_direct_count = len(templates) - len(needs_relay_tids)
    step4_relay_count = len(needs_relay_tids)
    actual_found_tids = set(task_gap_counts.keys())
    false_positives = needs_relay_tids - actual_found_tids
    false_negatives = actual_found_tids - needs_relay_tids

    stats = {
        "n_templates_total": len(templates),
        "n_skipped": skipped,
        "n_direct_tasks": n_direct,
        "n_relay_tasks": n_relay,
        "n_total_gaps": total_gaps,
        "n_unique_outage_states": len(state_map),
        "step4_direct_label": step4_direct_count,
        "step4_relay_label": step4_relay_count,
        "step5_direct_found": n_direct,
        "step5_relay_found": n_relay,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "direct_match": (
            "PASS" if n_direct == step4_direct_count
            else f"MISMATCH ({n_direct} vs {step4_direct_count})"
        ),
        "relay_match": (
            "PASS" if n_relay == step4_relay_count
            else f"MISMATCH ({n_relay} vs {step4_relay_count})"
        ),
    }

    if verbose:
        print(f"\n{'─' * 40}", flush=True)
        print(f"  全直连任务: {n_direct}", flush=True)
        print(f"  有断连任务: {n_relay}", flush=True)
        print(f"  总 gap 数:   {total_gaps}", flush=True)
        if n_relay > 0:
            print(f"  avg gaps/task: {total_gaps / n_relay:.1f}", flush=True)
            if len(gaps_df) > 0:
                print(
                    f"  avg gap dur:   {gaps_df['duration_s'].mean():.1f}s",
                    flush=True,
                )
        print(f"\n  Step4 direct 标签: {step4_direct_count}", flush=True)
        print(f"  Step5 direct 实测: {n_direct}", flush=True)
        print(f"  direct 一致: {stats['direct_match']}", flush=True)
        print(f"\n  Step4 relay 标签:  {step4_relay_count}", flush=True)
        print(f"  Step5 relay 实测:  {n_relay}", flush=True)
        print(f"  relay 一致: {stats['relay_match']}", flush=True)
        if false_positives:
            print(
                f"  ⚠ false positive (label=1, no gap): {len(false_positives)}",
                flush=True,
            )
        if false_negatives:
            print(
                f"  ⚠ false negative (label=0, has gap): {len(false_negatives)}",
                flush=True,
            )

    return gaps_df, gap_states_df, stats


def _prune_outage_states(outage_df: pd.DataFrame, gap_states_df: pd.DataFrame):
    u"""仅保留任务 gap 真正引用的状态，并重新连续编号。"""
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


def save_gap_outputs(
    outage_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    gap_states_df: pd.DataFrame,
    stats: dict,
) -> None:
    _safe_csv(outage_df, OUTAGE_STATES)
    print(f"\n输出: {OUTAGE_STATES.name} ({len(outage_df)} 行)", flush=True)

    _safe_csv(gaps_df, TASK_GAPS)
    print(f"输出: {TASK_GAPS.name} ({len(gaps_df)} 行)", flush=True)

    if len(gap_states_df) > 0:
        _safe_csv(gap_states_df, TASK_GAP_STATES)
        print(
            f"输出: {TASK_GAP_STATES.name} ({len(gap_states_df)} 行)",
            flush=True,
        )

    _safe_csv(
        pd.DataFrame(
            [{"metric": k, "value": v} for k, v in stats.items()]
        ),
        DATA / "q3_comm_gap_summary.csv",
    )
    print(f"输出: q3_comm_gap_summary.csv", flush=True)

    manifest = {
        "step": "Q3 Step5: 通信缺口模板化",
        "n_templates": stats["n_templates_total"],
        "n_direct_tasks": stats["n_direct_tasks"],
        "n_relay_tasks": stats["n_relay_tasks"],
        "n_total_gaps": stats["n_total_gaps"],
        "n_unique_outage_states": stats["n_unique_outage_states"],
        "n_gap_state_mappings": len(gap_states_df),
        "dt_s": stats["dt_s"],
        "gap_boundary_policy": (
            "tau_start/tau_end retain sampled outage semantics; relay service uses "
            "[coverage_start, coverage_end], including the preceding direct sample "
            "when available; boundary position/phase/margin are retained for refinement"
        ),
        "phase_source": "TrajectoryGenerator sample phase stored in SegmentDirectProfile",
        "outage_states_before_prune": stats["n_outage_states_before_prune"],
        "orphan_states_pruned": stats["n_orphan_states_pruned"],
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                CANDIDATE_TASKS, CANDIDATE_DELIVERIES,
                STEP3_MANIFEST, STEP4_MANIFEST,
                ROUTE_PARAMETERS, NODE_PARAMETERS, UAV_PARAMETERS,
                DEM_PATH, PARAMETER_PATH,
            )
            if path.exists()
        },
        "output_sha256": {},
    }
    for path in (OUTAGE_STATES, TASK_GAPS, TASK_GAP_STATES):
        if path.exists():
            manifest["output_sha256"][path.name] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
            )

    tmp = GAP_MANIFEST.with_suffix(GAP_MANIFEST.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    tmp.replace(GAP_MANIFEST)
    print(f"输出: {GAP_MANIFEST.name}", flush=True)


def extract_gap_templates(
    tasks_path: Path = CANDIDATE_TASKS,
    deliveries_path: Path = CANDIDATE_DELIVERIES,
    dt: float = 10.0,
    verbose: bool = True,
):
    u"""主入口：加载任务 → 构建缓存 → 提取 gap + 输出。"""
    print("=" * 60, flush=True)
    print("Q3 Step5: 通信缺口模板化", flush=True)
    print("=" * 60, flush=True)

    templates = load_candidate_tasks(tasks_path, deliveries_path)
    print(f"载入候选任务模板: {len(templates)}", flush=True)
    box_services = load_box_services()
    print(f"载入货箱-服务区映射: {len(box_services)} 箱", flush=True)

    print("\n构建直连通信缓存 (复用 Step3 口径)...", flush=True)
    cache = DirectProfileCache(dt=dt)
    try:
        cache.build_nodes()
        seg_keys = required_segment_keys(templates)
        cache.build_segments(keys=seg_keys, verbose=verbose)
        state_map, outage_states_df = _build_outage_state_library(
            cache, verbose=verbose,
        )
        gaps_df, gap_states_df, stats = _extract_gaps_with_states(
            templates, cache, state_map, box_services=box_services, verbose=verbose,
        )
    finally:
        cache.close()

    outage_states_df, gap_states_df, orphan_count = _prune_outage_states(
        outage_states_df, gap_states_df,
    )
    stats["n_outage_states_before_prune"] = len(state_map)
    stats["n_orphan_states_pruned"] = orphan_count
    stats["n_unique_outage_states"] = len(outage_states_df)
    stats["dt_s"] = dt
    if verbose:
        print(
            f"  状态压缩: {len(state_map)} → {len(outage_states_df)} "
            f"(裁掉 {orphan_count})",
            flush=True,
        )

    print(f"\n{'=' * 60}", flush=True)
    print("Step5 完成", flush=True)

    return outage_states_df, gaps_df, gap_states_df, stats


if __name__ == "__main__":
    extract_gap_templates()
