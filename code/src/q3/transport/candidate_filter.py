u"""Q3 运输 + 通信 + Q2 Anchor（可选 seed）联合候选筛选。

按 (class_id, UAV type) 分组，运输维度 + 通信维度 + 可选 Anchor seed 压缩。
Q2 Anchor 缺失/过期/未证明最优均不阻止 Q3 求解，仅降级为警告。
"""

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from src.q2.compact_classes import select_compact_patterns

from .compact_loader import load_q2_compact_artifacts

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"

TOP_K_PER_DIM = 8

COMM_SUMMARY = DATA / "q3_pattern_comm_summary.csv"

PATTERNS_OUT = DATA / "q3_compact_patterns.csv"
COUNTS_OUT = DATA / "q3_compact_pattern_counts.csv"
MANIFEST_OUT = DATA / "q3_compact_filter_manifest.json"

ANCHOR_OBJECTIVES = ("F1", "Cmax", "E", "N")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_q2_anchor_pattern_ids():
    u"""读取 Q2 Anchor 中使用的 pattern_id。

    Q2 Anchor 文件缺失/过期/未证明最优均不阻止 Q3：
    缺 anchor → 警告并返回空集。
    候选 manifest 缺失 → 警告并跳过 manifest 校验。
    """

    candidate_manifest_path = DATA / "Q2_compact_candidates_manifest.json"

    if not candidate_manifest_path.exists():
        print("  ⚠ Q2 compact candidates manifest 不存在，跳过 Q2 Anchor seed", flush=True)
        return set()

    try:
        candidate_sha = _sha256(candidate_manifest_path)
    except OSError:
        print("  ⚠ 无法读取 Q2 manifest，跳过 Q2 Anchor seed", flush=True)
        return set()

    pattern_ids = set()

    for objective in ANCHOR_OBJECTIVES:
        selected_path = DATA / f"Q2_anchor_{objective}_selected.csv"
        manifest_path = DATA / f"Q2_anchor_{objective}_manifest.json"

        if not selected_path.exists():
            print(f"  ⚠ 缺少 Q2/{objective} anchor csv，跳过该 anchor", flush=True)
            continue

        if not manifest_path.exists():
            print(f"  ⚠ 缺少 Q2/{objective} anchor manifest，跳过目标校验", flush=True)
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                print(f"  ⚠ Q2/{objective} anchor manifest 不可读，不使用该 anchor", flush=True)
                continue
            else:
                stored = manifest.get("candidate_manifest_sha256", "")
                if stored and stored != candidate_sha:
                    print(
                        f"  ⚠ Q2/{objective} anchor 已过期，"
                        "不使用该 seed",
                        flush=True,
                    )
                    continue
                if not manifest.get("anchor_ready", False):
                    print(
                        f"  ⚠ Q2/{objective} anchor 未证明全局最优，"
                        "仍保留 seed",
                        flush=True,
                    )

        try:
            selected = pd.read_csv(selected_path, encoding="utf-8-sig")
        except Exception:
            print(f"  ⚠ 无法读取 Q2/{objective} anchor csv", flush=True)
            continue

        for sortie_id in selected["task_id"].astype(str):
            pattern_id = sortie_id.rsplit("-", 1)[0]
            pattern_ids.add(pattern_id)

    return pattern_ids


def _select_communication_patterns(
    comm: pd.DataFrame,
    pattern_counts: pd.DataFrame,
    top_k: int,
):
    u"""按 (class_id, UAV type) 做通信维度 Top-K 并集。

    通信维度压缩为 3 个核心指标：
      - outage_time_s（断连总时长）
      - gap_count（断连段数）
      - min_margin_db（最差信号余量）
    """

    comm = comm.copy()
    comm["pattern_id"] = comm["pattern_id"].astype(str)
    comm_index = comm.set_index("pattern_id", drop=False)
    chosen = set()

    communication_dims = [
        ("outage_time_s", True),
        ("gap_count", True),
        ("min_margin_db", False),
    ]

    for class_id, count_group in pattern_counts.groupby("class_id"):
        pattern_ids = count_group["pattern_id"].astype(str).unique()
        local = comm_index.loc[comm_index.index.intersection(pattern_ids)].copy()
        if local.empty:
            continue

        for uav_type, typed in local.groupby("uav_type"):
            for column, ascending in communication_dims:
                if ascending:
                    selected = typed.nsmallest(min(top_k, len(typed)), column)
                else:
                    selected = typed.nlargest(min(top_k, len(typed)), column)
                chosen.update(selected["pattern_id"].astype(str))

            direct = typed[typed["needs_relay"] == 0]
            if not direct.empty:
                chosen.update(
                    direct.nsmallest(min(top_k, len(direct)), "energy_kWh")[
                        "pattern_id"
                    ].astype(str)
                )

    return chosen


def filter_compact_candidates(top_k: int = TOP_K_PER_DIM):

    print("=" * 60)
    print("Q3 Compact Joint Candidate Filter")
    print("=" * 60)

    classes, patterns, counts = load_q2_compact_artifacts()

    comm = pd.read_csv(COMM_SUMMARY, encoding="utf-8-sig")

    pattern_ids = set(patterns["pattern_id"].astype(str))
    comm_ids = set(comm["pattern_id"].astype(str))

    if pattern_ids != comm_ids:
        raise RuntimeError(
            "Q2 compact patterns 与 Q3 communication summary 不一致"
        )

    transport_selected, _ = select_compact_patterns(
        patterns, counts, classes, top_k=top_k,
    )
    transport_ids = set(transport_selected["pattern_id"].astype(str))

    communication_ids = _select_communication_patterns(comm, counts, top_k)

    anchor_ids = _load_q2_anchor_pattern_ids()

    unknown_anchor = anchor_ids - pattern_ids
    if unknown_anchor:
        print(
            f"  ⚠ Anchor 引用了未知 pattern（已丢弃）: "
            f"{sorted(unknown_anchor)[:10]}",
            flush=True,
        )
        anchor_ids -= unknown_anchor

    chosen = transport_ids | communication_ids | anchor_ids

    selected_patterns = patterns[
        patterns["pattern_id"].astype(str).isin(chosen)
    ].copy()
    selected_counts = counts[
        counts["pattern_id"].astype(str).isin(chosen)
    ].copy()

    supplied_classes = set(classes["class_id"].astype(str))
    covered_classes = set(selected_counts["class_id"].astype(str))
    missing = supplied_classes - covered_classes

    if missing:
        raise RuntimeError(f"联合压缩造成 class 丢失: {sorted(missing)}")

    retained_ids = set(selected_patterns["pattern_id"].astype(str))
    lost_anchors = anchor_ids - retained_ids
    if lost_anchors:
        print(
            f"  ⚠ {len(lost_anchors)} 个 Q2 Anchor pattern 未被联合压缩保留",
            flush=True,
        )

    selected_patterns = selected_patterns.merge(
        comm.drop(
            columns=[
                "uav_type", "n_stops", "visit_order",
                "duration_s", "energy_kWh", "n_boxes",
            ],
            errors="ignore",
        ),
        on="pattern_id",
        how="left",
        validate="one_to_one",
    )

    selected_patterns.to_csv(PATTERNS_OUT, index=False, encoding="utf-8-sig")
    selected_counts.to_csv(COUNTS_OUT, index=False, encoding="utf-8-sig")

    stats = {
        "full_patterns": len(patterns),
        "transport_selected": len(transport_ids),
        "communication_selected": len(communication_ids),
        "anchor_seed_patterns": len(anchor_ids),
        "anchor_lost": len(lost_anchors),
        "final_selected": len(selected_patterns),
        "classes": len(classes),
        "class_coverage": "PASS",
    }

    manifest = {
        "step": "Q3 compact joint candidate filter",
        "inputs": {
            "q3_pattern_comm_summary.csv": _sha256(COMM_SUMMARY) if COMM_SUMMARY.exists() else "",
            "Q2_compact_candidates_manifest.json": (
                _sha256(DATA / "Q2_compact_candidates_manifest.json")
                if (DATA / "Q2_compact_candidates_manifest.json").exists()
                else ""
            ),
        },
        "outputs": {
            "q3_compact_patterns.csv": _sha256(PATTERNS_OUT) if PATTERNS_OUT.exists() else "",
            "q3_compact_pattern_counts.csv": _sha256(COUNTS_OUT) if COUNTS_OUT.exists() else "",
        },
        "stats": stats,
    }

    MANIFEST_OUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(stats)
    return selected_patterns, selected_counts, stats


def filter_candidates(*args, **kwargs):
    return filter_compact_candidates(top_k=kwargs.get("top_k", TOP_K_PER_DIM))


if __name__ == "__main__":
    filter_compact_candidates()