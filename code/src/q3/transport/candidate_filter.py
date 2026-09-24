u"""Q3 Step4: 通信感知候选运输任务筛选。

对 ~71,595 个 Q2 候选任务进行多维保留式筛选，将候选池压缩到
CP-SAT 可求解的规模（~2,000–5,000），同时保持：
  - 80 箱全覆盖
  - 机型多样性
  - Q2 N/E/T-opt 种子任务全部保留
  - 通信维度候选不被 energy-only 筛选误删
  - 输出 deliveries 中的 deadline_s 使用统一口径重建

筛选策略: 按 (货箱, 机型) 分组，每个维度取 Top-K 取并集。
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .candidate_loader import load_box_deadlines

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"

TOP_K_PER_DIM = 8
SEED_DIR = DATA
COMM_SUMMARY = DATA / "q3_transport_comm_summary.csv"
TASKS_IN = DATA / "Q2_candidate_tasks.csv"
DELIVERIES_IN = DATA / "Q2_candidate_deliveries.csv"

TASKS_OUT = DATA / "q3_candidate_tasks.csv"
DELIVERIES_OUT = DATA / "q3_candidate_deliveries.csv"
SUMMARY_OUT = DATA / "q3_candidate_filter_summary.csv"
MANIFEST_OUT = DATA / "q3_candidate_filter_manifest.json"


def _load_q2_seeds() -> Set[str]:
    u"""读取 Q2 N-opt / E-opt / T-opt 三个方案选中的全部 task_id。"""
    seeds: Set[str] = set()
    for obj in ("N", "E", "T"):
        path = SEED_DIR / f"Q2_joint_selected_{obj}.csv"
        if not path.exists():
            print(f"  ⚠ 缺少种子文件 {path.name}，跳过", flush=True)
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        seeds.update(df["task_id"].astype(str).str.strip())
    return seeds


def _load_delivery_map(deliveries_path: Path = DELIVERIES_IN) -> Dict[str, List[str]]:
    u"""返回 {task_id → [box_id, ...]}。"""
    mapping: Dict[str, List[str]] = defaultdict(list)
    with deliveries_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            mapping[row["task_id"].strip()].append(row["box_id"].strip())
    return dict(mapping)


def _top_k_indices(
    values: np.ndarray,
    k: int,
    ascending: bool = True,
    finite_only: bool = False,
) -> np.ndarray:
    u"""返回 values 中 Top-K 的索引（不改变原数组顺序）。

    Args:
        values: 数值数组。
        k: 保留数量。
        ascending: True 取最小 K，False 取最大 K。
        finite_only: True 时只考虑有限值。
    """
    if len(values) == 0:
        return np.array([], dtype=np.intp)
    k = min(k, len(values))
    if finite_only:
        mask = np.isfinite(values)
        if mask.sum() == 0:
            return np.array([], dtype=np.intp)
        indices = np.where(mask)[0]
        vals = values[indices]
        k = min(k, len(vals))
        order = np.argsort(vals)
        if not ascending:
            order = order[::-1]
        return indices[order[:k]]
    order = np.argsort(values)
    if not ascending:
        order = order[::-1]
    return order[:k]


def _build_box_uav_groups(
    comm: pd.DataFrame,
    deliveries_map: Dict[str, List[str]],
) -> Dict[Tuple[str, str], pd.Index]:
    u"""按 (box_id, uav_type) 分组，返回每组对应的 DataFrame 行索引。"""
    task_uav = dict(zip(
        comm["task_id"].astype(str).str.strip(),
        comm["uav_type"].astype(str).str.strip(),
    ))
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, row in comm.iterrows():
        tid_str = str(row["task_id"]).strip()
        boxes = deliveries_map.get(tid_str, [])
        uav = task_uav.get(tid_str, "?")
        for box in boxes:
            groups[(box, uav)].append(int(idx))
    return {key: pd.Index(vals) for key, vals in groups.items()}


def filter_candidates(
    comm_path: Path = COMM_SUMMARY,
    tasks_path: Path = TASKS_IN,
    deliveries_path: Path = DELIVERIES_IN,
    top_k: int = TOP_K_PER_DIM,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    u"""执行通信感知候选筛选。

    Returns:
        (filtered_tasks, filtered_deliveries, stats_dict)
    """
    print("=" * 60, flush=True)
    print("Q3 Step4: 通信感知候选运输任务筛选", flush=True)
    print("=" * 60, flush=True)

    comm = pd.read_csv(comm_path, encoding="utf-8-sig")
    tasks = pd.read_csv(tasks_path, encoding="utf-8-sig")
    deliveries = pd.read_csv(deliveries_path, encoding="utf-8-sig")

    # 输入一致性 fail-fast
    comm_ids = set(comm["task_id"].astype(str).str.strip())
    task_ids = set(tasks["task_id"].astype(str).str.strip())
    deliv_ids = set(deliveries["task_id"].astype(str).str.strip())
    assert comm_ids == task_ids, (
        f"Task ID 不一致: comm={len(comm_ids)}, tasks={len(task_ids)}, "
        f"comm独有={len(comm_ids - task_ids)}, tasks独有={len(task_ids - comm_ids)}"
    )
    assert deliv_ids == task_ids, (
        f"Deliveries ID 不一致: deliv={len(deliv_ids)}, tasks={len(task_ids)}, "
        f"deliv独有={len(deliv_ids - task_ids)}, tasks独有={len(task_ids - deliv_ids)}"
    )
    print("输入一致性检查: PASS", flush=True)

    # 关键字段一致性
    comm_key = comm.set_index(comm["task_id"].astype(str).str.strip())[
        ["uav_type", "n_stops", "duration_s", "energy_kWh"]
    ]
    task_key = tasks.set_index(tasks["task_id"].astype(str).str.strip())[
        ["uav_type", "n_stops", "duration_s", "energy_kWh"]
    ]
    overlap_idx = comm_key.index.intersection(task_key.index)
    mismatches = []
    for col in ["uav_type", "n_stops"]:
        diff = (comm_key.loc[overlap_idx, col].astype(str)
                != task_key.loc[overlap_idx, col].astype(str)).sum()
        if diff > 0:
            mismatches.append(f"{col}: {diff}")
    for col in ["duration_s", "energy_kWh"]:
        diff = (np.abs(comm_key.loc[overlap_idx, col].astype(float)
                       - task_key.loc[overlap_idx, col].astype(float)) > 1e-3).sum()
        if diff > 0:
            mismatches.append(f"{col}: {diff}")
    if mismatches:
        raise AssertionError(f"Step3/Step2 字段不一致: {mismatches}")
    print("关键字段一致性: PASS", flush=True)

    n_original = len(comm)
    print(f"输入候选任务: {n_original}", flush=True)

    deliveries_map = _load_delivery_map(deliveries_path)
    box_uav_groups = _build_box_uav_groups(comm, deliveries_map)

    n_box_uav_groups = len(box_uav_groups)
    print(f"(货箱, 机型) 分组: {n_box_uav_groups}", flush=True)

    # 收集入选索引
    selected_mask = np.zeros(n_original, dtype=bool)

    # 维度定义
    dims = [
        ("energy_kWh", True, False, "E"),
        ("duration_s", True, False, "T"),
        ("latest_start_s", False, True, "S"),
        ("outage_time_s", True, False, "C_outage_time"),
        ("outage_ratio", True, False, "C_outage_ratio"),
        ("gap_count", True, False, "C_gap_count"),
    ]

    dim_counts: Dict[str, int] = defaultdict(int)

    for (box_id, uav_type), indices in box_uav_groups.items():
        local = comm.iloc[indices]
        for col, ascending, finite_only, dim_name in dims:
            vals = local[col].values
            if col == "latest_start_s":
                finite_only = True
            top_idx = _top_k_indices(vals, top_k, ascending=ascending,
                                     finite_only=finite_only)
            if len(top_idx) > 0:
                global_idx = indices[top_idx]
                selected_mask[global_idx] = True
                dim_counts[dim_name] += len(top_idx)

    n_from_dims = int(selected_mask.sum())
    print(f"\n维度 Top-{top_k} 并集: {n_from_dims} 个候选", flush=True)

    # 加入 Q2 种子
    seeds = _load_q2_seeds()
    print(f"Q2 种子任务(N+E+T): {len(seeds)} 个", flush=True)

    comm_ids_arr = comm["task_id"].astype(str).str.strip().values
    id_to_idx = {tid: i for i, tid in enumerate(comm_ids_arr)}
    n_seed_added = 0
    for tid in seeds:
        idx = id_to_idx.get(tid)
        if idx is not None and not selected_mask[idx]:
            selected_mask[idx] = True
            n_seed_added += 1
    print(f"种子任务新增: {n_seed_added}", flush=True)

    n_selected = int(selected_mask.sum())
    print(f"筛选后候选总数: {n_selected}", flush=True)

    # 构建输出
    selected_tids = set(comm_ids_arr[selected_mask])

    # --- 输出 tasks：完整 merge Q2 task 原字段 + Step3 通信字段 ---
    comm_sub = comm.iloc[selected_mask].copy()
    comm_sub["task_id"] = comm_sub["task_id"].astype(str).str.strip()

    tasks["task_id"] = tasks["task_id"].astype(str).str.strip()
    tasks_sub = tasks[tasks["task_id"].isin(selected_tids)].copy()

    # 从 tasks 中去掉 Step3 也有的列（避免重复）
    dup_cols = [c for c in comm_sub.columns
                if c in tasks_sub.columns and c != "task_id"]
    tasks_dedup = tasks_sub.drop(columns=dup_cols, errors="ignore")

    # merge: Q2 字段 + 通信摘要在右侧
    filtered_tasks = tasks_dedup.merge(
        comm_sub,
        on="task_id",
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)

    # 列顺序：task_id 先，Q2 列，再通信列
    comm_cols = [c for c in comm_sub.columns if c != "task_id"]
    q2_cols = [c for c in tasks_dedup.columns if c != "task_id"]
    ordered_cols = ["task_id"] + q2_cols + comm_cols
    filtered_tasks = filtered_tasks[ordered_cols]

    # --- 输出 deliveries：重建 hard_deadline_s ---
    box_deadlines = load_box_deadlines()

    filtered_deliveries = deliveries[
        deliveries["task_id"].astype(str).str.strip().isin(selected_tids)
    ].copy()
    filtered_deliveries["task_id"] = filtered_deliveries["task_id"].astype(str).str.strip()

    filtered_deliveries["deadline_s"] = (
        filtered_deliveries["box_id"]
        .astype(str).str.strip()
        .map(lambda bid: box_deadlines.get(bid, float("inf")))
    )
    filtered_deliveries = filtered_deliveries.reset_index(drop=True)

    n_hard = int((filtered_deliveries["deadline_s"].apply(math.isfinite)).sum())
    print(f"交付记录: {len(filtered_deliveries)} 条, "
          f"{n_hard} 条含硬时限 ({n_hard / len(filtered_deliveries) * 100:.1f}%)",
          flush=True)

    # --- 统计 ---
    # 货箱覆盖
    all_boxes: Set[str] = set()
    for box_list in deliveries_map.values():
        all_boxes.update(box_list)
    covered_boxes: Set[str] = set()
    for tid in selected_tids:
        covered_boxes.update(deliveries_map.get(tid, []))
    box_coverage_ok = covered_boxes == all_boxes
    missing_boxes = all_boxes - covered_boxes

    # 机型分布
    uav_counts = filtered_tasks["uav_type"].value_counts().to_dict()

    # 单点/双点
    single_stop = int((filtered_tasks["n_stops"] == 1).sum())
    double_stop = int((filtered_tasks["n_stops"] == 2).sum())

    # 直连/需中继
    direct_ok = int((filtered_tasks["needs_relay"] == 0).sum())
    needs_relay = int((filtered_tasks["needs_relay"] == 1).sum())

    # 种子覆盖
    seed_coverage: Dict[str, bool] = {}
    for obj in ("N", "E", "T"):
        path = SEED_DIR / f"Q2_joint_selected_{obj}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            seed_tids = set(df["task_id"].astype(str).str.strip())
            seed_coverage[obj] = seed_tids.issubset(selected_tids)

    # 机型完整性: 检测哪些 (box, uav_type) 组合在筛选后丢失了全部候选
    comm_tids_uav_map = dict(zip(
        comm["task_id"].astype(str).str.strip(),
        comm["uav_type"].astype(str).str.strip(),
    ))
    original_box_uav_sets: Dict[Tuple[str, str], int] = {
        key: len(vals) for key, vals in box_uav_groups.items()
    }
    selected_box_uav: Dict[Tuple[str, str], int] = defaultdict(int)
    for tid in selected_tids:
        boxes = deliveries_map.get(tid, [])
        uav = comm_tids_uav_map.get(tid, "?")
        for box in boxes:
            selected_box_uav[(box, uav)] += 1

    lost_groups: List[str] = []
    for (box, uav), orig_count in original_box_uav_sets.items():
        if selected_box_uav.get((box, uav), 0) == 0:
            lost_groups.append(f"{uav}-{box}")
    box_uav_coverage_ok = len(lost_groups) == 0

    stats = {
        "n_original": n_original,
        "n_selected": n_selected,
        "n_box_uav_groups": n_box_uav_groups,
        "n_from_dimensions": n_from_dims,
        "n_seed_total": len(seeds),
        "n_seed_added": n_seed_added,
        "uav_type_A": uav_counts.get("A", 0),
        "uav_type_B": uav_counts.get("B", 0),
        "uav_type_C": uav_counts.get("C", 0),
        "single_stop": single_stop,
        "double_stop": double_stop,
        "direct_only": direct_ok,
        "needs_relay": needs_relay,
        "box_coverage": "PASS" if box_coverage_ok else "FAIL",
        "missing_boxes": sorted(missing_boxes) if missing_boxes else [],
        "box_uav_coverage": "PASS" if box_uav_coverage_ok else "FAIL",
        "lost_box_uav_groups": lost_groups,
        "seed_N": "PASS" if seed_coverage.get("N", False) else "FAIL",
        "seed_E": "PASS" if seed_coverage.get("E", False) else "FAIL",
        "seed_T": "PASS" if seed_coverage.get("T", False) else "FAIL",
        "dim_counts": dict(dim_counts),
        "n_delivery_hard_deadlines": n_hard,
    }

    print(f"\n{'─' * 40}", flush=True)
    print(f"  A 型: {stats['uav_type_A']}", flush=True)
    print(f"  B 型: {stats['uav_type_B']}", flush=True)
    print(f"  C 型: {stats['uav_type_C']}", flush=True)
    print(f"  单点: {single_stop}  双点: {double_stop}", flush=True)
    print(f"  全直连: {direct_ok}  需中继评估: {needs_relay}", flush=True)
    print(f"  80 箱覆盖: {stats['box_coverage']}", flush=True)
    if missing_boxes:
        print(f"    缺失: {missing_boxes}", flush=True)
    print(f"  (货箱 x 机型) 覆盖: {stats['box_uav_coverage']}", flush=True)
    if lost_groups:
        print(f"    丢失组: {lost_groups[:10]}{' ...' if len(lost_groups) > 10 else ''}", flush=True)
    print(f"  Q2 N-opt seed: {stats['seed_N']}", flush=True)
    print(f"  Q2 E-opt seed: {stats['seed_E']}", flush=True)
    print(f"  Q2 T-opt seed: {stats['seed_T']}", flush=True)

    return filtered_tasks, filtered_deliveries, stats


def save_outputs(
    tasks_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    stats: dict,
    tasks_out: Path = TASKS_OUT,
    deliveries_out: Path = DELIVERIES_OUT,
    summary_out: Path = SUMMARY_OUT,
    manifest_out: Path = MANIFEST_OUT,
) -> None:
    u"""保存筛选结果。"""

    def _safe_write_csv(df: pd.DataFrame, path: Path, **kwargs):
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False, **kwargs)
        if path.exists():
            path.unlink()
        tmp.replace(path)

    _safe_write_csv(tasks_df, tasks_out, encoding="utf-8-sig")
    print(f"\n输出: {tasks_out.name} ({len(tasks_df)} 行, {len(tasks_df.columns)} 列)",
          flush=True)

    _safe_write_csv(deliveries_df, deliveries_out, encoding="utf-8-sig")
    print(f"输出: {deliveries_out.name} ({len(deliveries_df)} 行)", flush=True)

    summary_rows = [
        {"metric": k, "value": str(v) if isinstance(v, list) else v}
        for k, v in stats.items()
    ]
    summary_df = pd.DataFrame(summary_rows)
    _safe_write_csv(summary_df, summary_out, encoding="utf-8-sig")
    print(f"输出: {summary_out.name}", flush=True)

    manifest = {
        "step": "Q3 Step4: 通信感知候选运输任务筛选",
        "n_original": stats["n_original"],
        "n_selected": stats["n_selected"],
        "top_k_per_dim": TOP_K_PER_DIM,
        "dimensions": [
            "energy_kWh (min)",
            "duration_s (min)",
            "latest_start_s (max, finite only)",
            "outage_time_s (min)",
            "outage_ratio (min)",
            "gap_count (min)",
        ],
        "seeds": "Q2 N-opt + E-opt + T-opt 全部架次",
        "delivery_hard_deadlines": (
            f"{stats['n_delivery_hard_deadlines']} / "
            f"{stats.get('n_delivery_total', '?')} 条 (从 load_box_deadlines 统一重建)"
        ),
        "box_coverage": stats["box_coverage"],
        "uav_distribution": {
            "A": stats["uav_type_A"],
            "B": stats["uav_type_B"],
            "C": stats["uav_type_C"],
        },
        "direct_vs_relay": {
            "direct_only": stats["direct_only"],
            "needs_relay": stats["needs_relay"],
        },
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (COMM_SUMMARY, TASKS_IN, DELIVERIES_IN)
            if path.exists()
        },
    }
    tmp = manifest_out.with_suffix(manifest_out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if manifest_out.exists():
        manifest_out.unlink()
    tmp.replace(manifest_out)
    print(f"输出: {manifest_out.name}", flush=True)


def main(top_k: int = TOP_K_PER_DIM):
    comm_df, deliv_df, stats = filter_candidates(top_k=top_k)
    save_outputs(comm_df, deliv_df, stats)
    return comm_df, deliv_df, stats


if __name__ == "__main__":
    main()