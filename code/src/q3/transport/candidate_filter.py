u"""Q3 Step4：按货箱×机型多维 Top-K 并集压缩运输候选池。"""

import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
TASKS_SOURCE = DATA / "Q2_candidate_tasks.csv"
DELIVERIES_SOURCE = DATA / "Q2_candidate_deliveries.csv"
COMM_SOURCE = DATA / "q3_transport_comm_summary.csv"
TASKS_OUTPUT = DATA / "q3_candidate_tasks.csv"
DELIVERIES_OUTPUT = DATA / "q3_candidate_deliveries.csv"
SUMMARY_OUTPUT = DATA / "q3_candidate_filter_summary.csv"
COVERAGE_OUTPUT = DATA / "q3_candidate_box_coverage.csv"
MANIFEST_OUTPUT = DATA / "q3_candidate_filter_manifest.json"
SEED_PATHS = {
    objective: DATA / f"Q2_joint_selected_{objective}.csv"
    for objective in ("N", "E", "T")
}

LOWER_IS_BETTER = (
    "energy_kWh", "duration_s", "outage_time_s", "outage_ratio",
    "gap_count", "max_gap_s",
)
SLACK_METRIC = "latest_start_s"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_inputs():
    tasks = pd.read_csv(TASKS_SOURCE, encoding="utf-8-sig", low_memory=False)
    deliveries = pd.read_csv(DELIVERIES_SOURCE, encoding="utf-8-sig")
    comm = pd.read_csv(COMM_SOURCE, encoding="utf-8-sig")
    if tasks.task_id.duplicated().any() or comm.task_id.duplicated().any():
        raise ValueError("Q2任务或Q3通信摘要含重复 task_id")
    if set(tasks.task_id) != set(comm.task_id):
        raise ValueError("Q2候选任务与Q3通信摘要的 task_id 集合不一致，请先重跑 Step3")
    if deliveries.duplicated(["task_id", "box_id"]).any():
        raise ValueError("Q2候选交付表含重复 task_id×box_id")
    if set(deliveries.task_id) != set(tasks.task_id):
        raise ValueError("Q2候选交付表与任务表的 task_id 集合不一致")

    joined = tasks.merge(
        comm, on="task_id", how="inner", validate="one_to_one",
        suffixes=("_q2", ""),
    )
    for field in ("uav_type", "n_stops", "visit_order"):
        if not joined[f"{field}_q2"].equals(joined[field]):
            raise ValueError(f"Q2任务与Q3通信摘要的 {field} 不一致")
    for field in ("energy_kWh", "duration_s"):
        if (joined[f"{field}_q2"] - joined[field]).abs().max() > 1e-6:
            raise ValueError(f"Q2任务与Q3通信摘要的 {field} 不一致")
    # Q2 CSV 的旧 latest_start_s / has_hard_deadline 可能漏医疗箱；采用 Step3 重建值。
    joined = joined.drop(columns=[
        "uav_type_q2", "n_stops_q2", "visit_order_q2",
        "energy_kWh_q2", "duration_s_q2", "latest_start_s_q2",
        "has_hard_deadline", "charge_time_s",
    ])
    joined["has_hard_deadline"] = joined.latest_start_s.notna()
    if joined.task_id.isna().any() or joined.needs_relay.isna().any():
        raise ValueError("候选任务存在空 ID 或空通信状态")
    return joined, deliveries


def _read_seeds(task_ids, deliveries):
    seeds = {}
    for objective, path in SEED_PATHS.items():
        selected = pd.read_csv(path, encoding="utf-8-sig")
        ids = set(selected.task_id)
        if not ids or len(ids) != len(selected):
            raise ValueError(f"Q2 {objective}-opt seed 为空或 task_id 重复")
        missing = ids - task_ids
        if missing:
            raise ValueError(f"Q2 {objective}-opt seed 包含已不在候选池中的任务: {sorted(missing)[:5]}")
        covered = set(deliveries.loc[deliveries.task_id.isin(ids), "box_id"])
        all_boxes = set(deliveries.box_id)
        if covered != all_boxes:
            raise ValueError(f"Q2 {objective}-opt seed 未覆盖所有货箱: {len(covered)}/{len(all_boxes)}")
        seeds[objective] = ids
    return seeds


def _top_ids(group, metric, count, descending=False):
    if metric == SLACK_METRIC:
        group = group[group[metric].notna()]
    if group.empty:
        return []
    return group.sort_values(
        [metric, "task_id"], ascending=[not descending, True], kind="mergesort"
    ).head(count).task_id.tolist()


def select_candidates(tasks, deliveries, seeds, top_k=8, two_stop_top_k=4):
    if top_k < 1 or two_stop_top_k < 0:
        raise ValueError("Top-K 参数非法")
    metrics = ["task_id", "uav_type", "n_stops", SLACK_METRIC, *LOWER_IS_BETTER]
    membership = deliveries[["task_id", "box_id"]].merge(
        tasks[metrics], on="task_id", validate="many_to_one"
    )
    reasons = defaultdict(set)
    for (box_id, uav_type), group in membership.groupby(
        ["box_id", "uav_type"], sort=True
    ):
        strata = [(group, top_k, "all")]
        if two_stop_top_k:
            two_stop = group[group.n_stops == 2]
            if not two_stop.empty:
                strata.append((two_stop, two_stop_top_k, "two_stop"))
        for candidates, count, stratum in strata:
            for metric in LOWER_IS_BETTER:
                for task_id in _top_ids(candidates, metric, count):
                    reasons[task_id].add(f"{box_id}:{uav_type}:{stratum}:{metric}")
            for task_id in _top_ids(candidates, SLACK_METRIC, count, descending=True):
                reasons[task_id].add(f"{box_id}:{uav_type}:{stratum}:slack")
    for objective, ids in seeds.items():
        for task_id in ids:
            reasons[task_id].add(f"Q2_{objective}_seed")
    return set(reasons), reasons, membership


def _validate_retention(tasks, selected, deliveries, seeds, membership):
    all_boxes = set(deliveries.box_id)
    reduced_deliveries = deliveries[deliveries.task_id.isin(selected)]
    if set(reduced_deliveries.box_id) != all_boxes:
        raise AssertionError("筛选后未覆盖全部货箱")
    for objective, ids in seeds.items():
        if not ids <= selected:
            raise AssertionError(f"Q2 {objective}-opt seed 未全部保留")
    before_pairs = set(zip(membership.box_id, membership.uav_type))
    retained_pairs = membership[membership.task_id.isin(selected)]
    after_pairs = set(zip(retained_pairs.box_id, retained_pairs.uav_type))
    if before_pairs != after_pairs:
        raise AssertionError(f"筛选使 {len(before_pairs - after_pairs)} 个货箱×机型组合失去候选")
    counts_before = membership.groupby(["box_id", "uav_type"]).size()
    counts_after = retained_pairs.groupby(["box_id", "uav_type"]).size()
    coverage = pd.DataFrame({
        "box_id": [pair[0] for pair in counts_before.index],
        "uav_type": [pair[1] for pair in counts_before.index],
        "before_tasks": counts_before.to_numpy(),
        "after_tasks": counts_after.reindex(counts_before.index, fill_value=0).to_numpy(),
    })
    return reduced_deliveries, coverage


def _structure_summary(tasks, retained):
    rows = [{"dimension": "all", "category": "all", "before": len(tasks), "after": len(retained)}]
    for field in ("uav_type", "n_stops", "needs_relay"):
        before = tasks[field].value_counts()
        after = retained[field].value_counts()
        for category in sorted(before.index):
            rows.append({
                "dimension": field, "category": category,
                "before": int(before[category]), "after": int(after.get(category, 0)),
            })
    return pd.DataFrame(rows)


def run_candidate_filter(top_k=8, two_stop_top_k=4, write=True):
    tasks, deliveries = _read_inputs()
    task_ids = set(tasks.task_id)
    seeds = _read_seeds(task_ids, deliveries)
    selected, reasons, membership = select_candidates(
        tasks, deliveries, seeds, top_k=top_k,
        two_stop_top_k=two_stop_top_k,
    )
    retained = tasks[tasks.task_id.isin(selected)].copy().sort_values("task_id")
    reduced_deliveries, coverage = _validate_retention(
        tasks, selected, deliveries, seeds, membership
    )
    retained["q2_seed_objectives"] = retained.task_id.map(
        lambda task_id: ",".join(
            objective for objective in ("N", "E", "T")
            if task_id in seeds[objective]
        )
    )
    reduced_deliveries = reduced_deliveries.sort_values(["task_id", "box_id"])
    structure = _structure_summary(tasks, retained)
    if write:
        retained.to_csv(TASKS_OUTPUT, index=False, encoding="utf-8-sig")
        reduced_deliveries.to_csv(DELIVERIES_OUTPUT, index=False, encoding="utf-8-sig")
        structure.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
        coverage.to_csv(COVERAGE_OUTPUT, index=False, encoding="utf-8-sig")
        inputs = [TASKS_SOURCE, DELIVERIES_SOURCE, COMM_SOURCE, *SEED_PATHS.values()]
        outputs = [TASKS_OUTPUT, DELIVERIES_OUTPUT, SUMMARY_OUTPUT, COVERAGE_OUTPUT]
        manifest = {
            "step": "Q3 Step4 communication-aware transport candidate filtering",
            "run_command": "python code/Q3_step4.py",
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "policy": {
                "grouping": "each originally feasible box x UAV type",
                "top_k_each_metric": top_k,
                "two_stop_top_k_each_metric": two_stop_top_k,
                "metrics_ascending": list(LOWER_IS_BETTER),
                "metric_descending_finite_only": SLACK_METRIC,
                "combination": "union, then force-include Q2 N/E/T seeds",
                "no_absolute_energy_or_outage_threshold": True,
            },
            "before_tasks": len(tasks),
            "after_tasks": len(retained),
            "before_deliveries": len(deliveries),
            "after_deliveries": len(reduced_deliveries),
            "boxes_total": len(set(deliveries.box_id)),
            "box_type_pairs_preserved": len(coverage),
            "seed_task_counts": {objective: len(ids) for objective, ids in seeds.items()},
            "seed_retained": {
                objective: bool(ids <= selected) for objective, ids in seeds.items()
            },
            "structure": structure.to_dict(orient="records"),
            "input_sha256": {path.name: _sha256(path) for path in inputs},
            "output_sha256": {path.name: _sha256(path) for path in outputs},
        }
        MANIFEST_OUTPUT.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"Q3 Step4: {len(tasks)} → {len(retained)} 运输候选；"
        f"{len(set(deliveries.box_id))} 箱覆盖、{len(coverage)} 个货箱×机型组合、"
        "Q2 N/E/T seed 均保留。",
        flush=True,
    )
    print(structure.to_string(index=False), flush=True)
    return retained, reduced_deliveries, structure, coverage


if __name__ == "__main__":
    run_candidate_filter()
