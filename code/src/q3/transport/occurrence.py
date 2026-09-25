u"""Q3 Pattern→Sortie Occurrence 实例化。

将筛选后的 compact patterns 展开为独立 sortie occurrence，
每个 occurrence 携带 pattern 模板、class counts、gaps。
默认内存生成，可选保存一份 CSV 供查验。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.q2.compact_classes import expand_pattern_counts, pattern_multiplicity

from .compact_loader import build_compact_pattern_templates, load_q2_compact_artifacts

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"

CANDIDATE_PATTERNS = DATA / "q3_compact_patterns.csv"
CANDIDATE_COUNTS = DATA / "q3_compact_pattern_counts.csv"
PATTERN_GAPS = DATA / "q3_pattern_comm_gaps.csv"

OCCURRENCE_CSV = DATA / "q3_sortie_occurrences.csv"


@dataclass
class SortieOccurrence:
    sortie_id: str
    pattern_id: str
    copy_index: int
    uav_type: str
    visit_order: List[str]
    route: List[str]
    n_boxes: int
    class_counts: Dict[str, int]
    service_counts: Dict[str, int]
    energy_kWh: float
    duration_s: float
    end_SOC: float
    has_hard_deadline: bool
    latest_start_s: float
    gap_ids: List[str]


def generate_occurrences(
    classes: Optional[pd.DataFrame] = None,
    patterns: Optional[pd.DataFrame] = None,
    counts: Optional[pd.DataFrame] = None,
    gaps: Optional[pd.DataFrame] = None,
    save_csv: bool = False,
) -> List[SortieOccurrence]:
    u"""将 compact pattern × multiplicity 展开为 sortie occurrence 列表。

    Parameters
    ----------
    classes, patterns, counts, gaps:
        若未提供则自动从 CSV 加载。
    save_csv:
        若 True，额外写一份 q3_sortie_occurrences.csv 供查验。
    """

    if classes is None or patterns is None or counts is None:
        classes, patterns, counts = load_q2_compact_artifacts()
        patterns = pd.read_csv(CANDIDATE_PATTERNS, encoding="utf-8-sig")
        counts = pd.read_csv(CANDIDATE_COUNTS, encoding="utf-8-sig")

    templates = build_compact_pattern_templates(patterns, counts, classes)
    template_map = {t.pattern_id: t for t in templates}

    class_supply = {
        row.class_id: len(row.box_ids)
        for row in classes.itertuples(index=False)
    }

    pattern_counts_dict = {}
    for pattern_id, group in counts.groupby("pattern_id", sort=False):
        pattern_counts_dict[str(pattern_id)] = {
            str(row.class_id): int(row.count)
            for row in group.itertuples(index=False)
        }

    slots = expand_pattern_counts(pattern_counts_dict, class_supply)

    if gaps is None and PATTERN_GAPS.exists():
        gaps = pd.read_csv(PATTERN_GAPS, encoding="utf-8-sig")

    gap_map: Dict[str, List[str]] = {}
    if gaps is not None:
        for row in gaps.itertuples(index=False):
            pid = str(row.pattern_id)
            gid = str(row.gap_id)
            gap_map.setdefault(pid, []).append(gid)

    occurrences: List[SortieOccurrence] = []

    for slot in slots:
        pid = slot["pattern_id"]
        template = template_map.get(pid)
        if template is None:
            continue

        occurrences.append(
            SortieOccurrence(
                sortie_id=slot["sortie_id"],
                pattern_id=pid,
                copy_index=int(slot["copy"]),
                uav_type=template.uav_type,
                visit_order=list(template.visit_order),
                route=list(template.route),
                n_boxes=template.n_boxes,
                class_counts=dict(slot["class_counts"]),
                service_counts=dict(template.service_counts),
                energy_kWh=template.energy_kWh,
                duration_s=template.duration_s,
                end_SOC=template.end_SOC,
                has_hard_deadline=template.has_hard_deadline,
                latest_start_s=template.latest_start_s,
                gap_ids=gap_map.get(pid, []),
            )
        )

    if save_csv:
        rows = [
            {
                "sortie_id": occ.sortie_id,
                "pattern_id": occ.pattern_id,
                "copy_index": occ.copy_index,
                "uav_type": occ.uav_type,
                "visit_order": ">".join(occ.visit_order),
                "n_boxes": occ.n_boxes,
                "class_counts_str": ";".join(
                    f"{c}={n}" for c, n in sorted(occ.class_counts.items())
                ),
                "energy_kWh": occ.energy_kWh,
                "duration_s": occ.duration_s,
                "end_SOC": occ.end_SOC,
                "has_hard_deadline": int(occ.has_hard_deadline),
                "latest_start_s": occ.latest_start_s
                if occ.latest_start_s != float("inf")
                else "",
                "n_gaps": len(occ.gap_ids),
            }
            for occ in occurrences
        ]
        pd.DataFrame(rows).to_csv(OCCURRENCE_CSV, index=False, encoding="utf-8-sig")
        print(f"  输出: {OCCURRENCE_CSV.name} ({len(rows)} occurrences)", flush=True)

    return occurrences


if __name__ == "__main__":
    occs = generate_occurrences(save_csv=True)
    print(f"  总 occurrence 数: {len(occs)}")
    if occs:
        print(f"  pattern 种类: {len(set(o.pattern_id for o in occs))}")
        print(f"  含 gap occurrence: {sum(1 for o in occs if o.gap_ids)}")
        print(f"  全直连 occurrence: {sum(1 for o in occs if not o.gap_ids)}")