"""Q3 compact transport loader.

Q3 直接消费 Q2 已保存的 62-class / compact-pattern 数据，
禁止重新构造另一套 physical-box candidate pool。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.q2.compact_classes import build_box_classes
from src.q2.data_model import load_q2_data


PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"

CLASSES_PATH = DATA / "Q2_compact_classes.csv"
MEMBERS_PATH = DATA / "Q2_compact_class_members.csv"
PATTERNS_PATH = DATA / "Q2_compact_patterns.csv"
COUNTS_PATH = DATA / "Q2_compact_pattern_counts.csv"
MANIFEST_PATH = DATA / "Q2_compact_candidates_manifest.json"


@dataclass(frozen=True)
class CompactPatternTemplate:
    pattern_id: str
    uav_type: str
    n_stops: int

    visit_order: tuple
    route: tuple

    n_boxes: int
    class_counts: Dict[str, int]
    delivery_offsets: Dict[str, float]
    service_counts: Dict[str, int]

    energy_kWh: float
    duration_s: float
    end_SOC: float

    has_hard_deadline: bool
    latest_start_s: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_q2_compact_artifacts():
    """读取并严格验证 Q2 compact artifacts。

    Returns
    -------
    classes:
        含 box_ids tuple 的真实 class 表。
    patterns:
        Q2 全量 compact patterns。
    pattern_counts:
        pattern-class count / delivery offset 表。
    """

    required = [
        CLASSES_PATH,
        MEMBERS_PATH,
        PATTERNS_PATH,
        COUNTS_PATH,
        MANIFEST_PATH,
    ]

    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少 Q2 compact artifacts: {missing}"
        )

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    output_names = {
        "classes": CLASSES_PATH,
        "class_members": MEMBERS_PATH,
        "patterns": PATTERNS_PATH,
        "pattern_counts": COUNTS_PATH,
    }

    for key, path in output_names.items():
        expected = manifest["outputs_sha256"][key]
        actual = _sha256(path)

        if actual != expected:
            raise RuntimeError(
                f"Q2 compact artifact 已变化或过期: {path.name}"
            )

    data = load_q2_data()
    boxes = data["boxes"]

    classes, _ = build_box_classes(boxes)

    saved_classes = pd.read_csv(
        CLASSES_PATH,
        encoding="utf-8-sig",
    )

    pd.testing.assert_frame_equal(
        classes.drop(columns=["box_ids"]).reset_index(drop=True),
        saved_classes.reset_index(drop=True),
        check_dtype=False,
    )

    current_members = pd.DataFrame([
        {
            "class_id": row.class_id,
            "box_id": box_id,
        }
        for row in classes.itertuples(index=False)
        for box_id in row.box_ids
    ])

    saved_members = pd.read_csv(
        MEMBERS_PATH,
        encoding="utf-8-sig",
    )

    pd.testing.assert_frame_equal(
        current_members.reset_index(drop=True),
        saved_members.reset_index(drop=True),
        check_dtype=False,
    )

    patterns = pd.read_csv(
        PATTERNS_PATH,
        encoding="utf-8-sig",
    )

    pattern_counts = pd.read_csv(
        COUNTS_PATH,
        encoding="utf-8-sig",
    )

    if len(classes) != int(manifest["n_classes"]):
        raise RuntimeError(
            f"class 数不一致: {len(classes)} != "
            f"{manifest['n_classes']}"
        )

    if len(boxes) != int(manifest["n_boxes"]):
        raise RuntimeError(
            f"box 数不一致: {len(boxes)} != "
            f"{manifest['n_boxes']}"
        )

    if pattern_counts["pattern_id"].isin(
        patterns["pattern_id"]
    ).all() is False:
        raise RuntimeError(
            "pattern_counts 存在未知 pattern_id"
        )

    return classes, patterns, pattern_counts


def build_compact_pattern_templates(
    patterns: pd.DataFrame,
    pattern_counts: pd.DataFrame,
    classes: pd.DataFrame,
) -> List[CompactPatternTemplate]:
    """将 DataFrame 转成 Q3 通信层使用的 pattern template。"""

    class_rows = classes.set_index("class_id")

    counts_by_pattern = {
        str(pattern_id): group.copy()
        for pattern_id, group
        in pattern_counts.groupby("pattern_id", sort=False)
    }

    templates: List[CompactPatternTemplate] = []

    for row in patterns.itertuples(index=False):

        pattern_id = str(row.pattern_id)

        if pattern_id not in counts_by_pattern:
            raise RuntimeError(
                f"{pattern_id} 没有 class-count 数据"
            )

        group = counts_by_pattern[pattern_id]

        class_counts = {
            str(item.class_id): int(item.count)
            for item in group.itertuples(index=False)
        }

        delivery_offsets = {
            str(item.class_id): float(item.delivery_offset_s)
            for item in group.itertuples(index=False)
        }

        service_counts = defaultdict(int)

        for class_id, amount in class_counts.items():
            service = str(class_rows.loc[class_id, "service"])
            service_counts[service] += int(amount)

        visit_order = tuple(
            node.strip()
            for node in str(row.visit_order).split(">")
            if node.strip()
        )

        if set(service_counts) != set(visit_order):
            raise RuntimeError(
                f"{pattern_id}: visit_order 与 class service 不一致"
            )

        latest = float(row.latest_start_s)

        if pd.isna(latest):
            latest = math.inf

        templates.append(
            CompactPatternTemplate(
                pattern_id=pattern_id,
                uav_type=str(row.uav_type),
                n_stops=int(row.n_stops),

                visit_order=visit_order,
                route=("O01",) + visit_order + ("O01",),

                n_boxes=int(row.n_boxes),
                class_counts=class_counts,
                delivery_offsets=delivery_offsets,
                service_counts=dict(service_counts),

                energy_kWh=float(row.energy_kWh),
                duration_s=float(row.duration_s),
                end_SOC=float(row.end_SOC),

                has_hard_deadline=bool(row.has_hard_deadline),
                latest_start_s=latest,
            )
        )

    return templates


def load_all_compact_pattern_templates():
    classes, patterns, counts = load_q2_compact_artifacts()

    templates = build_compact_pattern_templates(
        patterns,
        counts,
        classes,
    )

    return classes, patterns, counts, templates