"""Adaptively refine selected Step5 communication gaps at 1-second resolution."""

from __future__ import annotations

import pandas as pd

from src.q3.communication.direct_profile import (
    DirectProfileCache,
    required_segment_keys,
)
from src.q3.communication.relay_link import (
    evaluate_link,
    load_relay_link_parameters,
)
from src.q3.communication.terrain_block import DemTerrain
from src.q3.relay.coverage_library import _actual_coverage
from src.q3.transport.comm_gap import _assemble_pattern_profile_with_sources
from src.q3.transport.compact_loader import (
    build_compact_pattern_templates,
    load_q2_compact_artifacts,
)


AUDIT_COLUMNS = [
    "pattern_id", "gap_id", "fine_dt_s",
    "coarse_tau_start_s", "fine_tau_start_s", "refined_tau_start_s",
    "coarse_tau_end_s", "fine_tau_end_s", "refined_tau_end_s",
    "coarse_coverage_start_s", "fine_coverage_start_s",
    "refined_coverage_start_s", "coarse_coverage_end_s",
    "fine_coverage_end_s", "refined_coverage_end_s",
    "candidate_count_before", "candidate_count_after",
]


def _extract_fine_windows(samples):
    """Extract outages using the same boundary convention as Step5."""
    windows = []
    in_gap = False
    before = None
    previous = None
    outage_samples = []

    def append_window(after):
        first = outage_samples[0]
        end = after if after is not None else samples[-1]
        windows.append({
            "tau_start": float(first.tau),
            "tau_end": float(end.tau),
            "coverage_start": float(
                before.tau if before is not None else first.tau
            ),
            "coverage_end": float(end.tau),
            "samples": tuple(outage_samples),
        })

    for sample in samples:
        if not sample.direct and not in_gap:
            in_gap = True
            before = previous if previous is not None and previous.direct else None
            outage_samples = [sample]
        elif not sample.direct and in_gap:
            outage_samples.append(sample)
        elif sample.direct and in_gap:
            append_window(sample)
            in_gap = False
            before = None
            outage_samples = []
        previous = sample

    if in_gap:
        append_window(None)
    return windows


def _merge_gap_intervals(coarse, fine):
    """Take the conservative union so refinement can never relax Step5."""
    tau_start = min(float(coarse.tau_start), float(fine["tau_start"]))
    tau_end = max(float(coarse.tau_end), float(fine["tau_end"]))
    coverage_start = min(
        float(coarse.coverage_start), float(fine["coverage_start"])
    )
    coverage_end = max(
        float(coarse.coverage_end), float(fine["coverage_end"])
    )
    return {
        "tau_start": tau_start,
        "tau_end": tau_end,
        "duration_s": tau_end - tau_start,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "coverage_duration_s": coverage_end - coverage_start,
    }


def _combine_fine_windows(windows):
    """Conservatively combine several fine outages represented by one coarse gap."""
    if not windows:
        raise ValueError("Cannot combine an empty set of fine gap windows")
    return {
        "tau_start": min(float(window["tau_start"]) for window in windows),
        "tau_end": max(float(window["tau_end"]) for window in windows),
        "coverage_start": min(
            float(window["coverage_start"]) for window in windows
        ),
        "coverage_end": max(float(window["coverage_end"]) for window in windows),
        "samples": tuple(sorted(
            (sample for window in windows for sample in window["samples"]),
            key=lambda sample: float(sample.tau),
        )),
    }


def _retain_refined_options(gap_options, gap_id, candidate_metrics):
    """Filter options by fine coverage and conservatively update their margin."""
    mask = gap_options["gap_id"].astype(str) == str(gap_id)
    before = gap_options.loc[mask].copy()
    kept = []
    for _, option in before.iterrows():
        metric = candidate_metrics.get(str(option.candidate_id))
        if metric is None or not metric[0]:
            continue
        row = option.copy()
        row["min_access_margin_db"] = min(
            float(row["min_access_margin_db"]), float(metric[1])
        )
        kept.append(row)
    if not kept:
        raise RuntimeError(
            f"No Step6 candidate fully covers the 1s refined gap {gap_id}"
        )
    unaffected = gap_options.loc[~mask]
    refined = pd.concat([unaffected, pd.DataFrame(kept)], ignore_index=True)
    return refined, len(before), len(kept)


def _candidate_metrics(samples, options, sites, parameters, terrain):
    """Revalidate candidates on all fine outage samples using Step6 physics."""
    candidate_ids = set(options["candidate_id"].astype(str))
    site_rows = sites[sites["candidate_id"].astype(str).isin(candidate_ids)].copy()
    site_rows["candidate_id"] = site_rows["candidate_id"].astype(str)
    site_rows = site_rows.drop_duplicates("candidate_id").reset_index(drop=True)
    if site_rows.empty:
        raise RuntimeError("No Step6 candidate sites were found for refined gap")
    if not samples:
        raise RuntimeError("Fine communication gap has no outage samples")
    states = pd.DataFrame([
        {"x": float(sample.x), "y": float(sample.y), "z": float(sample.z),
         "phase": str(sample.phase)}
        for sample in samples
    ])
    packed, _ = _actual_coverage(states, site_rows, parameters, terrain)

    metrics = {}
    for index, site in site_rows.iterrows():
        cid = str(site.candidate_id)
        if float(site["relay_g01_margin_db"]) < -1e-9:
            metrics[cid] = (False, float("-inf"))
            continue
        covered = all(
            bool((packed[index, sample_index // 8]
                  >> (sample_index % 8)) & 1)
            for sample_index in range(len(samples))
        )
        if not covered:
            metrics[cid] = (False, float("-inf"))
            continue
        relay_position = (
            float(site.lon), float(site.lat), float(site.absolute_height)
        )
        fine_margin = min(
            float(evaluate_link(
                (float(sample.x), float(sample.y), float(sample.z)),
                relay_position,
                parameters.uav_relay_limit_db,
                parameters,
                terrain,
            )["margin_db"])
            for sample in samples
        )
        metrics[cid] = (True, fine_margin)
    return metrics


def refine_gap_inputs(
    gaps: pd.DataFrame,
    gap_options: pd.DataFrame,
    sites: pd.DataFrame,
    pattern_ids: set[str],
    dt: float = 1.0,
):
    """Refine selected patterns and return gaps, validated options, and audit."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    pattern_ids = {str(pattern_id) for pattern_id in pattern_ids}
    if not pattern_ids:
        return gaps.copy(), gap_options.copy(), pd.DataFrame(columns=AUDIT_COLUMNS)

    classes, all_patterns, all_counts = load_q2_compact_artifacts()
    patterns = all_patterns[
        all_patterns["pattern_id"].astype(str).isin(pattern_ids)
    ].copy()
    counts = all_counts[
        all_counts["pattern_id"].astype(str).isin(pattern_ids)
    ].copy()
    found = set(patterns["pattern_id"].astype(str))
    missing = sorted(pattern_ids - found)
    if missing:
        raise ValueError(f"Unknown fine-refinement pattern IDs: {missing}")
    templates = build_compact_pattern_templates(patterns, counts, classes)

    cache = DirectProfileCache(dt=float(dt))
    fine_by_pattern = {}
    try:
        cache.build_nodes()
        cache.build_segments(required_segment_keys(templates), verbose=False)
        for template in templates:
            profile = _assemble_pattern_profile_with_sources(template, cache)
            fine_by_pattern[str(template.pattern_id)] = _extract_fine_windows(profile)
    finally:
        cache.close()

    refined_gaps = gaps.copy()
    refined_options = gap_options.copy()
    audit = []
    link_parameters = load_relay_link_parameters()
    terrain = DemTerrain()
    try:
        for pattern_id in sorted(pattern_ids):
            coarse_rows = (
                refined_gaps[
                    refined_gaps["pattern_id"].astype(str) == pattern_id
                ].sort_values("gap_index")
            )
            fine_windows = fine_by_pattern[pattern_id]
            if len(coarse_rows) == 1 and len(fine_windows) >= 1:
                paired_fine_windows = [_combine_fine_windows(fine_windows)]
            elif len(coarse_rows) == len(fine_windows):
                paired_fine_windows = fine_windows
            else:
                raise RuntimeError(
                    f"{pattern_id}: 10s gap count={len(coarse_rows)}, "
                    f"1s gap count={len(fine_windows)}; cannot safely pair by gap_index"
                )
            for coarse, fine in zip(
                coarse_rows.itertuples(index=False), paired_fine_windows
            ):
                merged = _merge_gap_intervals(coarse, fine)
                gap_mask = refined_gaps["gap_id"].astype(str) == str(coarse.gap_id)
                for column, value in merged.items():
                    refined_gaps.loc[gap_mask, column] = value

                option_rows = refined_options[
                    refined_options["gap_id"].astype(str) == str(coarse.gap_id)
                ]
                if option_rows.empty:
                    raise RuntimeError(
                        f"{pattern_id}/{coarse.gap_id}: no Step6 candidates to refine"
                    )
                metrics = _candidate_metrics(
                    fine["samples"], option_rows, sites,
                    link_parameters, terrain,
                )
                refined_options, before_count, after_count = _retain_refined_options(
                    refined_options, coarse.gap_id, metrics
                )
                audit.append({
                    "pattern_id": pattern_id,
                    "gap_id": str(coarse.gap_id),
                    "fine_dt_s": float(dt),
                    "coarse_tau_start_s": float(coarse.tau_start),
                    "fine_tau_start_s": float(fine["tau_start"]),
                    "refined_tau_start_s": merged["tau_start"],
                    "coarse_tau_end_s": float(coarse.tau_end),
                    "fine_tau_end_s": float(fine["tau_end"]),
                    "refined_tau_end_s": merged["tau_end"],
                    "coarse_coverage_start_s": float(coarse.coverage_start),
                    "fine_coverage_start_s": float(fine["coverage_start"]),
                    "refined_coverage_start_s": merged["coverage_start"],
                    "coarse_coverage_end_s": float(coarse.coverage_end),
                    "fine_coverage_end_s": float(fine["coverage_end"]),
                    "refined_coverage_end_s": merged["coverage_end"],
                    "candidate_count_before": before_count,
                    "candidate_count_after": after_count,
                })
    finally:
        terrain.close()
    return refined_gaps, refined_options, pd.DataFrame(audit, columns=AUDIT_COLUMNS)
