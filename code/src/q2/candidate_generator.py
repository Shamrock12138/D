u"""
Q2 候选多点运输任务生成器
===========================

对 O01→visit_order→O01 的每条路线，枚举货箱组合、访问顺序和机型，
通过 4 层剪枝（质量/体积/能耗/硬时限）保留物理可行任务。

第一版限定 MAX_STOPS = 2（单点 + 双点）。

优化策略:
- 按机型预先过滤组合 (质量 ≤ Q_g, 体积 ≤ V_g)
- 预建 box_lookup 传给 evaluate_route 避免重复构建
- 双点: 机型在外层, 仅组合已被该机型预过滤的组合

输出
----
Q2_candidate_tasks.csv       — 每候选任务的全局属性
Q2_candidate_deliveries.csv  — 每候选任务 × 货箱 的逐箱映射
Q2_candidate_summary.csv     — 按 (n_stops, type) 的汇总统计
"""

import itertools
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent

CARGO_TYPES = ["医疗物资", "饮用水", "应急食品", "生活卫生用品"]
SHORT = {"医疗物资": "医疗", "饮用水": "饮水",
         "应急食品": "食品", "生活卫生用品": "卫生"}


def _group_boxes(boxes_df):
    u"""将 boxes_df 按 (service, cargo_type) 分组。"""
    pool = {}
    for sv in boxes_df["service"].unique():
        sv_df = boxes_df[boxes_df["service"] == sv].copy()
        cargo_info = {}
        for ct in CARGO_TYPES:
            ct_df = sv_df[sv_df["cargo_type"] == ct]
            if len(ct_df) == 0:
                continue
            cargo_info[ct] = {
                "count": len(ct_df),
                "mass": float(ct_df["mass"].iloc[0]),
                "volume": float(ct_df["volume"].iloc[0]),
                "first_required": int(ct_df["is_first_batch"].sum()),
                "deadline": (
                    float(ct_df["first_deadline"].iloc[0])
                    if pd.notna(ct_df["first_deadline"].iloc[0]) else None
                ),
            }

        pool[sv] = {
            "cargo": cargo_info,
            "total_boxes": len(sv_df),
            "box_ids": list(sv_df["box_id"]),
            "box_types": list(sv_df["cargo_type"]),
            "is_first_batch": [bool(f) for f in sv_df["is_first_batch"]],
        }
    return pool


def _enumerate_local_combos(cargo_info):
    u"""对单个服务区枚举所有非空货物组合。"""
    ranges = {}
    for ct in CARGO_TYPES:
        if ct in cargo_info:
            ranges[ct] = list(range(cargo_info[ct]["count"] + 1))
        else:
            ranges[ct] = [0]

    keys = list(ranges.keys())
    combos = []
    for values in itertools.product(*[ranges[k] for k in keys]):
        combo = dict(zip(keys, values))
        if sum(combo.values()) > 0:
            combos.append(combo)
    return combos


def _combo_mass_vol(combo, cargo_info):
    u"""计算组合的总质量和总体积。"""
    m = 0.0
    v = 0.0
    for ct, n in combo.items():
        if n > 0 and ct in cargo_info:
            m += n * cargo_info[ct]["mass"]
            v += n * cargo_info[ct]["volume"]
    return m, v


def _pick_boxes(combo, pool, sv, offset=0):
    u"""取 cargo_type 的前 n 个 box_id (同质消除)。

    offset 参数用于覆盖轮转: offset=0 取前 n 个, offset=1 跳过第一个, 以此类推。
    确保在同质货箱中, 所有箱子都有机会被候选任务覆盖。
    """
    delivery = []
    type_counters = defaultdict(int)
    target = sum(combo.values())
    for bid, bt in zip(pool[sv]["box_ids"], pool[sv]["box_types"]):
        if bt not in combo:
            continue
        type_counters[bt] += 1
        if type_counters[bt] <= offset:  # skip first 'offset' boxes
            continue
        if type_counters[bt] <= combo[bt] + offset:
            delivery.append(bid)
        if len(delivery) == target:
            break
    return delivery


def _deadline_violated(delivery_offsets, pool, sv, combo):
    u"""首批箱送达时间 > deadline → True。"""
    cargo_info = pool[sv]["cargo"]
    for ct, n in combo.items():
        if n == 0 or ct not in cargo_info:
            continue
        deadline = cargo_info[ct]["deadline"]
        if deadline is None or cargo_info[ct]["first_required"] == 0:
            continue

        offsets = [
            delivery_offsets[bid]
            for bid, bt in zip(pool[sv]["box_ids"], pool[sv]["box_types"])
            if bt == ct and bid in delivery_offsets
        ]

        if any(t > deadline + 1e-6 for t in offsets):
            return True
    return False


def _fmt(combo):
    u"""格式化 cargo pattern。"""
    parts = [
        f"{SHORT[ct]}:{combo.get(ct, 0)}"
        for ct in CARGO_TYPES
        if combo.get(ct, 0) > 0
    ]
    return "|".join(parts)


def generate_candidate_pool(boxes_df, models, max_stops=2):
    u"""生成所有物理可行的候选多点运输任务。

    按机型外循环 + 预过滤组合, 大幅减少 evaluate_route 调用。
    """
    from .route_evaluator import _box_data, evaluate_route

    pool = _group_boxes(boxes_df)
    services = sorted(pool.keys())
    g_names = sorted(models.keys())
    box_lookup = _box_data(boxes_df)

    print("=" * 60)
    print(f"Q2 Step 1: Candidate Task Generation  (max_stops={max_stops})")
    print("=" * 60)

    # 预计算每个服务区的所有 combo → (combo, mass, vol)
    sv_all_combos = {}
    sv_combo_count = 0
    for sv in services:
        combos = _enumerate_local_combos(pool[sv]["cargo"])
        sv_all_combos[sv] = [
            (c, *_combo_mass_vol(c, pool[sv]["cargo"]))
            for c in combos
        ]
        sv_combo_count += len(combos)
    print(f"服务区组合总数: {sv_combo_count} (avg {sv_combo_count/15:.0f}/sv)")

    # 按机型预过滤
    sv_viable = {}  # sv -> g -> [(combo, mass, vol)]
    for g_name in g_names:
        Q_g = float(models[g_name].u["Q_g"])
        V_g = float(models[g_name].u["V_g"])
        for sv in services:
            sv_viable.setdefault(sv, {})[g_name] = [
                (c, m, v)
                for c, m, v in sv_all_combos[sv]
                if m <= Q_g and v <= V_g
            ]

    viable_counts = {}
    for g_name in g_names:
        viable_counts[g_name] = sum(
            len(sv_viable[sv][g_name]) for sv in services
        )
    print(f"预过滤后每机型组合数: "
          f"A={viable_counts.get('A',0)} "
          f"B={viable_counts.get('B',0)} "
          f"C={viable_counts.get('C',0)}")

    pruned_mass = 0
    pruned_volume = 0
    pruned_energy = 0
    pruned_time = 0

    task_rows = []
    delivery_rows = []
    tid_counter = 0

    # ── 单点候选 (每机型) ────────────────────────────────
    print("\n── 单服务区候选 ──")
    t0 = time.time()
    for g_name in g_names:
        model = models[g_name]
        for sv in services:
            for combo, mass, vol in sv_viable[sv][g_name]:
                delivery = _pick_boxes(combo, pool, sv)

                r = evaluate_route(model, [sv], {sv: delivery},
                                   box_lookup=box_lookup)

                if not r["feasible"]:
                    pruned_energy += 1
                    continue

                if _deadline_violated(r["delivery_offsets"], pool, sv, combo):
                    pruned_time += 1
                    continue

                tid_counter += 1
                tid = f"R{tid_counter:06d}"
                task_rows.append({
                    "task_id": tid, "uav_type": g_name, "n_stops": 1,
                    "visit_order": sv, "cargo_pattern": _fmt(combo),
                    "n_boxes": sum(combo.values()),
                    "total_mass_kg": mass, "total_volume_m3": vol,
                    "energy_kWh": r["energy_kwh"],
                    "duration_s": r["duration_s"],
                    "end_SOC": round(r["end_soc"], 6),
                    "charge_time_s": 0.0,
                })
                for bid in delivery:
                    delivery_rows.append({
                        "task_id": tid,
                        "box_id": bid,
                        "delivery_offset_s": round(
                            r["delivery_offsets"].get(bid, 0.0), 1
                        ),
                    })
    t1 = time.time()
    print(f"  耗时: {t1 - t0:.1f}s, 候选数: {tid_counter}")

    # ── 双点候选 ──────────────────────────────────────────
    if max_stops >= 2:
        print("\n── 双服务区候选 ──")
        t0 = time.time()
        pairs = list(itertools.combinations(services, 2))
        npairs = len(pairs)

        for g_name in g_names:
            model = models[g_name]
            Q_g = float(model.u["Q_g"])
            V_g = float(model.u["V_g"])
            print(f"\n  机型 {g_name} (Q={Q_g}kg)...")
            t_g0 = time.time()

            for pi, (sv_a, sv_b) in enumerate(pairs):
                if npairs > 10 and pi % 30 == 0 and pi > 0:
                    t_elapsed = time.time() - t_g0
                    eta = t_elapsed / pi * npairs - t_elapsed
                    print(f"    {pi}/{npairs}  已用时 {t_elapsed:.0f}s  "
                          f"预计剩余 {eta:.0f}s  当前候选数 {tid_counter}")

                va = sv_viable[sv_a][g_name]
                vb = sv_viable[sv_b][g_name]

                for (ca, ma, va_vol), (cb, mb, vb_vol) in itertools.product(va, vb):
                    if ma + mb > Q_g:
                        pruned_mass += 1
                        continue
                    if va_vol + vb_vol > V_g:
                        pruned_volume += 1
                        continue

                    da = _pick_boxes(ca, pool, sv_a)
                    db = _pick_boxes(cb, pool, sv_b)

                    for first, second in [(sv_a, sv_b), (sv_b, sv_a)]:
                        visit = [first, second]
                        dels = {
                            first: da if first == sv_a else db,
                            second: db if second == sv_b else da,
                        }

                        r = evaluate_route(model, visit, dels,
                                           box_lookup=box_lookup)

                        if not r["feasible"]:
                            pruned_energy += 1
                            continue

                        ok = True
                        for sv_check, combo_check in [(sv_a, ca), (sv_b, cb)]:
                            if _deadline_violated(
                                r["delivery_offsets"], pool,
                                sv_check, combo_check
                            ):
                                pruned_time += 1
                                ok = False
                                break
                        if not ok:
                            continue

                        tid_counter += 1
                        tid = f"R{tid_counter:06d}"
                        task_rows.append({
                            "task_id": tid, "uav_type": g_name, "n_stops": 2,
                            "visit_order": ">".join(visit),
                            "cargo_pattern": (
                                f"{first}:[{_fmt(ca)}] "
                                f"{second}:[{_fmt(cb)}]"
                            ),
                            "n_boxes": sum(ca.values()) + sum(cb.values()),
                            "total_mass_kg": ma + mb,
                            "total_volume_m3": va_vol + vb_vol,
                            "energy_kWh": r["energy_kwh"],
                            "duration_s": r["duration_s"],
                            "end_SOC": round(r["end_soc"], 6),
                            "charge_time_s": 0.0,
                        })
                        for bid in da + db:
                            delivery_rows.append({
                                "task_id": tid,
                                "box_id": bid,
                                "delivery_offset_s": round(
                                    r["delivery_offsets"].get(bid, 0.0), 1
                                ),
                            })
            t_g1 = time.time()
            print(f"  机型 {g_name} 耗时: {t_g1 - t_g0:.1f}s")
        t1 = time.time()
        print(f"\n  双点总耗时: {t1 - t0:.1f}s")

    # ── 汇总 ──────────────────────────────────────────────
    tasks_df = pd.DataFrame(task_rows)
    deliveries_df = pd.DataFrame(delivery_rows)

    # ── 覆盖填补: 未覆盖的货箱用最小单箱任务补齐 ──────────
    all_ids = set(boxes_df["box_id"])
    covered = set(deliveries_df["box_id"])
    missing = all_ids - covered

    if missing:
        box_info = dict(zip(boxes_df["box_id"],
                            zip(boxes_df["service"], boxes_df["mass"],
                                boxes_df["cargo_type"])))
        for bid in sorted(missing):
            sv_svc, mass, ct = box_info[bid]
            tiny_combo = {c: (1 if c == ct else 0) for c in CARGO_TYPES}

            ct_boxes = [
                b for b, bt in zip(pool[sv_svc]["box_ids"],
                                   pool[sv_svc]["box_types"])
                if bt == ct
            ]
            pos = ct_boxes.index(bid)
            if pos < 0:
                continue
            sv_del = ct_boxes[pos:pos+1]

            for g_name in g_names:
                model = models[g_name]
                Q_g = float(model.u["Q_g"])
                if mass > Q_g:
                    continue
                r = evaluate_route(model, [sv_svc], {sv_svc: sv_del},
                                   box_lookup=box_lookup)
                if r["feasible"]:
                    tid_counter += 1
                    tid = f"R{tid_counter:06d}"
                    task_rows.append({
                        "task_id": tid, "uav_type": g_name, "n_stops": 1,
                        "visit_order": sv_svc, "cargo_pattern": _fmt(tiny_combo),
                        "n_boxes": 1, "total_mass_kg": mass,
                        "total_volume_m3": r["total_volume"],
                        "energy_kWh": r["energy_kwh"],
                        "duration_s": r["duration_s"],
                        "end_SOC": round(r["end_soc"], 6),
                        "charge_time_s": 0.0,
                    })
                    delivery_rows.append({
                        "task_id": tid,
                        "box_id": bid,
                        "delivery_offset_s": round(
                            r["delivery_offsets"].get(bid, 0.0), 1
                        ),
                    })
                    break

        tasks_df = pd.DataFrame(task_rows)
        deliveries_df = pd.DataFrame(delivery_rows)
        covered = set(deliveries_df["box_id"])
        missing = all_ids - covered

    single = len(tasks_df[tasks_df["n_stops"] == 1])
    double = len(tasks_df[tasks_df["n_stops"] == 2])
    print(f"\n{'='*60}")
    print(f"单服务区候选: {single}")
    print(f"双服务区候选: {double}")
    print(f"\n质量剪枝: {pruned_mass}")
    print(f"体积剪枝: {pruned_volume}")
    print(f"能耗剪枝: {pruned_energy}")
    print(f"时间剪枝: {pruned_time}")
    print(f"\n最终候选任务数: {len(tasks_df)}")

    all_ids = set(boxes_df["box_id"])
    covered = set(deliveries_df["box_id"])
    missing = all_ids - covered
    extra = covered - all_ids
    tag = "PASS" if len(missing) == 0 else "FAIL"
    print(f"\n[{tag}] 80箱全覆盖: missing={len(missing)}, extra={len(extra)}")
    if missing:
        print(f"  缺失: {sorted(missing)}")

    summary_rows = []
    for ns in sorted(tasks_df["n_stops"].unique()):
        for g in sorted(models.keys()):
            sub = tasks_df[(tasks_df["n_stops"] == ns)
                           & (tasks_df["uav_type"] == g)]
            summary_rows.append({
                "n_stops": ns, "uav_type": g, "count": len(sub),
                "avg_energy": (
                    round(float(sub["energy_kWh"].mean()), 4)
                    if len(sub) > 0 else 0.0
                ),
                "avg_duration": (
                    round(float(sub["duration_s"].mean()), 1)
                    if len(sub) > 0 else 0.0
                ),
            })
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    _validate_pool(tasks_df, deliveries_df, boxes_df, models)
    return tasks_df, deliveries_df, summary_df


def _validate_pool(tasks_df, deliveries_df, boxes_df, models):
    u"""候选池内部一致性校验。"""
    all_box_ids = set(boxes_df["box_id"])
    ok = True

    # 货箱真实性
    delivery_ids = set(deliveries_df["box_id"])
    extra = delivery_ids - all_box_ids
    if extra:
        print(f"  FAIL: 候选引用了不存在的货箱: {sorted(extra)}")
        ok = False

    # 每任务内无重复货箱
    for tid, group in deliveries_df.groupby("task_id"):
        if len(group) != len(set(group["box_id"])):
            print(f"  FAIL {tid}: 任务内货箱重复")
            ok = False

    for _, task in tasks_df.iterrows():
        tid = task["task_id"]
        g = task["uav_type"]
        model = models[g]
        Q_g = float(model.u["Q_g"])
        V_g = float(model.u["V_g"])
        E_avail = model.available_energy

        if task["total_mass_kg"] > Q_g + 1e-9:
            print(f"  FAIL {tid}: mass {task['total_mass_kg']} > {Q_g}")
            ok = False
        if task["total_volume_m3"] > V_g + 1e-9:
            print(f"  FAIL {tid}: vol {task['total_volume_m3']} > {V_g}")
            ok = False
        if task["energy_kWh"] > E_avail + 1e-9:
            print(f"  FAIL {tid}: energy {task['energy_kWh']} > {E_avail}")
            ok = False

        if task["n_stops"] >= 2:
            visit = task["visit_order"].split(">")
            if len(set(visit)) != len(visit):
                print(f"  FAIL {tid}: visit repeat")
                ok = False
            if len(visit) != task["n_stops"]:
                print(f"  FAIL {tid}: visit len {len(visit)} != n_stops {task['n_stops']}")
                ok = False

    if ok:
        print("[PASS] 所有 candidate 质量/体积/能量合法")
        print("[PASS] 货箱真实存在且任务内无重复")
        print("[PASS] 服务区无重复访问")
    return ok


if __name__ == "__main__":
    from src.physics import load_models
    from src.q2.data_model import load_q2_data

    data = load_q2_data()
    models = load_models()

    tasks_df, deliveries_df, summary_df = generate_candidate_pool(
        data["boxes"], models, max_stops=2
    )

    out = PROJECT / "data"
    print(f"\n准备保存: tasks={len(tasks_df)} deliveries={len(deliveries_df)}")
    print(f"  → {out / 'Q2_candidate_tasks.csv'}")
    tasks_df.to_csv(out / "Q2_candidate_tasks.csv",
                    index=False, encoding="utf-8-sig")
    print(f"  → {out / 'Q2_candidate_deliveries.csv'}")
    deliveries_df.to_csv(out / "Q2_candidate_deliveries.csv",
                         index=False, encoding="utf-8-sig")
    print(f"  → {out / 'Q2_candidate_summary.csv'}")
    summary_df.to_csv(out / "Q2_candidate_summary.csv",
                      index=False, encoding="utf-8-sig")
    print(f"\n已保存: data/Q2_candidate_tasks.csv        ({len(tasks_df)} rows)")
    print(f"已保存: data/Q2_candidate_deliveries.csv   ({len(deliveries_df)} rows)")
    print(f"已保存: data/Q2_candidate_summary.csv      ({len(summary_df)} rows)")