













import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent


def load_boxes():
    










    demand = pd.read_csv(PROJECT / "data" / "物资需求.csv")

    rows = []
    box_counter = 0

    for _, row in demand.iterrows():
        n_boxes = int(row["total_boxes"])
        first_required = int(row.get("first_batch", 0))
        for local_idx in range(n_boxes):
            box_counter += 1
            rows.append({
                "box_id": f"B{box_counter:03d}",
                "service": row["service"],
                "cargo_type": row["cargo_type"],
                "mass": float(row["mass_per_box"]),
                "volume": float(row["volume_per_box"]),
                "priority": int(row.get("priority", 0)),
                "first_batch_required": first_required,
                "is_first_batch": local_idx < first_required,
                "first_deadline": (
                    float(row["first_deadline"])
                    if pd.notna(row.get("first_deadline"))
                    else None
                ),
                "expected_time": (
                    float(row["expected_time"])
                    if pd.notna(row.get("expected_time"))
                    else None
                ),
            })

    df = pd.DataFrame(rows)

    cargo_map = {"医疗物资": 12, "饮用水": 8, "应急食品": 6, "生活卫生用品": 4}
    df["cargo_code"] = df["cargo_type"].map(cargo_map)

    return df


def load_uavs():
    








    return pd.read_csv(PROJECT / "data" / "运输无人机_清单.csv")


def load_batteries():
    








    battery_spec = pd.read_csv(PROJECT / "data" / "运输无人机_共享电池.csv")

    rows = []
    for _, row in battery_spec.iterrows():
        typ = row["type"]
        count = int(row["shared_battery_count"])
        charge_time = int(row["charge_time"])
        for i in range(count):
            rows.append({
                "battery_id": f"BAT_{typ}{i + 1:02d}",
                "type": typ,
                "full_charge_time": charge_time,
            })

    return pd.DataFrame(rows)


def load_q2_data():
    








    return {
        "boxes": load_boxes(),
        "uavs": load_uavs(),
        "batteries": load_batteries(),
    }


if __name__ == "__main__":
    data = load_q2_data()
    print(f"货箱总数: {len(data['boxes'])}")
    print(f"无人机数: {len(data['uavs'])}")
    print(f"电池数:   {len(data['batteries'])}")
    print()
    print(data["boxes"].head(10).to_string(index=False))
    print()
    print(data["batteries"].to_string(index=False))