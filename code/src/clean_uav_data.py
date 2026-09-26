import pandas as pd
import numpy as np
from pathlib import Path













































def clean_uav_params():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_path = (
        project_root / "数据" / "无人机应急物资运输基础数据" / "运输无人机数据.xlsx"
    )
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_excel(raw_path, header=None)

    header_row = df_raw.iloc[1, :].tolist()
    type_start = None
    for i, val in enumerate(header_row):
        if isinstance(val, str) and "机型编号" in val:
            type_start = i
            break

    if type_start is None:
        raise ValueError("未找到机型编号表头")

    col_map = {
        "机型编号": "type",
        "机型名称": "name",
        "含电池空载总质量（kg）": "m_empty",
        "最大载货质量（kg）": "M_k",
        "可用装载体积（m³）": "V_k",
        "计划巡航速度（m/s）": "v_k^cr",
        "空载标准航程（m）": "range_empty",
        "满载标准航程（m）": "range_full",
        "电池可用能量（kWh）": "B_k",
        "返航电量下限（%）": "ρ",
        "工位固定准备时间（s）": "T_setup",
        "每箱装载时间（s）": "T_load",
        "接收点基础交接时间（s）": "T_handover",
        "每箱增加交接时间（s）": "T_handover_p",
        "最大爬升速度（m/s）": "v_k^up",
        "最大下降速度（m/s）": "v_k^down",
        "爬升能耗效率": "eta_up",
        "下降能耗效率": "eta_down",
    }

    df_types = df_raw.iloc[2:5, :].copy()
    df_types.columns = header_row
    df_types = df_types[list(col_map.keys())].rename(columns=col_map)
    df_types = df_types.reset_index(drop=True)

    for col in ["M_k", "V_k", "B_k", "v_k^up", "v_k^cr", "v_k^down",
                "m_empty", "range_empty", "range_full", "eta_up", "eta_down",
                "ρ", "T_setup", "T_load", "T_handover", "T_handover_p"]:
        df_types[col] = pd.to_numeric(df_types[col], errors="coerce")

    g = 9.8
    df_types["P_k^cr"] = (
        df_types["B_k"] * 3600 * df_types["v_k^cr"] / df_types["range_empty"]
    )
    df_types["P_k^up"] = (
        (df_types["m_empty"] + df_types["M_k"]) * g
        * df_types["v_k^up"] / df_types["eta_up"] / 1000
    )

    for col in ["P_k^cr", "P_k^up"]:
        df_types[col] = df_types[col].round(3)

    df_types["M_g0"] = df_types["m_empty"]
    df_types["Q_g"] = df_types["M_k"]
    df_types["V_g"] = df_types["V_k"]
    df_types["L_0"] = df_types["range_empty"]
    df_types["L_F"] = df_types["range_full"]
    df_types["E_use"] = df_types["B_k"]
    df_types["ρ_g"] = df_types["ρ"]
    df_types["η_up"] = df_types["eta_up"]

    cols_ordered = ["type", "name",
                    "M_g0", "Q_g", "V_g",
                    "L_0", "L_F", "E_use",
                    "ρ_g", "η_up",
                    "v_k^up", "v_k^cr", "v_k^down",
                    "T_setup", "T_load", "T_handover", "T_handover_p"]
    df_types = df_types[cols_ordered]

    out_path = output_dir / "运输无人机_机型参数.csv"
    df_types.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[1/3] 机型参数 -> {out_path.name}  ({len(df_types)} 条)")

    return df_types


def clean_uav_inventory():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_path = (
        project_root / "数据" / "无人机应急物资运输基础数据" / "运输无人机数据.xlsx"
    )
    output_dir = Path(__file__).resolve().parent.parent / "data"

    df_raw = pd.read_excel(raw_path, header=None)

    inv_header = df_raw.iloc[7, :3].tolist()
    df_inv = df_raw.iloc[8:16, :3].copy()
    df_inv.columns = inv_header
    df_inv = df_inv.rename(columns={
        "无人机编号": "UAV_id",
        "机型编号": "type",
        "初始位置": "location",
    })
    df_inv = df_inv.dropna().reset_index(drop=True)

    out_path = output_dir / "运输无人机_清单.csv"
    df_inv.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[2/3] 无人机清单 -> {out_path.name}  ({len(df_inv)} 架)")

    return df_inv


def clean_battery_inventory():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_path = (
        project_root / "数据" / "无人机应急物资运输基础数据" / "运输无人机数据.xlsx"
    )
    output_dir = Path(__file__).resolve().parent.parent / "data"

    df_raw = pd.read_excel(raw_path, header=None)

    bat_header = df_raw.iloc[18, :3].tolist()
    df_bat = df_raw.iloc[19:22, :3].copy()
    df_bat.columns = bat_header
    df_bat = df_bat.rename(columns={
        "机型编号": "type",
        "共享电池组总数（组）": "shared_battery_count",
        "等效完全充电时间（s）": "charge_time",
    })
    df_bat = df_bat.dropna().reset_index(drop=True)
    df_bat["shared_battery_count"] = pd.to_numeric(df_bat["shared_battery_count"])
    df_bat["charge_time"] = pd.to_numeric(df_bat["charge_time"])

    out_path = output_dir / "运输无人机_共享电池.csv"
    df_bat.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[3/3] 共享电池 -> {out_path.name}  ({len(df_bat)} 条)")

    return df_bat


def main():
    print("=" * 55)
    print("运输无人机数据清洗")
    print("=" * 55)
    print()

    types = clean_uav_params()
    print(types.to_string())
    print()

    inv = clean_uav_inventory()
    print(inv.to_string())
    print()

    bat = clean_battery_inventory()
    print(bat.to_string())

    print()
    print("=" * 55)
    print("完成!")


if __name__ == "__main__":
    main()