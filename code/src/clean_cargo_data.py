import pandas as pd
from pathlib import Path

r"""
物资需求数据清洗模块
====================

从 ``物资需求与配送时限.xlsx`` 中提取各服务区的货箱需求数据并输出为 CSV。

输出字段
--------
=============== ==================
字段             含义
=============== ==================
service         服务区编号 (S001~S015)
cargo_type      物资类型
total_boxes     总需求箱数
first_batch     首批必须送达箱数
mass_per_box    单箱质量 (kg)
volume_per_box  单箱体积 (m³)
priority        应急优先系数
first_deadline  首批截止时间 (s)
expected_time   期望送达时间 (s)
=============== ==================
"""


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_path = (
        project_root / "数据" / "无人机应急物资运输基础数据" / "物资需求与配送时限.xlsx"
    )
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(raw_path, header=0)

    col_map = {
        "服务区编号": "service",
        "物资类型": "cargo_type",
        "总需求箱数": "total_boxes",
        "首批必须送达箱数": "first_batch",
        "单箱质量（kg）": "mass_per_box",
        "单箱体积（m³）": "volume_per_box",
        "应急优先系数": "priority",
        "首批截止时间（s）": "first_deadline",
        "期望送达时间（s）": "expected_time",
    }

    df = df[list(col_map.keys())].rename(columns=col_map)
    df = df.dropna(subset=["service"]).reset_index(drop=True)

    for col in ["total_boxes", "first_batch", "mass_per_box",
                "volume_per_box", "priority"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out_path = output_dir / "物资需求.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"物资需求 -> {out_path.name}  ({len(df)} 条)")
    print(df.to_string())


if __name__ == "__main__":
    main()