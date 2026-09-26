import pandas as pd
from pathlib import Path

























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