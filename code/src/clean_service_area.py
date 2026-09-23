import pandas as pd
from pathlib import Path


def clean_service_area():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_path = project_root / "数据" / "无人机应急物资运输基础数据" / "调度中心与服务区.xlsx"
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "服务区数据.csv"

    df_raw = pd.read_excel(raw_path, header=None)

    service_area_start = None
    for i, row in df_raw.iterrows():
        if row.astype(str).str.contains("服务区编号").any():
            service_area_start = i
            break

    if service_area_start is None:
        raise ValueError("未找到'服务区编号'表头行")

    df = df_raw.iloc[service_area_start:].copy()
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    df = df[["服务区编号", "经度（°）", "纬度（°）", "海拔（m）", "本次需保障人口（人）"]].copy()
    df.columns = ["V", "x", "y", "h", "d"]

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["h"] = pd.to_numeric(df["h"], errors="coerce")
    df["d"] = pd.to_numeric(df["d"], errors="coerce")

    df = df.dropna(subset=["V"])
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"清洗完成，共 {len(df)} 条记录，已保存至: {output_path}")
    print(df.to_string())


if __name__ == "__main__":
    clean_service_area()