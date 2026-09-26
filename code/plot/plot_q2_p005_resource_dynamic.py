from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 路径
# ============================================================

THIS_FILE = Path(__file__).resolve()
CODE_ROOT = THIS_FILE.parents[1]

P005_DIR = CODE_ROOT / "data" / "q2_compact_moead" / "P005"

SCHEDULE_FILE = P005_DIR / "schedule.csv"
VALIDATION_FILE = P005_DIR / "validation.json"

OUTPUT_DIR = CODE_ROOT / "figures" / "q2"

OUTPUT_PDF = OUTPUT_DIR / "q2_p005_resource_dynamic.pdf"
OUTPUT_PNG = OUTPUT_DIR / "q2_p005_resource_dynamic.png"


# ============================================================
# 当前题设资源配置
# A: 4 UAV / 6 batteries
# B: 2 UAV / 4 batteries
# C: 2 UAV / 4 batteries
# ============================================================

TOTAL_UAVS = 8
TOTAL_BATTERIES = 14


# ============================================================
# 绘图设置
# 与当前 code/plot 下脚本保持基本一致
# ============================================================

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )


# ============================================================
# 数据读取
# ============================================================

def load_schedule() -> pd.DataFrame:
    if not SCHEDULE_FILE.exists():
        raise FileNotFoundError(
            f"未找到 P005 排程文件：{SCHEDULE_FILE}"
        )

    df = pd.read_csv(SCHEDULE_FILE, encoding="utf-8-sig")

    required = {
        "task_id",
        "uav_id",
        "uav_type",
        "battery_id",
        "start_time_s",
        "end_time_s",
        "charge_start_s",
        "charge_end_s",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"schedule.csv 缺少字段：{sorted(missing)}"
        )

    numeric_cols = [
        "start_time_s",
        "end_time_s",
        "charge_start_s",
        "charge_end_s",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def load_validation() -> dict:
    if not VALIDATION_FILE.exists():
        return {}

    with open(VALIDATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 区间统计
# ============================================================

def interval_count(
    df: pd.DataFrame,
    times: np.ndarray,
    start_col: str,
    end_col: str,
) -> np.ndarray:
    """
    统计每个时刻处于 [start, end) 区间内的记录数量。
    """
    result = np.zeros(len(times), dtype=int)

    starts = df[start_col].to_numpy(dtype=float)
    ends = df[end_col].to_numpy(dtype=float)

    for i, t in enumerate(times):
        result[i] = np.sum(
            (starts <= t) & (t < ends)
        )

    return result


def build_event_times(
    df: pd.DataFrame,
    plot_end: float,
) -> np.ndarray:
    """
    使用所有任务开始/结束与充电开始/结束时刻构造事件时间轴，
    从而得到精确阶梯曲线，而不是固定时间步长采样。
    """

    values = {
        0.0,
        float(plot_end),
    }

    for col in [
        "start_time_s",
        "end_time_s",
        "charge_start_s",
        "charge_end_s",
    ]:
        values.update(
            df[col].astype(float).tolist()
        )

    return np.array(sorted(values), dtype=float)


# ============================================================
# 数据检查
# ============================================================

def validate_schedule(
    df: pd.DataFrame,
    validation: dict,
) -> tuple[float, float]:

    sorties = len(df)

    used_uavs = df["uav_id"].nunique()
    used_batteries = df["battery_id"].nunique()

    cmax = float(df["end_time_s"].max())
    battery_ready_time = float(
        df["charge_end_s"].max()
    )

    print("=" * 60)
    print("P005 dynamic resource plot")
    print("=" * 60)

    print(f"sorties              : {sorties}")
    print(f"used UAVs            : {used_uavs}")
    print(f"used batteries       : {used_batteries}")
    print(f"Cmax                 : {cmax:.6f} s")
    print(
        f"last battery recovery: "
        f"{battery_ready_time:.6f} s"
    )
    print(
        f"recovery lag         : "
        f"{battery_ready_time - cmax:.6f} s"
    )

    if used_uavs > TOTAL_UAVS:
        raise ValueError(
            f"实际使用 UAV 数 {used_uavs} "
            f"超过库存 {TOTAL_UAVS}"
        )

    if used_batteries > TOTAL_BATTERIES:
        raise ValueError(
            f"实际使用电池数 {used_batteries} "
            f"超过库存 {TOTAL_BATTERIES}"
        )

    if validation:
        if not validation.get("all_pass", False):
            raise ValueError(
                "P005 validation.json 未通过全部检查"
            )

        validation_cmax = (
            validation
            .get("objectives", {})
            .get("Cmax")
        )

        if validation_cmax is not None:
            if not np.isclose(
                cmax,
                float(validation_cmax),
                atol=1e-6,
            ):
                raise ValueError(
                    "schedule.csv 与 validation.json "
                    "中的 Cmax 不一致"
                )

        print("validation           : PASS")

    return cmax, battery_ready_time


# ============================================================
# 作图
# ============================================================

def plot_resource_dynamic(
    df: pd.DataFrame,
    cmax: float,
    battery_ready_time: float,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 让 8551.43 s 后仍留少量空白区域
    plot_end = (
        np.ceil(battery_ready_time / 500.0)
        * 500.0
    )

    times = build_event_times(
        df,
        plot_end=plot_end,
    )

    # --------------------------------------------------------
    # (a) UAV 并发作业数量
    # --------------------------------------------------------

    uav_busy = interval_count(
        df,
        times,
        "start_time_s",
        "end_time_s",
    )

    # --------------------------------------------------------
    # (b) 电池状态
    #
    # task:
    #   随无人机执行任务
    #
    # charging:
    #   返航后至充满
    #
    # idle:
    #   库存中当前未参与任务且未充电
    # --------------------------------------------------------

    battery_task = interval_count(
        df,
        times,
        "start_time_s",
        "end_time_s",
    )

    battery_charging = interval_count(
        df,
        times,
        "charge_start_s",
        "charge_end_s",
    )

    battery_idle = (
        TOTAL_BATTERIES
        - battery_task
        - battery_charging
    )

    if np.any(battery_idle < 0):
        bad_idx = np.where(
            battery_idle < 0
        )[0][0]

        raise ValueError(
            "电池状态统计出现负空闲数量："
            f"t={times[bad_idx]:.3f}"
        )

    # --------------------------------------------------------
    # 画布
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.6),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.0, 1.08],
            "hspace": 0.30,
        },
    )

    ax1, ax2 = axes

    # 柔和配色，打印时也有明度差异
    task_color = "#5B9BD5"
    charge_color = "#F4A261"
    idle_color = "#D9D9D9"

    end_color = "#B03A2E"
    ready_color = "#3567A8"

    # ========================================================
    # Panel (a)
    # ========================================================

    ax1.step(
        times,
        uav_busy,
        where="post",
        linewidth=1.5,
        color=task_color,
    )

    ax1.fill_between(
        times,
        0,
        uav_busy,
        step="post",
        alpha=0.32,
        color=task_color,
    )

    ax1.axvline(
        cmax,
        color=end_color,
        linestyle="--",
        linewidth=1.0,
        zorder=5,
    )

    ax1.annotate(
        "运输任务结束\n"
        f"{cmax:.2f} s",
        xy=(cmax, 0),
        xytext=(
            cmax + 500,
            TOTAL_UAVS - 0.5,
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color=end_color,
    )

    ax1.set_ylim(
        0,
        TOTAL_UAVS + 1.15,
    )

    ax1.set_yticks(
        np.arange(
            0,
            TOTAL_UAVS + 1,
            2,
        )
    )

    ax1.set_ylabel(
        "并发作业 UAV 数量 / 架"
    )

    ax1.set_title(
        "(a) 运输无人机并发作业数量",
        loc="left",
        pad=8,
    )

    ax1.grid(
        True,
        linestyle="--",
        linewidth=0.45,
        alpha=0.35,
    )

    ax1.set_axisbelow(True)

    # ========================================================
    # Panel (b)
    # ========================================================

    task_upper = battery_task

    charge_upper = (
        battery_task
        + battery_charging
    )

    total_upper = np.full(
        len(times),
        TOTAL_BATTERIES,
        dtype=float,
    )

    # 任务占用
    ax2.fill_between(
        times,
        0,
        task_upper,
        step="post",
        color=task_color,
        alpha=0.82,
        label="任务占用",
    )

    # 充电
    ax2.fill_between(
        times,
        task_upper,
        charge_upper,
        step="post",
        color=charge_color,
        alpha=0.55,
        hatch="//",
        edgecolor=charge_color,
        linewidth=0.45,
        label="充电状态",
    )

    # 空闲
    ax2.fill_between(
        times,
        charge_upper,
        total_upper,
        step="post",
        color=idle_color,
        alpha=0.65,
        label="空闲状态",
    )

    # 边界线
    ax2.step(
        times,
        task_upper,
        where="post",
        linewidth=0.85,
        color=task_color,
    )

    ax2.step(
        times,
        charge_upper,
        where="post",
        linewidth=0.85,
        color=charge_color,
    )

    # 运输结束
    ax2.axvline(
        cmax,
        color=end_color,
        linestyle="--",
        linewidth=1.0,
        zorder=5,
    )

    # 全部电池恢复
    ax2.axvline(
        battery_ready_time,
        color=ready_color,
        linestyle="--",
        linewidth=1.0,
        zorder=5,
    )

    ax2.annotate(
        "运输任务结束\n"
        f"{cmax:.2f} s",
        xy=(cmax, TOTAL_BATTERIES),
        xytext=(
            cmax + 500,
            TOTAL_BATTERIES - 1,
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color=end_color,
    )

    ax2.annotate(
        "全部电池充满\n"
        f"{battery_ready_time:.2f} s",
        xy=(
            battery_ready_time,
            TOTAL_BATTERIES,
        ),
        xytext=(
            battery_ready_time - 500,
            TOTAL_BATTERIES - 1,
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color=ready_color,
    )

    ax2.set_ylim(
        0,
        TOTAL_BATTERIES + 1.65,
    )

    ax2.set_yticks(
        np.arange(
            0,
            TOTAL_BATTERIES + 1,
            2,
        )
    )

    ax2.set_ylabel(
        "电池数量 / 块"
    )

    ax2.set_xlabel(
        "时间 / s"
    )

    ax2.set_title(
        "(b) 共享电池状态数量",
        loc="left",
        pad=8,
    )

    ax2.grid(
        True,
        linestyle="--",
        linewidth=0.45,
        alpha=0.35,
    )

    ax2.set_axisbelow(True)

    # --------------------------------------------------------
    # 横轴
    # --------------------------------------------------------

    ax2.set_xlim(
        0,
        plot_end,
    )

    tick_step = 1000

    ax2.set_xticks(
        np.arange(
            0,
            plot_end + 1,
            tick_step,
        )
    )

    # --------------------------------------------------------
    # 图例
    # --------------------------------------------------------

    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.25,
        ),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=2.0,
    )

    # --------------------------------------------------------
    # 边框统一
    # --------------------------------------------------------

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.92,
        bottom=0.18,
    )

    # 不在图片内部写"图18"，交给 LaTeX caption
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"PDF saved to: {OUTPUT_PDF}")
    print(f"PNG saved to: {OUTPUT_PNG}")


# ============================================================
# main
# ============================================================

def main() -> None:
    configure_matplotlib()

    df = load_schedule()

    validation = load_validation()

    cmax, battery_ready_time = (
        validate_schedule(
            df,
            validation,
        )
    )

    plot_resource_dynamic(
        df,
        cmax,
        battery_ready_time,
    )


if __name__ == "__main__":
    main()