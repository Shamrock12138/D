from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import (
    LightSource,
    LinearSegmentedColormap,
    TwoSlopeNorm,
)
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
from rasterio.coords import BoundingBox
from rasterio.windows import from_bounds


# ============================================================
# 路径
# ============================================================

THIS_FILE = Path(__file__).resolve()

# .../D/code
CODE_ROOT = THIS_FILE.parents[1]

# .../D
REPO_ROOT = CODE_ROOT.parent

NODE_FILE = CODE_ROOT / "data" / "服务区数据.csv"

DEM_FILE = (
    REPO_ROOT
    / "数据"
    / "镇龙乡地理空间数据"
    / "镇龙乡及周边地理数据"
    / "数字高程模型数据（DEM）"
    / "镇龙乡及周边30米DEM.tif"
)

OUTPUT_DIR = CODE_ROOT / "figures" / "terrain"


# ============================================================
# 绘图参数
# ============================================================

# 推荐：调度中心到所有服务区的放射状虚线
CONNECT_MODE = "hub"

# 如果你希望 O01-S001-S002-...-S015 按编号依次连接，
# 改为：
# CONNECT_MODE = "sequence"

# ============================================================
# 地图显示范围（二维 / 三维共用）
# ============================================================

LON_MIN = 109.10
LON_MAX = 109.35

LAT_MIN = 22.95
LAT_MAX = 23.17

# ============================================================
# 统一地形高程色带
# 低海拔：蓝
# 中海拔：白
# 高海拔：红
# 二维与三维必须共用该 cmap 和 norm
# ============================================================

TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "terrain_blue_white_red",
    [
        (0.00, "#2B6CA8"),  # 低海拔：深蓝
        (0.20, "#5599CC"),
        (0.38, "#A6D0EA"),
        (0.50, "#FFFFFF"),  # 400 m：白色
        (0.62, "#F0B8B8"),
        (0.80, "#E06060"),
        (1.00, "#B0303A"),  # 高海拔：深红
    ],
    N=256,
)


def configure_matplotlib() -> None:
    """论文绘图字体及基础样式。"""

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
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def load_dem():
    """读取指定经纬度范围内的 DEM。"""

    src = rasterio.open(DEM_FILE)

    if "4326" not in str(src.crs):
        raise ValueError(
            f"DEM CRS = {src.crs}，当前脚本要求 EPSG:4326"
        )

    # 根据指定经纬度创建栅格窗口
    window = from_bounds(
        left=LON_MIN,
        bottom=LAT_MIN,
        right=LON_MAX,
        top=LAT_MAX,
        transform=src.transform,
    )

    # 对齐到实际 DEM 像元
    window = window.round_offsets().round_lengths()

    dem = src.read(
        1,
        window=window,
        masked=True,
    )

    dem_array = dem.filled(np.nan).astype(float)

    # 裁剪后 DEM 对应的 transform
    dem_transform = src.window_transform(window)

    # 裁剪后实际边界
    bounds = BoundingBox(
        *rasterio.windows.bounds(
            window,
            src.transform,
        )
    )

    return src, dem_array, bounds, dem_transform


def load_nodes() -> pd.DataFrame:
    """读取指定地图范围内的节点。"""

    nodes = pd.read_csv(NODE_FILE)

    required = {"V", "x", "y", "h"}

    missing = required - set(nodes.columns)

    if missing:
        raise ValueError(f"节点文件缺少字段: {missing}")

    nodes = nodes.loc[
        (nodes["x"] >= LON_MIN)
        & (nodes["x"] <= LON_MAX)
        & (nodes["y"] >= LAT_MIN)
        & (nodes["y"] <= LAT_MAX)
    ].copy()

    return nodes


def build_terrain_norm(dem: np.ndarray) -> TwoSlopeNorm:
    """
    建立二维/三维共用的高程颜色归一化。

    TERRAIN_CMAP 以 0.5 为白色中心，TwoSlopeNorm 将
    vcenter = 400 m 映射到 0.5，因此：
        低于 200 m → 蓝
        400 m → 白
        高于 400 m → 逐渐变红
    """

    valid = dem[np.isfinite(dem)]

    vmin = float(np.min(valid))
    vmax = float(np.max(valid))

    return TwoSlopeNorm(
        vmin=vmin,
        vcenter=400,
        vmax=vmax,
    )


def calculate_hillshade(
    dem: np.ndarray,
    src: rasterio.io.DatasetReader,
):
    """
    根据 DEM 构建轻度 hillshade。

    DEM 是经纬度坐标，因此把像元尺寸近似换算为米，
    避免直接使用 degree 造成阴影夸张。
    """

    mean_lat = (src.bounds.top + src.bounds.bottom) / 2.0

    dx_deg = abs(src.transform.a)
    dy_deg = abs(src.transform.e)

    dx_m = dx_deg * 111_320.0 * np.cos(np.radians(mean_lat))
    dy_m = dy_deg * 110_540.0

    valid_dem = dem.copy()

    # LightSource 不喜欢 NaN，使用中位数暂时填充；
    # 最终 NoData 区域仍通过 mask 控制。
    median_elevation = np.nanmedian(valid_dem)

    filled_dem = np.where(
        np.isfinite(valid_dem),
        valid_dem,
        median_elevation,
    )

    light = LightSource(
        azdeg=315,
        altdeg=45,
    )

    hillshade = light.hillshade(
        filled_dem,
        vert_exag=1.0,
        dx=dx_m,
        dy=dy_m,
    )

    hillshade[~np.isfinite(dem)] = np.nan

    return hillshade


def draw_connections(
    ax: plt.Axes,
    nodes: pd.DataFrame,
) -> None:
    """绘制节点间虚线。"""

    if CONNECT_MODE == "hub":

        base = nodes.loc[nodes["V"] == "O01"].iloc[0]

        services = nodes.loc[nodes["V"] != "O01"]

        for _, row in services.iterrows():

            ax.plot(
                [base["x"], row["x"]],
                [base["y"], row["y"]],
                linestyle=(0, (4, 3)),
                linewidth=1.2,
                color="#000000",
                alpha=0.62,
                zorder=4,
            )

    elif CONNECT_MODE == "sequence":

        ordered = nodes.copy()

        def sort_key(v: str):
            if v == "O01":
                return 0
            return int(v[1:])

        ordered["_order"] = ordered["V"].map(sort_key)
        ordered = ordered.sort_values("_order")

        ax.plot(
            ordered["x"],
            ordered["y"],
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            color="#E87E1A",
            alpha=0.68,
            zorder=4,
        )

    else:
        raise ValueError(
            f"未知 CONNECT_MODE: {CONNECT_MODE}"
        )


def draw_nodes(
    ax: plt.Axes,
    nodes: pd.DataFrame,
) -> None:
    """绘制调度中心和服务区。"""

    base = nodes.loc[nodes["V"] == "O01"]

    services = nodes.loc[nodes["V"] != "O01"]

    # --------------------------------------------------------
    # 服务区
    # --------------------------------------------------------

    ax.scatter(
        services["x"],
        services["y"],
        s=30,
        marker="o",
        facecolor="#4E79A7",
        edgecolor="white",
        linewidth=0.75,
        zorder=7,
    )

    # --------------------------------------------------------
    # 调度中心
    # --------------------------------------------------------

    ax.scatter(
        base["x"],
        base["y"],
        s=105,
        marker="*",
        facecolor="#CC5A6A",
        edgecolor="white",
        linewidth=0.9,
        zorder=8,
    )

    # --------------------------------------------------------
    # 标签
    # --------------------------------------------------------

    for _, row in nodes.iterrows():

        node_id = row["V"]

        if node_id == "O01":
            offset = (7, -1)
            weight = "bold"
            size = 9
        else:
            # 交替偏移，稍微减少标签重叠
            index = int(node_id[1:])

            if index % 2 == 0:
                offset = (4, 5)
            else:
                offset = (4, -8)

            weight = "normal"
            size = 7.6

        text = ax.annotate(
            node_id,
            xy=(row["x"], row["y"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=size,
            fontweight=weight,
            color="#202124",
            zorder=10,
        )

        # 白色 halo，在复杂 DEM 背景上保持文字清晰
        text.set_path_effects(
            [
                pe.withStroke(
                    linewidth=2.2,
                    foreground="white",
                )
            ]
        )


def add_north_arrow(ax: plt.Axes) -> None:
    """添加简洁指北针。"""

    ax.annotate(
        "N",
        xy=(0.945, 0.91),
        xytext=(0.945, 0.82),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.0,
            color="#202124",
        ),
        zorder=20,
    )


def add_scale_bar(
    ax: plt.Axes,
    mean_lat: float,
) -> None:
    """添加约 1 km 比例尺。"""

    # 纬度 mean_lat 下，1 km 对应的经度差
    km_lon = 1.0 / (
        111.32 * np.cos(np.radians(mean_lat))
    )

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_start = xmin + 0.060 * (xmax - xmin)
    y = ymin + 0.060 * (ymax - ymin)

    x_end = x_start + km_lon

    ax.plot(
        [x_start, x_end],
        [y, y],
        color="#202124",
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=20,
    )

    # 两端短竖线
    tick_h = 0.006 * (ymax - ymin)

    ax.plot(
        [x_start, x_start],
        [y - tick_h, y + tick_h],
        color="#202124",
        linewidth=1.0,
        zorder=20,
    )

    ax.plot(
        [x_end, x_end],
        [y - tick_h, y + tick_h],
        color="#202124",
        linewidth=1.0,
        zorder=20,
    )

    ax.text(
        (x_start + x_end) / 2,
        y + 0.010 * (ymax - ymin),
        "1 km",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#202124",
        zorder=20,
        path_effects=[
            pe.withStroke(
                linewidth=2,
                foreground="white",
            )
        ],
    )


def preview_and_save_3d(
    fig,
    ax,
    output_dir: Path,
    save_callback=None,
) -> None:
    """
    显示三维交互窗口。

    操作：
      鼠标左键拖动：旋转
      鼠标滚轮：缩放
      S：调用 save_callback 保存高分辨率版本
      Q：关闭窗口
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_file = output_dir / "terrain_3d_bwr.pdf"
    png_file = output_dir / "terrain_3d_bwr.png"

    def on_key(event):

        if event.key is None:
            return

        key = event.key.lower()

        if key == "s":

            elev = float(ax.elev)
            azim = float(ax.azim)
            roll = float(getattr(ax, "roll", 0.0))

            if save_callback is not None:
                save_callback(elev, azim)
            else:
                # 无回调时直接保存当前预览窗口
                fig.canvas.draw()
                fig.savefig(pdf_file, dpi=400, bbox_inches="tight")
                fig.savefig(png_file, dpi=600, bbox_inches="tight")

            print()
            print("=" * 60)
            print("保存当前 3D 视角")
            print(
                f"elev = {elev:.2f}, "
                f"azim = {azim:.2f}, "
                f"roll = {roll:.2f}"
            )
            print(f"PDF: {pdf_file}")
            print(f"PNG: {png_file}")
            print("=" * 60)

        elif key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    print()
    print("=" * 60)
    print("3D 地形图交互预览")
    print("=" * 60)
    print("鼠标拖动 : 调整观察角度")
    print("鼠标滚轮 : 缩放")
    print("按 S      : 保存当前视角（高分辨率）")
    print("按 Q      : 关闭窗口")
    print("=" * 60)

    # 必须最后调用
    plt.show()


def plot_terrain_3d(
    src,
    dem,
    nodes,
    bounds,
    dem_transform,
    terrain_norm,
) -> None:
    """绘制蓝-白-红统一高程编码的三维 DEM。"""

    nrows, ncols = dem.shape

    PREVIEW_TARGET = 300

    # DEM 下采样，避免 plot_surface 面片过多
    step = max(
        1,
        int(
            np.ceil(
                max(nrows, ncols)
                / PREVIEW_TARGET
            )
        ),
    )

    rows = np.arange(0, nrows, step)
    cols = np.arange(0, ncols, step)

    Z = dem[np.ix_(rows, cols)]

    # 每个栅格中心的经纬度
    x_all = (
        dem_transform.c
        + (np.arange(ncols) + 0.5)
        * dem_transform.a
    )

    y_all = (
        dem_transform.f
        + (np.arange(nrows) + 0.5)
        * dem_transform.e
    )

    x = x_all[cols]
    y = y_all[rows]

    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8.0, 6.3))
    ax = fig.add_subplot(111, projection="3d")

    # ========================================================
    # DEM（预览分辨率）
    # ========================================================

    ax.plot_surface(
        X, Y, Z,
        cmap=TERRAIN_CMAP,
        norm=terrain_norm,
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
        rasterized=True,
        alpha=1.0,
    )

    # ========================================================
    # 节点
    # ========================================================

    base = nodes.loc[nodes["V"] == "O01"].iloc[0]
    services = nodes.loc[nodes["V"] != "O01"]

    ax.scatter(
        services["x"], services["y"], services["h"] + 15,
        s=28, marker="o",
        facecolor="#4E79A7", edgecolor="white", linewidth=0.8,
        depthshade=False, zorder=10,
    )

    ax.scatter(
        [base["x"]], [base["y"]], [float(base["h"]) + 20],
        s=115, marker="*",
        facecolor="#CC5A6A", edgecolor="white", linewidth=1.0,
        depthshade=False, zorder=11,
    )

    # ========================================================
    # O01 到服务区虚线
    # ========================================================

    for _, row in services.iterrows():
        n = 100
        lons = np.linspace(float(base["x"]), float(row["x"]), n)
        lats = np.linspace(float(base["y"]), float(row["y"]), n)

        samples = np.array([float(v[0]) for v in src.sample(zip(lons, lats))])
        valid_mask = np.isfinite(samples)
        if src.nodata is not None:
            valid_mask &= samples != src.nodata
        if not np.all(valid_mask):
            idx = np.arange(n)
            if np.sum(valid_mask) >= 2:
                samples[~valid_mask] = np.interp(
                    idx[~valid_mask], idx[valid_mask], samples[valid_mask],
                )

        ax.plot(
            lons, lats, samples + 10,
            linestyle=(0, (4, 3)), linewidth=0.85,
            color="#E87E1A", alpha=0.72, zorder=6,
        )

    # ========================================================
    # 标签
    # ========================================================

    for _, row in nodes.iterrows():
        ax.text(
            float(row["x"]), float(row["y"]), float(row["h"]) + 28,
            f" {row['V']}", fontsize=7, color="#202124",
            fontweight="bold" if row["V"] == "O01" else "normal",
        )

    # ========================================================
    # 坐标
    # ========================================================

    ax.set_xlabel("经度 / °E")
    ax.set_ylabel("纬度 / °N")
    ax.set_zlabel("地面高程 / m")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.view_init(elev=30, azim=-60)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # ========================================================
    # 与二维完全相同的 Colorbar
    # ========================================================

    from matplotlib.cm import ScalarMappable

    sm = ScalarMappable(norm=terrain_norm, cmap=TERRAIN_CMAP)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, shrink=0.64, pad=0.08, aspect=24)
    cbar.set_label("地面高程 / m", labelpad=7)

    # ========================================================
    # 高分辨率保存回调
    # ========================================================

    SAVE_TARGET = 600

    def save_high_res(elev: float, azim: float) -> None:
        # 构建高分辨率网格
        step_save = max(
            1,
            int(np.ceil(max(nrows, ncols) / SAVE_TARGET)),
        )
        rows_s = np.arange(0, nrows, step_save)
        cols_s = np.arange(0, ncols, step_save)

        Z_s = dem[np.ix_(rows_s, cols_s)]

        x_s = x_all[cols_s]
        y_s = y_all[rows_s]
        X_s, Y_s = np.meshgrid(x_s, y_s)

        fig_s = plt.figure(figsize=(8.0, 6.3))
        ax_s = fig_s.add_subplot(111, projection="3d")

        ax_s.plot_surface(
            X_s, Y_s, Z_s,
            cmap=TERRAIN_CMAP, norm=terrain_norm,
            rstride=1, cstride=1,
            linewidth=0, edgecolor="none",
            antialiased=True, shade=False, rasterized=True, alpha=1.0,
        )

        # 节点
        ax_s.scatter(
            services["x"], services["y"], services["h"] + 15,
            s=28, marker="o",
            facecolor="#4E79A7", edgecolor="white", linewidth=0.8,
            depthshade=False, zorder=10,
        )
        ax_s.scatter(
            [base["x"]], [base["y"]], [float(base["h"]) + 20],
            s=115, marker="*",
            facecolor="#CC5A6A", edgecolor="white", linewidth=1.0,
            depthshade=False, zorder=11,
        )

        # 虚线
        for _, row_sv in services.iterrows():
            n = 100
            lons = np.linspace(float(base["x"]), float(row_sv["x"]), n)
            lats = np.linspace(float(base["y"]), float(row_sv["y"]), n)
            samples = np.array([float(v[0]) for v in src.sample(zip(lons, lats))])
            valid_mask = np.isfinite(samples)
            if src.nodata is not None:
                valid_mask &= samples != src.nodata
            if not np.all(valid_mask):
                idx = np.arange(n)
                if np.sum(valid_mask) >= 2:
                    samples[~valid_mask] = np.interp(
                        idx[~valid_mask], idx[valid_mask], samples[valid_mask],
                    )
            ax_s.plot(
                lons, lats, samples + 10,
                linestyle=(0, (4, 3)), linewidth=0.85,
                color="#E87E1A", alpha=0.72, zorder=6,
            )

        # 标签
        for _, row_sv in nodes.iterrows():
            ax_s.text(
                float(row_sv["x"]), float(row_sv["y"]), float(row_sv["h"]) + 28,
                f" {row_sv['V']}", fontsize=7, color="#202124",
                fontweight="bold" if row_sv["V"] == "O01" else "normal",
            )

        ax_s.set_xlabel("经度 / °E")
        ax_s.set_ylabel("纬度 / °N")
        ax_s.set_zlabel("地面高程 / m")
        ax_s.set_xlim(LON_MIN, LON_MAX)
        ax_s.set_ylim(LAT_MIN, LAT_MAX)
        ax_s.view_init(elev=elev, azim=azim)
        ax_s.xaxis.pane.fill = False
        ax_s.yaxis.pane.fill = False
        ax_s.zaxis.pane.fill = False

        sm_s = ScalarMappable(norm=terrain_norm, cmap=TERRAIN_CMAP)
        sm_s.set_array([])
        cbar_s = fig_s.colorbar(sm_s, ax=ax_s, shrink=0.64, pad=0.08, aspect=24)
        cbar_s.set_label("地面高程 / m", labelpad=7)

        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_s.savefig(output_dir / "terrain_3d_bwr.pdf", dpi=400, bbox_inches="tight")
        fig_s.savefig(output_dir / "terrain_3d_bwr.png", dpi=600, bbox_inches="tight")
        plt.close(fig_s)

    # ========================================================
    # 交互预览 + 手动保存
    # ========================================================

    preview_and_save_3d(fig, ax, OUTPUT_DIR, save_high_res)


def main() -> None:

    configure_matplotlib()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    nodes = load_nodes()

    src, dem, bounds, dem_transform = load_dem()

    try:

        # ====================================================
        # 画布
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(7.2, 6.2),
        )

        # ====================================================
        # DEM 高程热力图
        # ====================================================

        valid = dem[np.isfinite(dem)]

        vmin = float(np.min(valid))
        vmax = float(np.max(valid))

        terrain_norm = build_terrain_norm(dem)

        im = ax.imshow(
            dem,
            extent=[
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top,
            ],
            origin="upper",
            cmap=TERRAIN_CMAP,
            norm=terrain_norm,
            interpolation="bilinear",
            zorder=1,
        )

        # ====================================================
        # 地形阴影
        # ====================================================

        hillshade = calculate_hillshade(
            dem,
            src,
        )

        ax.imshow(
            hillshade,
            extent=[
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top,
            ],
            origin="upper",
            cmap="gray",
            alpha=0.035,
            interpolation="bilinear",
            zorder=2,
        )

        # ====================================================
        # 等高线
        # ====================================================

        contour_step = 100

        contour_min = (
            np.floor(vmin / contour_step)
            * contour_step
        )

        contour_max = (
            np.ceil(vmax / contour_step)
            * contour_step
        )

        levels = np.arange(
            contour_min,
            contour_max + contour_step,
            contour_step,
        )

        # raster 第 0 行位于北部，因此翻转 DEM
        dem_contour = np.flipud(dem)

        x_grid = np.linspace(
            bounds.left,
            bounds.right,
            dem.shape[1],
        )

        y_grid = np.linspace(
            bounds.bottom,
            bounds.top,
            dem.shape[0],
        )

        ax.contour(
            x_grid,
            y_grid,
            dem_contour,
            levels=levels,
            colors="#3F4650",
            linewidths=0.35,
            alpha=0.30,
            zorder=3,
        )

        # ====================================================
        # 节点虚线
        # ====================================================

        draw_connections(
            ax,
            nodes,
        )

        # ====================================================
        # 节点
        # ====================================================

        draw_nodes(
            ax,
            nodes,
        )

        # ====================================================
        # Colorbar
        # ====================================================

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.034,
            pad=0.025,
        )

        cbar.set_label(
            "地面高程 / m",
            rotation=90,
            labelpad=8,
        )

        cbar.outline.set_linewidth(0.6)

        cbar.ax.tick_params(
            width=0.6,
            length=3,
            labelsize=8,
        )

        # ====================================================
        # 图例
        # ====================================================

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markersize=10,
                markerfacecolor="#CC5A6A",
                markeredgecolor="white",
                label="调度中心 O01",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=6,
                markerfacecolor="#4E79A7",
                markeredgecolor="white",
                label="服务区",
            ),
            Line2D(
                [0],
                [0],
                linestyle=(0, (4, 3)),
                linewidth=0.9,
                color="#E87E1A",
                label="节点连接",
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            framealpha=0.90,
            edgecolor="#D0D0D0",
            fancybox=False,
        )

        # ====================================================
        # 坐标轴
        # ====================================================

        ax.set_xlabel("经度 / °E")
        ax.set_ylabel("纬度 / °N")

        ax.set_xlim(
            LON_MIN,
            LON_MAX,
        )

        ax.set_ylim(
            LAT_MIN,
            LAT_MAX,
        )

        # 不设置总标题，更适合直接放论文
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.35,
            alpha=0.25,
        )

        # 经纬度地图按实际纬度比例校正
        mean_lat = (
            bounds.top + bounds.bottom
        ) / 2.0

        ax.set_aspect(
            1.0 / np.cos(np.radians(mean_lat))
        )

        # ====================================================
        # 指北针和比例尺
        # ====================================================

        add_north_arrow(ax)

        add_scale_bar(
            ax,
            mean_lat,
        )

        # ====================================================
        # 输出
        # ====================================================

        fig.tight_layout()

        pdf_file = (
            OUTPUT_DIR
            / "terrain_heatmap_blue.pdf"
        )

        svg_file = (
            OUTPUT_DIR
            / "terrain_heatmap_blue.svg"
        )

        png_file = (
            OUTPUT_DIR
            / "terrain_heatmap_blue.png"
        )

        fig.savefig(
            pdf_file,
            bbox_inches="tight",
        )

        fig.savefig(
            svg_file,
            bbox_inches="tight",
        )

        fig.savefig(
            png_file,
            dpi=600,
            bbox_inches="tight",
        )

        plt.close(fig)

        print("地形图绘制完成：")
        print(f"  PDF: {pdf_file}")
        print(f"  SVG: {svg_file}")
        print(f"  PNG: {png_file}")

    finally:
        src.close()


if __name__ == "__main__":
    main()