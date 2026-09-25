from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio


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

# 统一浅蓝色高程色带：低处极浅接近白，高处较深蓝但不压暗
BLUE_TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "blue_terrain_soft",
    [
        "#C8D9EE",
        "#B3CDE5",
        "#9BBFD9",
        "#81AECD",
        "#6A9DC1",
        "#548BB3",
        "#3D7AA5",
        "#2E6DA0",
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
    """读取 DEM。"""

    src = rasterio.open(DEM_FILE)

    if "4326" not in str(src.crs):
        raise ValueError(
            f"DEM CRS = {src.crs}，当前脚本要求 EPSG:4326"
        )

    dem = src.read(1, masked=True)

    # 转换成 NaN 数组，方便 matplotlib 处理
    dem_array = dem.filled(np.nan).astype(float)

    bounds = src.bounds

    return src, dem_array, bounds


def load_nodes() -> pd.DataFrame:
    """读取 O01 与 15 个服务区。"""

    nodes = pd.read_csv(NODE_FILE)

    required = {"V", "x", "y", "h"}

    missing = required - set(nodes.columns)

    if missing:
        raise ValueError(f"节点文件缺少字段: {missing}")

    return nodes


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

    line_color = "#D4380D"

    if CONNECT_MODE == "hub":

        base = nodes.loc[nodes["V"] == "O01"].iloc[0]

        services = nodes.loc[nodes["V"] != "O01"]

        for _, row in services.iterrows():

            ax.plot(
                [base["x"], row["x"]],
                [base["y"], row["y"]],
                linestyle=(0, (4, 3)),
                linewidth=0.75,
                color=line_color,
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
            linewidth=0.8,
            color=line_color,
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


def main() -> None:

    configure_matplotlib()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    nodes = load_nodes()

    src, dem, bounds = load_dem()

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

        im = ax.imshow(
            dem,
            extent=[
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top,
            ],
            origin="upper",
            cmap=BLUE_TERRAIN_CMAP,
            vmin=vmin,
            vmax=vmax,
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
            alpha=0.06,
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
                color="#D4380D",
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