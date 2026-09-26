from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize, LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds






THIS_FILE = Path(__file__).resolve()
CODE_ROOT = THIS_FILE.parents[1]
REPO_ROOT = CODE_ROOT.parent

NODE_FILE = CODE_ROOT / "data" / "服务区数据.csv"
ROUTE_FILE = CODE_ROOT / "data" / "q2_lopt_routes.csv"

DEM_FILE = (
    REPO_ROOT
    / "数据"
    / "镇龙乡地理空间数据"
    / "镇龙乡及周边地理数据"
    / "数字高程模型数据（DEM）"
    / "镇龙乡及周边30米DEM.tif"
)

OUTPUT_DIR = CODE_ROOT / "figures" / "q2"






LON_MIN = 109.10
LON_MAX = 109.35
LAT_MIN = 22.95
LAT_MAX = 23.17







TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "terrain_blue_white_red",
    [
        (0.00, "#2B6CA8"),
        (0.20, "#5599CC"),
        (0.38, "#A6D0EA"),
        (0.50, "#FFFFFF"),
        (0.62, "#F0B8B8"),
        (0.80, "#E06060"),
        (1.00, "#B0303A"),
    ],
    N=256,
)

TYPE_COLORS = {
    "A": "#F0C050",
    "B": "#E89030",
    "C": "#C0501A",
}

BASE_COLOR = "#C44E52"
NODE_COLOR = "#3B6FB6"






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
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )






def load_dem():
    src = rasterio.open(DEM_FILE)

    if "4326" not in str(src.crs):
        raise ValueError(f"DEM CRS = {src.crs}，当前脚本要求 EPSG:4326")

    window = from_bounds(
        left=LON_MIN,
        bottom=LAT_MIN,
        right=LON_MAX,
        top=LAT_MAX,
        transform=src.transform,
    )
    window = window.round_offsets().round_lengths()

    dem = src.read(1, window=window, masked=True)
    dem_array = dem.filled(np.nan).astype(float)

    bounds = rasterio.windows.bounds(window, src.transform)
    dem_transform = src.window_transform(window)

    return src, dem_array, bounds, dem_transform


def load_nodes() -> pd.DataFrame:
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


def load_routes() -> pd.DataFrame:
    if not ROUTE_FILE.exists():
        raise FileNotFoundError(
            f"未找到路线文件：{ROUTE_FILE}\n"
            "请先创建 code/data/q2_lopt_routes.csv"
        )

    df = pd.read_csv(ROUTE_FILE)

    required = {"uav_type", "route", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"路线文件缺少字段: {missing}")

    df["uav_type"] = df["uav_type"].astype(str).str.strip()
    df["route"] = df["route"].astype(str).str.strip()
    df["count"] = df["count"].astype(int)

    return df






def build_terrain_norm(dem: np.ndarray) -> Normalize:
    valid = dem[np.isfinite(dem)]
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    return Normalize(vmin=vmin, vmax=vmax)


def calculate_hillshade(dem: np.ndarray, transform, bounds):
    mean_lat = (bounds[3] + bounds[1]) / 2.0

    dx_deg = abs(transform.a)
    dy_deg = abs(transform.e)

    dx_m = dx_deg * 111_320.0 * np.cos(np.radians(mean_lat))
    dy_m = dy_deg * 110_540.0

    median_elev = np.nanmedian(dem)
    filled_dem = np.where(np.isfinite(dem), dem, median_elev)

    light = LightSource(azdeg=315, altdeg=45)
    hillshade = light.hillshade(
        filled_dem,
        vert_exag=1.0,
        dx=dx_m,
        dy=dy_m,
    )
    hillshade[~np.isfinite(dem)] = np.nan
    return hillshade


def draw_dem_base(ax, dem, bounds, dem_transform, terrain_norm):
    xmin, ymin, xmax, ymax = bounds

    im = ax.imshow(
        dem,
        extent=[xmin, xmax, ymin, ymax],
        origin="upper",
        cmap=TERRAIN_CMAP,
        norm=terrain_norm,
        interpolation="bilinear",
        zorder=1,
    )

    hillshade = calculate_hillshade(dem, dem_transform, bounds)

    ax.imshow(
        hillshade,
        extent=[xmin, xmax, ymin, ymax],
        origin="upper",
        cmap="gray",
        alpha=0.04,
        interpolation="bilinear",
        zorder=2,
    )

    valid = dem[np.isfinite(dem)]
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))

    levels = np.arange(
        np.floor(vmin / 100) * 100,
        np.ceil(vmax / 100) * 100 + 100,
        100,
    )

    dem_contour = np.flipud(dem)
    x_grid = np.linspace(xmin, xmax, dem.shape[1])
    y_grid = np.linspace(ymin, ymax, dem.shape[0])

    ax.contour(
        x_grid,
        y_grid,
        dem_contour,
        levels=levels,
        colors="#4E4E4E",
        linewidths=0.32,
        alpha=0.22,
        zorder=3,
    )

    return im






def get_node_xy(nodes: pd.DataFrame, node_id: str) -> tuple[float, float]:
    row = nodes.loc[nodes["V"] == node_id]
    if row.empty:
        raise KeyError(f"未找到节点 {node_id}")
    row = row.iloc[0]
    return float(row["x"]), float(row["y"])


def draw_nodes(ax, nodes: pd.DataFrame) -> None:
    base = nodes.loc[nodes["V"] == "O01"]
    services = nodes.loc[nodes["V"] != "O01"]

    ax.scatter(
        services["x"],
        services["y"],
        s=32,
        marker="o",
        facecolor=NODE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=20,
    )

    ax.scatter(
        base["x"],
        base["y"],
        s=130,
        marker="*",
        facecolor=BASE_COLOR,
        edgecolor="white",
        linewidth=1.0,
        zorder=21,
    )

    for _, row in nodes.iterrows():
        node_id = row["V"]
        if node_id == "O01":
            offset = (7, -2)
            fsize = 9
            weight = "bold"
        else:
            idx = int(node_id[1:])
            offset = (4, 5) if idx % 2 == 0 else (4, -8)
            fsize = 7.5
            weight = "normal"

        text = ax.annotate(
            node_id,
            xy=(row["x"], row["y"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=fsize,
            fontweight=weight,
            color="#202124",
            zorder=30,
        )
        text.set_path_effects(
            [pe.withStroke(linewidth=2.0, foreground="white")]
        )


def parse_route(route: str) -> list[str]:
    seq = [x.strip() for x in route.split(">") if x.strip()]
    if len(seq) not in {3, 4}:
        raise ValueError(f"非法 route 表达：{route}")
    if seq[0] != "O01" or seq[-1] != "O01":
        raise ValueError(f"路线必须以 O01 开始并以 O01 结束：{route}")
    return seq


def add_line_with_halo(line):
    line.set_path_effects(
        [
            pe.Stroke(linewidth=line.get_linewidth() + 1.5, foreground="white"),
            pe.Normal(),
        ]
    )


def add_patch_with_halo(patch, lw: float):
    patch.set_path_effects(
        [
            pe.Stroke(linewidth=lw + 1.5, foreground="white"),
            pe.Normal(),
        ]
    )


def annotate_count(ax, p1, p2, text, color, dy=0.0):
    mx = 0.5 * (p1[0] + p2[0])
    my = 0.5 * (p1[1] + p2[1]) + dy

    t = ax.text(
        mx,
        my,
        text,
        fontsize=7.5,
        color=color,
        ha="center",
        va="center",
        zorder=50,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="none",
            edgecolor="none",
        ),
    )
    t.set_path_effects(
        [pe.withStroke(linewidth=1.5, foreground="white")]
    )


def draw_single_route(
    ax,
    nodes: pd.DataFrame,
    stop: str,
    uav_type: str,
    count: int,
):
    p0 = get_node_xy(nodes, "O01")
    p1 = get_node_xy(nodes, stop)

    color = TYPE_COLORS[uav_type]
    lw = 0.4 + 0.12 * max(count - 1, 0)

    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=color,
        linewidth=lw,
        alpha=0.6,
        zorder=8,
    )

    if count > 1:
        annotate_count(ax, p0, p1, f"×{count}", color="#404040", dy=0.0015)


def draw_arrow_segment(
    ax,
    p_start,
    p_end,
    color,
    lw,
    alpha,
    linestyle="-",
    rad=0.0,
    zorder=12,
):
    patch = FancyArrowPatch(
        p_start,
        p_end,
        arrowstyle="-|>",
        mutation_scale=8 + 1.8 * lw,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=6,
        shrinkB=6,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_double_route(
    ax,
    nodes: pd.DataFrame,
    stop_i: str,
    stop_j: str,
    uav_type: str,
    count: int,
):
    p0 = get_node_xy(nodes, "O01")
    pi = get_node_xy(nodes, stop_i)
    pj = get_node_xy(nodes, stop_j)

    color = TYPE_COLORS[uav_type]
    lw_outer = 0.5 + 0.1 * max(count - 1, 0)
    lw_inner = 0.8 + 0.15 * max(count - 1, 0)


    draw_arrow_segment(
        ax,
        p0,
        pi,
        color=color,
        lw=lw_outer,
        alpha=0.65,
        linestyle="-",
        rad=0.05,
        zorder=12,
    )


    draw_arrow_segment(
        ax,
        pi,
        pj,
        color=color,
        lw=lw_inner,
        alpha=0.95,
        linestyle="-",
        rad=0.0,
        zorder=15,
    )


    draw_arrow_segment(
        ax,
        pj,
        p0,
        color=color,
        lw=lw_outer,
        alpha=0.65,
        linestyle="-",
        rad=-0.05,
        zorder=12,
    )

    if count > 1:
        annotate_count(ax, pi, pj, f"×{count}", color="#303030", dy=0.0015)


def draw_routes(ax, nodes: pd.DataFrame, route_df: pd.DataFrame):
    for _, row in route_df.iterrows():
        uav_type = row["uav_type"]
        route = row["route"]
        count = int(row["count"])

        seq = parse_route(route)


        if len(seq) == 3:
            stop = seq[1]
            draw_single_route(ax, nodes, stop, uav_type, count)


        elif len(seq) == 4:
            stop_i = seq[1]
            stop_j = seq[2]
            draw_double_route(ax, nodes, stop_i, stop_j, uav_type, count)






def add_legend(ax):
    handles = [
        Line2D(
            [0], [0],
            marker="*",
            linestyle="none",
            markersize=11,
            markerfacecolor=BASE_COLOR,
            markeredgecolor="white",
            label="调度中心 O01",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=NODE_COLOR,
            markeredgecolor="white",
            label="服务区",
        ),
        Line2D(
            [0], [0],
            color="#7A7A7A",
            linewidth=0.5,
            alpha=0.6,
            label="单站路线（浅连线）",
        ),
        Line2D(
            [0], [0],
            color="#4A4A4A",
            linewidth=0.9,
            alpha=0.95,
            label="双站服务区间访问边",
        ),
        Line2D(
            [0], [0],
            color=TYPE_COLORS["A"],
            linewidth=1.0,
            label="A 型",
        ),
        Line2D(
            [0], [0],
            color=TYPE_COLORS["B"],
            linewidth=1.0,
            label="B 型",
        ),
        Line2D(
            [0], [0],
            color=TYPE_COLORS["C"],
            linewidth=1.0,
            label="C 型",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#D0D0D0",
        fancybox=False,
        ncol=1,
    )


def main():
    configure_matplotlib()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes()
    route_df = load_routes()

    src, dem, bounds, dem_transform = load_dem()

    try:
        terrain_norm = build_terrain_norm(dem)

        fig, ax = plt.subplots(figsize=(7.5, 6.6))

        im = draw_dem_base(ax, dem, bounds, dem_transform, terrain_norm)

        draw_routes(ax, nodes, route_df)
        draw_nodes(ax, nodes)

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.034,
            pad=0.025,
        )
        cbar.set_label("地面高程 / m", rotation=90, labelpad=8)
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(width=0.6, length=3, labelsize=8)

        add_legend(ax)

        ax.set_xlabel("经度 / °E")
        ax.set_ylabel("纬度 / °N")

        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)

        mean_lat = 0.5 * (LAT_MIN + LAT_MAX)
        ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.35,
            alpha=0.20,
        )

        fig.tight_layout()

        pdf_file = OUTPUT_DIR / "q2_f1_transport_routes.pdf"
        svg_file = OUTPUT_DIR / "q2_f1_transport_routes.svg"
        png_file = OUTPUT_DIR / "q2_f1_transport_routes.png"

        fig.savefig(pdf_file, bbox_inches="tight")
        fig.savefig(svg_file, bbox_inches="tight")
        fig.savefig(png_file, dpi=600, bbox_inches="tight")

        plt.close(fig)

        print("绘图完成：")
        print(f"  PDF: {pdf_file}")
        print(f"  SVG: {svg_file}")
        print(f"  PNG: {png_file}")

    finally:
        src.close()


if __name__ == "__main__":
    main()