from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import (
    LightSource,
    LinearSegmentedColormap,
    Normalize,
)
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import numpy as np
import pandas as pd
import rasterio
from rasterio.coords import BoundingBox
from rasterio.windows import from_bounds






THIS_FILE = Path(__file__).resolve()
CODE_ROOT = THIS_FILE.parents[1]
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


class TransportMapBase:
    























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

    def __init__(
        self,
        schedule_file: Path,
        output_stem: str,
        expected_single: int | None = None,
        expected_double: int | None = None,
        figsize: tuple[float, float] = (7.5, 6.6),
    ):
        self.schedule_file = Path(schedule_file)
        self.output_stem = output_stem

        self.expected_single = expected_single
        self.expected_double = expected_double

        self.figsize = figsize

        self.output_dir = (
            CODE_ROOT
            / "figures"
            / "q2"
        )





    @staticmethod
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





    def load_dem(self):
        src = rasterio.open(
            DEM_FILE
        )

        if "4326" not in str(src.crs):
            raise ValueError(
                f"DEM CRS = {src.crs}，"
                "当前绘图基类要求 EPSG:4326"
            )

        window = from_bounds(
            left=self.LON_MIN,
            bottom=self.LAT_MIN,
            right=self.LON_MAX,
            top=self.LAT_MAX,
            transform=src.transform,
        )

        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        dem = src.read(
            1,
            window=window,
            masked=True,
        )

        dem = (
            dem
            .filled(np.nan)
            .astype(float)
        )

        transform = (
            src.window_transform(
                window
            )
        )

        bounds = BoundingBox(
            *rasterio.windows.bounds(
                window,
                src.transform,
            )
        )

        return (
            src,
            dem,
            bounds,
            transform,
        )





    def load_nodes(
        self,
    ) -> pd.DataFrame:

        nodes = pd.read_csv(
            NODE_FILE
        )

        required = {
            "V",
            "x",
            "y",
            "h",
        }

        missing = (
            required
            - set(nodes.columns)
        )

        if missing:
            raise ValueError(
                f"节点文件缺少字段: {missing}"
            )

        nodes = nodes.loc[
            (nodes["x"] >= self.LON_MIN)
            & (nodes["x"] <= self.LON_MAX)
            & (nodes["y"] >= self.LAT_MIN)
            & (nodes["y"] <= self.LAT_MAX)
        ].copy()

        return nodes





    def load_routes(
        self,
    ) -> pd.DataFrame:
        












        if not self.schedule_file.exists():
            raise FileNotFoundError(
                f"未找到排程文件："
                f"{self.schedule_file}"
            )

        df = pd.read_csv(
            self.schedule_file
        )

        required = {
            "uav_type",
            "n_stops",
            "visit_order",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"schedule 缺少字段: "
                f"{missing}"
            )

        df["uav_type"] = (
            df["uav_type"]
            .astype(str)
            .str.strip()
        )

        df["visit_order"] = (
            df["visit_order"]
            .astype(str)
            .str.strip()
        )

        df["n_stops"] = (
            df["n_stops"]
            .astype(int)
        )

        n_single = int(
            (df["n_stops"] == 1)
            .sum()
        )

        n_double = int(
            (df["n_stops"] == 2)
            .sum()
        )





        if (
            self.expected_single
            is not None
            and n_single
            != self.expected_single
        ):
            raise ValueError(
                f"单站架次数量错误："
                f"{n_single}，"
                f"预期 "
                f"{self.expected_single}"
            )

        if (
            self.expected_double
            is not None
            and n_double
            != self.expected_double
        ):
            raise ValueError(
                f"双站架次数量错误："
                f"{n_double}，"
                f"预期 "
                f"{self.expected_double}"
            )





        df["route"] = (
            "O01>"
            + df["visit_order"]
            + ">O01"
        )

        routes = (
            df.groupby(
                [
                    "uav_type",
                    "n_stops",
                    "route",
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size": "count"
                }
            )
        )

        print(
            f"{self.output_stem}: "
            f"{len(df)} 个架次，"
            f"{n_single} 个单站，"
            f"{n_double} 个双站，"
            f"{len(routes)} 个聚合路线"
        )

        return routes





    def build_terrain_norm(
        self,
        dem: np.ndarray,
    ) -> Normalize:
        valid = dem[
            np.isfinite(dem)
        ]

        vmin = float(
            np.min(valid)
        )

        vmax = float(
            np.max(valid)
        )

        return Normalize(
            vmin=vmin,
            vmax=vmax,
        )





    @staticmethod
    def calculate_hillshade(
        dem: np.ndarray,
        transform,
        bounds,
    ):
        mean_lat = (
            bounds.top
            + bounds.bottom
        ) / 2.0

        dx_deg = abs(
            transform.a
        )

        dy_deg = abs(
            transform.e
        )

        dx_m = (
            dx_deg
            * 111_320.0
            * np.cos(
                np.radians(
                    mean_lat
                )
            )
        )

        dy_m = (
            dy_deg
            * 110_540.0
        )

        median_elev = (
            np.nanmedian(dem)
        )

        filled_dem = np.where(
            np.isfinite(dem),
            dem,
            median_elev,
        )

        light = LightSource(
            azdeg=315,
            altdeg=45,
        )

        hillshade = (
            light.hillshade(
                filled_dem,
                vert_exag=1.0,
                dx=dx_m,
                dy=dy_m,
            )
        )

        hillshade[
            ~np.isfinite(dem)
        ] = np.nan

        return hillshade





    def draw_dem_base(
        self,
        ax,
        dem,
        bounds,
        transform,
        terrain_norm,
    ):
        im = ax.imshow(
            dem,
            extent=[
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top,
            ],
            origin="upper",
            cmap=self.TERRAIN_CMAP,
            norm=terrain_norm,
            interpolation="bilinear",
            zorder=1,
        )

        hillshade = (
            self.calculate_hillshade(
                dem,
                transform,
                bounds,
            )
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
            alpha=0.04,
            interpolation="bilinear",
            zorder=2,
        )





        valid = dem[
            np.isfinite(dem)
        ]

        vmin = float(
            np.min(valid)
        )

        vmax = float(
            np.max(valid)
        )

        levels = np.arange(
            np.floor(
                vmin / 100
            ) * 100,
            np.ceil(
                vmax / 100
            ) * 100 + 100,
            100,
        )

        dem_contour = (
            np.flipud(dem)
        )

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
            colors="#4E4E4E",
            linewidths=0.32,
            alpha=0.22,
            zorder=3,
        )

        return im





    @staticmethod
    def get_node_xy(
        nodes: pd.DataFrame,
        node_id: str,
    ) -> tuple[float, float]:

        row = nodes.loc[
            nodes["V"] == node_id
        ]

        if row.empty:
            raise KeyError(
                f"未找到节点 {node_id}"
            )

        row = row.iloc[0]

        return (
            float(row["x"]),
            float(row["y"]),
        )

    def draw_nodes(
        self,
        ax,
        nodes,
    ) -> None:

        base = nodes.loc[
            nodes["V"] == "O01"
        ]

        services = nodes.loc[
            nodes["V"] != "O01"
        ]

        ax.scatter(
            services["x"],
            services["y"],
            s=32,
            marker="o",
            facecolor=self.NODE_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=20,
        )

        ax.scatter(
            base["x"],
            base["y"],
            s=130,
            marker="*",
            facecolor=self.BASE_COLOR,
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
                idx = int(
                    node_id[1:]
                )

                offset = (
                    (4, 5)
                    if idx % 2 == 0
                    else (4, -8)
                )

                fsize = 7.5
                weight = "normal"

            text = ax.annotate(
                node_id,
                xy=(
                    row["x"],
                    row["y"],
                ),
                xytext=offset,
                textcoords="offset points",
                fontsize=fsize,
                fontweight=weight,
                color="#202124",
                zorder=30,
            )
            text.set_path_effects(
                [
                    pe.withStroke(
                        linewidth=2.0,
                        foreground="white",
                    )
                ]
            )





    @staticmethod
    def parse_route(
        route: str,
    ) -> list[str]:

        seq = [
            x.strip()
            for x
            in route.split(">")
            if x.strip()
        ]

        if len(seq) not in {
            3,
            4,
        }:
            raise ValueError(
                f"非法路线：{route}"
            )

        if (
            seq[0] != "O01"
            or seq[-1] != "O01"
        ):
            raise ValueError(
                f"路线必须从 O01 出发"
                f"并返回 O01：{route}"
            )

        return seq

    @staticmethod
    def add_line_with_halo(
        line,
    ) -> None:

        line.set_path_effects(
            [
                pe.Stroke(
                    linewidth=(
                        line.get_linewidth()
                        + 1.5
                    ),
                    foreground="white",
                ),
                pe.Normal(),
            ]
        )

    @staticmethod
    def add_patch_with_halo(
        patch,
        linewidth: float,
    ) -> None:

        patch.set_path_effects(
            [
                pe.Stroke(
                    linewidth=(
                        linewidth
                        + 1.5
                    ),
                    foreground="white",
                ),
                pe.Normal(),
            ]
        )





    @staticmethod
    def annotate_count(
        ax,
        p1,
        p2,
        text,
        color,
        dy=0.0,
    ) -> None:

        mx = (
            p1[0] + p2[0]
        ) / 2

        my = (
            p1[1] + p2[1]
        ) / 2 + dy

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
            [
                pe.withStroke(
                    linewidth=1.5,
                    foreground="white",
                )
            ]
        )





    def draw_single_route(
        self,
        ax,
        nodes,
        stop: str,
        uav_type: str,
        count: int,
    ) -> None:

        p0 = self.get_node_xy(
            nodes,
            "O01",
        )

        p1 = self.get_node_xy(
            nodes,
            stop,
        )

        color = (
            self.TYPE_COLORS[
                uav_type
            ]
        )

        lw = (
            0.4
            + 0.12
            * max(
                count - 1,
                0,
            )
        )

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            linewidth=lw,
            alpha=0.6,
            zorder=8,
        )

        if count > 1:
            self.annotate_count(
                ax,
                p0,
                p1,
                f"×{count}",
                color="#404040",
                dy=0.0015,
            )





    def draw_arrow_segment(
        self,
        ax,
        p_start,
        p_end,
        color,
        linewidth,
        alpha,
        linestyle="-",
        rad=0.0,
        zorder=12,
    ) -> None:

        patch = FancyArrowPatch(
            p_start,
            p_end,
            arrowstyle="-|>",
            mutation_scale=(
                8
                + 1.8 * linewidth
            ),
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            connectionstyle=(
                f"arc3,rad={rad}"
            ),
            shrinkA=6,
            shrinkB=6,
            zorder=zorder,
        )

        ax.add_patch(
            patch
        )





    def draw_double_route(
        self,
        ax,
        nodes,
        stop_i: str,
        stop_j: str,
        uav_type: str,
        count: int,
    ) -> None:

        p0 = self.get_node_xy(
            nodes,
            "O01",
        )

        pi = self.get_node_xy(
            nodes,
            stop_i,
        )

        pj = self.get_node_xy(
            nodes,
            stop_j,
        )

        color = (
            self.TYPE_COLORS[
                uav_type
            ]
        )

        lw_outer = (
            0.5
            + 0.1
            * max(
                count - 1,
                0,
            )
        )


        lw_inner = (
            0.8
            + 0.15
            * max(
                count - 1,
                0,
            )
        )


        self.draw_arrow_segment(
            ax,
            p0,
            pi,
            color,
            lw_outer,
            alpha=0.65,
            linestyle="-",
            rad=0.05,
            zorder=12,
        )


        self.draw_arrow_segment(
            ax,
            pi,
            pj,
            color,
            lw_inner,
            alpha=0.95,
            linestyle="-",
            rad=0.0,
            zorder=15,
        )


        self.draw_arrow_segment(
            ax,
            pj,
            p0,
            color,
            lw_outer,
            alpha=0.65,
            linestyle="-",
            rad=-0.05,
            zorder=12,
        )

        if count > 1:
            self.annotate_count(
                ax,
                pi,
                pj,
                f"×{count}",
                color="#303030",
                dy=0.0015,
            )





    def draw_routes(
        self,
        ax,
        nodes,
        routes,
    ) -> None:


        singles = routes.loc[
            routes["n_stops"] == 1
        ]

        doubles = routes.loc[
            routes["n_stops"] == 2
        ]

        for _, row in singles.iterrows():

            seq = self.parse_route(
                row["route"]
            )

            self.draw_single_route(
                ax,
                nodes,
                stop=seq[1],
                uav_type=row["uav_type"],
                count=int(row["count"]),
            )


        for _, row in doubles.iterrows():

            seq = self.parse_route(
                row["route"]
            )

            self.draw_double_route(
                ax,
                nodes,
                stop_i=seq[1],
                stop_j=seq[2],
                uav_type=row["uav_type"],
                count=int(row["count"]),
            )





    def add_legend(
        self,
        ax,
    ) -> None:

        handles = [
            Line2D(
                [0], [0],
                marker="*",
                linestyle="none",
                markersize=11,
                markerfacecolor=self.BASE_COLOR,
                markeredgecolor="white",
                label="调度中心 O01",
            ),

            Line2D(
                [0], [0],
                marker="o",
                linestyle="none",
                markersize=6,
                markerfacecolor=self.NODE_COLOR,
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
                color=self.TYPE_COLORS["A"],
                linewidth=1.0,
                label="A 型",
            ),

            Line2D(
                [0], [0],
                color=self.TYPE_COLORS["B"],
                linewidth=1.0,
                label="B 型",
            ),

            Line2D(
                [0], [0],
                color=self.TYPE_COLORS["C"],
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





    def setup_axes(
        self,
        ax,
    ) -> None:

        ax.set_xlabel(
            "经度 / °E"
        )

        ax.set_ylabel(
            "纬度 / °N"
        )

        ax.set_xlim(
            self.LON_MIN,
            self.LON_MAX,
        )

        ax.set_ylim(
            self.LAT_MIN,
            self.LAT_MAX,
        )

        mean_lat = (
            self.LAT_MIN
            + self.LAT_MAX
        ) / 2.0

        ax.set_aspect(
            1.0
            / np.cos(
                np.radians(
                    mean_lat
                )
            )
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.35,
            alpha=0.20,
        )





    def save_figure(
        self,
        fig,
    ) -> None:

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_file = (
            self.output_dir
            / f"{self.output_stem}.pdf"
        )

        svg_file = (
            self.output_dir
            / f"{self.output_stem}.svg"
        )

        png_file = (
            self.output_dir
            / f"{self.output_stem}.png"
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

        print(
            f"PDF: {pdf_file}"
        )

        print(
            f"SVG: {svg_file}"
        )

        print(
            f"PNG: {png_file}"
        )





    def draw(
        self,
    ) -> None:

        self.configure_matplotlib()

        nodes = (
            self.load_nodes()
        )

        routes = (
            self.load_routes()
        )

        (
            src,
            dem,
            bounds,
            transform,
        ) = self.load_dem()

        try:

            terrain_norm = (
                self.build_terrain_norm(
                    dem
                )
            )

            fig, ax = plt.subplots(
                figsize=self.figsize
            )


            im = self.draw_dem_base(
                ax,
                dem,
                bounds,
                transform,
                terrain_norm,
            )


            self.draw_routes(
                ax,
                nodes,
                routes,
            )


            self.draw_nodes(
                ax,
                nodes,
            )


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

            cbar.outline.set_linewidth(
                0.6
            )

            cbar.ax.tick_params(
                width=0.6,
                length=3,
                labelsize=8,
            )


            self.add_legend(
                ax
            )


            self.setup_axes(
                ax
            )

            fig.tight_layout()


            self.save_figure(
                fig
            )

            plt.close(
                fig
            )

        finally:
            src.close()