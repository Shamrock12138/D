"""临时脚本：实时展示三维 DEM（不保存文件）。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import (
    LightSource,
    LinearSegmentedColormap,
    Normalize,
)
from matplotlib.cm import ScalarMappable

# ============================================================
# 路径（与 plot_terrain_heatmap.py 一致）
# ============================================================

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

# ============================================================
# 共用色带与归一化（与 plot_terrain_heatmap.py 完全相同）
# ============================================================

TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "terrain_blue_white_red",
    [
        (0.00, "#6F9FC8"),
        (0.20, "#A4C4DD"),
        (0.38, "#D5E4F0"),
        (0.50, "#FFFFFF"),
        (0.62, "#F7DEDE"),
        (0.80, "#E8A0A0"),
        (1.00, "#CC6670"),
    ],
    N=256,
)


def build_terrain_norm(dem: np.ndarray) -> Normalize:
    valid = dem[np.isfinite(dem)]
    return Normalize(
        vmin=float(np.min(valid)),
        vmax=float(np.max(valid)),
    )


# ============================================================
# 加载数据
# ============================================================

src = rasterio.open(DEM_FILE)
dem = src.read(1, masked=True).filled(np.nan).astype(float)
bounds = src.bounds
nodes = pd.read_csv(NODE_FILE)

terrain_norm = build_terrain_norm(dem)

# ============================================================
# 三维绘图
# ============================================================

nrows, ncols = dem.shape

step = max(1, int(max(nrows, ncols) / 350))
rows = np.arange(0, nrows, step)
cols = np.arange(0, ncols, step)

Z = dem[np.ix_(rows, cols)]

x_all = src.transform.c + (np.arange(ncols) + 0.5) * src.transform.a
y_all = src.transform.f + (np.arange(nrows) + 0.5) * src.transform.e

x = x_all[cols]
y = y_all[rows]
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(8.0, 6.3))
ax = fig.add_subplot(111, projection="3d")

# DEM
ax.plot_surface(
    X, Y, Z,
    cmap=TERRAIN_CMAP,
    norm=terrain_norm,
    linewidth=0,
    antialiased=True,
    shade=False,
    rasterized=True,
    alpha=1.0,
)

# 节点
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

# O01 到服务区虚线
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
        color="#62676D", alpha=0.72, zorder=6,
    )

# 标签
for _, row in nodes.iterrows():
    ax.text(
        float(row["x"]), float(row["y"]), float(row["h"]) + 28,
        f" {row['V']}", fontsize=7, color="#202124",
        fontweight="bold" if row["V"] == "O01" else "normal",
    )

# 坐标
ax.set_xlabel("经度 / °E")
ax.set_ylabel("纬度 / °N")
ax.set_zlabel("地面高程 / m")
ax.view_init(elev=32, azim=-62)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Colorbar
sm = ScalarMappable(norm=terrain_norm, cmap=TERRAIN_CMAP)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.64, pad=0.08, aspect=24)
cbar.set_label("地面高程 / m", labelpad=7)

plt.show()
src.close()