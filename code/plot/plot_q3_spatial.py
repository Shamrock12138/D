"""P003 transport routes, direct-link outages and deployed relay sites.

Run from the repository root with:
    python3 code/plot/plot_q3_spatial.py

All direct-link states are recomputed at 1 s with the Q3 DEM/link model.
The source schedules and relay sites are the selected P003 snapshot in
code/data/q3_final_frozen_v3/.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from src.q3.trajectory_generator import (  # noqa: E402
    TrajectoryGenerator, load_nodes, load_route_parameters, load_uav_parameters,
)
from src.q3.communication.direct_profile import evaluate_direct_positions  # noqa: E402
from src.q3.communication.link_budget import load_direct_parameters  # noqa: E402
from src.q3.communication.terrain_block import DemTerrain  # noqa: E402

DATA = CODE / "data" / "q3_final_frozen_v3"
OUT = CODE / "figures" / "q3"
COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
GAP = "#772E92"
INK = "#263844"


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Noto Sans CJK SC",
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
            "Arial",
        ],
        "axes.unicode_minus": False,
        "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def load_inputs():
    transport = pd.read_csv(DATA / "q3_joint_transport_schedule.csv")
    deliveries = pd.read_csv(DATA / "q3_joint_delivery_schedule.csv")
    relay = pd.read_csv(DATA / "q3_joint_relay_schedule.csv")
    sites = pd.read_csv(CODE / "data" / "q3_relay_sites.csv")
    if len(transport) != 17 or len(deliveries) != 80 or len(relay) != 13:
        raise ValueError("P003 input counts must be 17 sorties, 80 boxes, 13 gaps")
    counts = relay.groupby("candidate_id").size().to_dict()
    if len(counts) != 7 or sum(counts.values()) != 13:
        raise ValueError("P003 must use seven physical relay sites")
    if any(deliveries.groupby("sortie_id").size().reindex(transport.sortie_id).isna()):
        raise ValueError("A transport sortie has no box delivery records")
    sites = sites.loc[sites.candidate_id.isin(counts)].copy()
    if len(sites) != 7:
        raise ValueError("Relay site coordinates do not cover all seven sites")
    return transport, deliveries, relay, sites, counts


def make_trajectories(transport, deliveries, nodes, terrain, *, check_only=False):
    generator = TrajectoryGenerator(nodes, load_route_parameters(), dt=1.0)
    uav_parameters = load_uav_parameters()
    link_parameters = load_direct_parameters()
    gateway = (nodes["O01"]["x"], nodes["O01"]["y"],
               nodes["O01"]["h"] + link_parameters.gateway_height_m)
    paths = []
    outage_samples = 0
    duration_errors = []
    for row in transport.itertuples(index=False):
        boxes = deliveries.loc[deliveries.sortie_id == row.sortie_id]
        route = ["O01", *str(row.visit_order).split(">"), "O01"]
        services = boxes.service.astype(str).tolist()
        if Counter(services).keys() != set(route[1:-1]):
            raise ValueError(f"Delivery stops differ from route: {row.sortie_id}")
        trajectory = generator.generate_relative(
            row.uav_type, route, services, uav_parameters[row.uav_type]
        )
        times = np.asarray(trajectory.times, dtype=float)
        lon = np.asarray(trajectory.x, dtype=float)
        lat = np.asarray(trajectory.y, dtype=float)
        z = np.asarray(trajectory.z, dtype=float)
        error = float(times[-1] - row.duration_s)
        duration_errors.append(abs(error))
        if abs(error) > 2.0:
            raise ValueError(
                f"{row.sortie_id}: trajectory and schedule differ by {error:.3f} s"
            )
        if check_only:
            continue
        positions = list(zip(lon.tolist(), lat.tolist(), z.tolist()))
        states = evaluate_direct_positions(
            positions, gateway, terrain, link_parameters, batch_size=64
        )
        direct = np.fromiter((state.direct for state in states), dtype=bool)
        outage_samples += int((~direct).sum())
        paths.append((row.sortie_id, row.uav_type, times, lon, lat, direct))
        print(f"{len(paths):02d}/17 {row.sortie_id}: {len(times)} samples, "
              f"{int((~direct).sum())} outage samples", flush=True)
    print(f"Maximum trajectory duration discrepancy: {max(duration_errors):.3f} s")
    if not check_only:
        print(f"Direct-link outage samples: {outage_samples}")
    return paths


def plot(transport, deliveries, relay, sites, counts, nodes, terrain):
    paths = make_trajectories(transport, deliveries, nodes, terrain)
    used_lon = [node["x"] for node in nodes.values()] + sites.lon.tolist()
    used_lat = [node["y"] for node in nodes.values()] + sites.lat.tolist()
    lon_min, lon_max = min(used_lon)-0.009, max(used_lon)+0.009
    lat_min, lat_max = min(used_lat)-0.008, max(used_lat)+0.008
    c0 = max(0, int((lon_min-terrain.origin_lon)/terrain.pixel_lon))
    c1 = min(terrain.image.width, int((lon_max-terrain.origin_lon)/terrain.pixel_lon)+1)
    r0 = max(0, int((terrain.origin_lat-lat_max)/terrain.pixel_lat))
    r1 = min(terrain.image.height, int((terrain.origin_lat-lat_min)/terrain.pixel_lat)+1)
    dem = np.asarray(terrain.data[r0:r1, c0:c1], dtype=float)
    extent = (terrain.origin_lon+c0*terrain.pixel_lon,
              terrain.origin_lon+c1*terrain.pixel_lon,
              terrain.origin_lat-r1*terrain.pixel_lat,
              terrain.origin_lat-r0*terrain.pixel_lat)

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    fig.subplots_adjust(left=0.11, right=0.88, bottom=0.16, top=0.96)
    im = ax.imshow(dem, extent=extent, origin="upper", cmap="Greys",
                   vmin=np.nanpercentile(dem, 3),
                   vmax=np.nanpercentile(dem, 98), alpha=0.43, zorder=0)

    # A/B/C lines retain the established Q2/Q3 fleet palette; the overlaid
    # purple pieces are the 1 s states with no direct G01 link.
    for _, uav_type, _, lon, lat, direct in paths:
        flying = (np.abs(np.diff(lon)) + np.abs(np.diff(lat))) > 1e-10
        for i in np.flatnonzero(flying):
            ax.plot(lon[i:i+2], lat[i:i+2], color=COLORS[uav_type],
                    linewidth=0.95, alpha=0.36, zorder=2)
        for i in np.flatnonzero(flying & ~(direct[:-1] & direct[1:])):
            ax.plot(lon[i:i+2], lat[i:i+2], color=GAP,
                    linewidth=2.0, alpha=0.82, zorder=4)

    base = nodes["O01"]
    for row in sites.itertuples(index=False):
        ax.plot([base["x"], row.lon], [base["y"], row.lat],
                color="#545D66", linestyle=(0, (2.5, 3.5)),
                linewidth=0.78, alpha=0.45, zorder=1)
    for name, node in nodes.items():
        if name == "O01":
            continue
        ax.scatter(node["x"], node["y"], marker="o", s=23,
                   facecolor="white", edgecolor=INK, linewidth=0.8, zorder=7)
        ax.annotate(name, (node["x"], node["y"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7.5,
                    color=INK, weight="medium", zorder=9)
    ax.scatter(base["x"], base["y"], s=100, marker="s", color="#C03536",
               edgecolor="white", linewidth=1, zorder=11)
    ax.annotate("O01 / G01", (base["x"], base["y"]), xytext=(-15, -17),
                textcoords="offset points", fontsize=8.5, weight="bold",
                color="#A02028", zorder=12)

    for row in sites.itertuples(index=False):
        major = row.candidate_id in {"RP016472", "RP065617"}
        ax.scatter(row.lon, row.lat, marker="D", s=72 if major else 39,
                   facecolor="#F5D15B" if major else "white",
                   edgecolor=INK, linewidth=1.0, zorder=12)
        if major:
            shift = (15, 12) if row.candidate_id == "RP016472" else (-86, 11)
            ax.annotate(f"{row.candidate_id}\n{counts[row.candidate_id]} 个缺口",
                        (row.lon, row.lat), xytext=shift,
                        textcoords="offset points", fontsize=8.1,
                        weight="bold", color=INK, linespacing=1.25,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.8),
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=1.6), zorder=13)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1 / np.cos(np.deg2rad((lat_min+lat_max)/2)))
    ax.set_xlabel("经度 / °")
    ax.set_ylabel("纬度 / °")
    ax.grid(color="#A9B2B9", linewidth=0.45, alpha=0.28)
    ax.tick_params(labelsize=8.0)
    handles = [Line2D([], [], color=COLORS[g], linewidth=2.2,
                      label=f"{g} 型直连航迹") for g in "ABC"]
    handles += [
        Line2D([], [], color=GAP, linewidth=2.6, label="需中继保障航迹"),
        Line2D([], [], color="#545D66", linestyle=(0, (2.5, 3.5)),
               linewidth=1.0, label="中继至 G01"),
        Line2D([], [], marker="D", linestyle="none", color=INK,
               markerfacecolor="white", markersize=6, label="实际中继悬停点"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.49, 0.065),
               handlelength=2.6, columnspacing=1.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.025, shrink=0.72)
    cbar.set_label("DEM 高程 / m", fontsize=8.4)
    cbar.ax.tick_params(labelsize=7.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"q3_transport_comm_spatial.{ext}", dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    set_style()
    transport, deliveries, relay, sites, counts = load_inputs()
    nodes = load_nodes()
    terrain = DemTerrain()
    try:
        if args.check_only:
            make_trajectories(transport, deliveries, nodes, terrain,
                              check_only=True)
        else:
            plot(transport, deliveries, relay, sites, counts, nodes, terrain)
    finally:
        terrain.close()


if __name__ == "__main__":
    main()