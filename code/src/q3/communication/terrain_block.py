

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from .link_budget import horizontal_distance_m

PROJECT = Path(__file__).resolve().parents[4]
DEM_PATH = (
    PROJECT / "数据" / "镇龙乡地理空间数据" / "镇龙乡及周边地理数据"
    / "数字高程模型数据（DEM）" / "镇龙乡及周边30米DEM.tif"
)


@dataclass(frozen=True)
class TerrainResult:
    blocked: bool
    max_intrusion_m: float
    samples_checked: int


class DemTerrain:


    def __init__(self, path: Path = DEM_PATH, sample_step_m: float = 15.0):
        if sample_step_m <= 0 or not math.isfinite(sample_step_m):
            raise ValueError("DEM 采样间隔必须为正的有限值")
        self.image = Image.open(path)
        self.image.load()
        self.data = np.asarray(self.image)
        scale = self.image.tag_v2.get(33550)
        tie = self.image.tag_v2.get(33922)
        geo_keys = self.image.tag_v2.get(34735)
        if not scale or not tie or not geo_keys or 4326 not in geo_keys:
            raise ValueError("DEM 必须包含 EPSG:4326 的 GeoTIFF 定位信息")
        self.pixel_lon = float(scale[0])
        self.pixel_lat = float(scale[1])
        self.origin_lon = float(tie[3]) - float(tie[0]) * self.pixel_lon
        self.origin_lat = float(tie[4]) + float(tie[1]) * self.pixel_lat
        self.sample_step_m = sample_step_m

    def height_at(self, lon: float, lat: float) -> float:
        col = math.floor((lon - self.origin_lon) / self.pixel_lon)
        row = math.floor((self.origin_lat - lat) / self.pixel_lat)
        if not (0 <= col < self.image.width and 0 <= row < self.image.height):
            raise ValueError(f"通信视线位置 ({lon:.7f}, {lat:.7f}) 超出 DEM")
        height = float(self.image.getpixel((col, row)))
        if not math.isfinite(height):
            raise ValueError(f"通信视线位置 ({lon:.7f}, {lat:.7f}) 的 DEM 高程无效")
        return height

    def check_line(
        self, a: Tuple[float, float, float], b: Tuple[float, float, float]
    ) -> TerrainResult:
        return self.check_lines([a], [b])[0]

    def check_lines(self, starts, ends) -> list:

        starts = np.asarray(starts, dtype=float)
        ends = np.asarray(ends, dtype=float)
        if starts.ndim != 2 or starts.shape[1] != 3 or ends.shape != starts.shape:
            raise ValueError("starts 和 ends 必须是形状相同的 N×3 坐标数组")
        phi1 = np.radians(starts[:, 1])
        phi2 = np.radians(ends[:, 1])
        dphi = phi2 - phi1
        dlon = np.radians(ends[:, 0] - starts[:, 0])
        hav = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
        horizontal = 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
        n = np.maximum(1, np.ceil(horizontal / self.sample_step_m).astype(int))
        max_sample = max(0, int(n.max(initial=1)) - 1)
        if max_sample == 0:
            return [TerrainResult(False, 0.0, 0) for _ in range(len(starts))]

        k = np.arange(1, max_sample + 1, dtype=float)[None, :]
        valid = k < n[:, None]
        ratio = k / n[:, None]
        lon = starts[:, 0, None] + ratio * (ends[:, 0, None] - starts[:, 0, None])
        lat = starts[:, 1, None] + ratio * (ends[:, 1, None] - starts[:, 1, None])
        sight_z = starts[:, 2, None] + ratio * (ends[:, 2, None] - starts[:, 2, None])
        cols = np.floor((lon - self.origin_lon) / self.pixel_lon).astype(np.int32)
        rows = np.floor((self.origin_lat - lat) / self.pixel_lat).astype(np.int32)
        outside = valid & ((cols < 0) | (cols >= self.image.width) | (rows < 0) | (rows >= self.image.height))
        if outside.any():
            r, c = np.argwhere(outside)[0]
            raise ValueError(f"通信视线采样位置 ({lon[r, c]:.7f}, {lat[r, c]:.7f}) 超出 DEM")
        safe_rows = np.where(valid, rows, 0)
        safe_cols = np.where(valid, cols, 0)
        intrusion = np.where(valid, self.data[safe_rows, safe_cols] - sight_z, -np.inf)
        maxima = np.max(intrusion, axis=1)
        maxima = np.where(np.isfinite(maxima), maxima, 0.0)
        return [
            TerrainResult(
                blocked=bool(maxima[i] >= 0.0),
                max_intrusion_m=float(maxima[i]),
                samples_checked=int(max(0, n[i] - 1)),
            )
            for i in range(len(starts))
        ]

    def close(self) -> None:
        self.image.close()
