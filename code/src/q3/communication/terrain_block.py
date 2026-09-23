u"""用原始 30 米 GeoTIFF DEM 检查两个通信端点的地形遮挡。"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
    """按 GeoTIFF 像元中心最近邻采样视线，间隔不大于 15 米。"""

    def __init__(self, path: Path = DEM_PATH, sample_step_m: float = 15.0):
        if sample_step_m <= 0 or not math.isfinite(sample_step_m):
            raise ValueError("DEM 采样间隔必须为正的有限值")
        self.image = Image.open(path)
        self.image.load()
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
        horizontal_m = horizontal_distance_m(a, b)
        if horizontal_m < 1e-9:
            return TerrainResult(blocked=False, max_intrusion_m=0.0, samples_checked=0)
        n = max(1, math.ceil(horizontal_m / self.sample_step_m))
        intrusion = -math.inf
        for k in range(1, n):
            ratio = k / n
            lon = a[0] + ratio * (b[0] - a[0])
            lat = a[1] + ratio * (b[1] - a[1])
            sight_z = a[2] + ratio * (b[2] - a[2])
            intrusion = max(intrusion, self.height_at(lon, lat) - sight_z)
        if intrusion == -math.inf:
            intrusion = 0.0
        return TerrainResult(
            blocked=intrusion >= 0.0,
            max_intrusion_m=intrusion,
            samples_checked=max(0, n - 1),
        )

    def close(self) -> None:
        self.image.close()
