import rasterio
import numpy as np
from pathlib import Path

r"""
DEM航线参数提取模块
====================

功能：给定两个节点的经纬度坐标，利用DEM（数字高程模型）栅格数据提取航线经过区域的
最高地面高程，并计算航段完整参数集。

模型对应
--------

.. math::

    h_{ij}^{\max} = \max_{p \in \mathcal{P}_{ij}} \text{DEM}(p)

    H_{ij}^{cr} = h_{ij}^{\max} + 50

    H_{ij}^{up} = H_{ij}^{cr} - h_i^{op}

    H_{ij}^{down} = H_{ij}^{cr} - h_j^{op}

    \mathcal{A}_{ij} = (L_{ij},\; H_{ij}^{cr},\; H_{ij}^{up},\; H_{ij}^{down})

使用示例
--------

>>> from dem_route import DEMRouteAnalyzer

>>> dem = DEMRouteAnalyzer()

>>> O01 = (109.230852, 23.008509)
>>> S01 = (109.243232, 23.033593)

>>> dem.get_max_dem_along_route(O01, S01)
231.7

>>> dem.get_cruise_height(O01, S01)
281.7

>>> dem.get_horizontal_distance(O01, S01)
3063.0

>>> dem.get_single_point_height(109.230852, 23.008509)
128.7

>>> dem.get_node_parameter(109.192379, 23.049455, h_ground=444.5, is_service_area=True)
{'lon': 109.192379, 'lat': 23.049455, 'h_ground': 444.5, 'h_operation': 474.5}

>>> dem.get_node_parameter(109.230852, 23.008509, is_service_area=False)
{'lon': 109.230852, 'lat': 23.008509, 'h_ground': 128.7, 'h_operation': 128.7}

>>> rp = dem.get_route_parameter(
...     O01, S01,
...     node1_ground=128.7, node1_op=128.7,
...     node2_ground=154.0, node2_op=184.0,
...     from_node="O01", to_node="S001",
... )
>>> rp
{
    'from_node': 'O01',
    'to_node': 'S001',
    'distance': 3063.0,
    'h_max': 231.7,
    'cruise_height': 281.7,
    'start_ground': 128.7,
    'end_ground': 154.0,
    'start_operation': 128.7,
    'end_operation': 184.0,
    'climb_height': 153.0,
    'descent_height': 97.7,
}

>>> dem.get_route_profile(O01, S01)  # 返回航线剖面的完整栅格序列
{'cells': [...], 'elevations': [...], 'lons': [...], 'lats': [...],
 'h_max': 231.7, 'h_min': 125.0}

>>> dem.close()

输入要求
--------
- 节点坐标为 ``(lon, lat)`` 即 ``(经度, 纬度)``，单位：度 (EPSG:4326)
- DEM 默认为镇龙乡 30m 分辨率 GeoTIFF
- 服务区作业高度 = 地面高程 + 30m；调度中心作业高度 = 地面高程

输出字段说明（get_route_parameter）
-----------------------------------
=============== ====== ==========================================
字段             单位   含义
=============== ====== ==========================================
from_node             起点节点编号
to_node               终点节点编号
distance       m      水平距离（Haversine 球面距离）
h_max          m      航线经过区域的最高地面高程
cruise_height  m      巡航高度 = h_max + 50
start_ground   m      起点地面高程
end_ground     m      终点地面高程
start_operation m     起点作业高度
end_operation  m      终点作业高度
climb_height   m      爬升高度 = cruise_height - start_operation
descent_height m      下降高度 = cruise_height - end_operation
=============== ====== ==========================================
"""

class DEMRouteAnalyzer:
    def __init__(self, dem_path=None):
        if dem_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            dem_path = (
                project_root
                / "数据"
                / "镇龙乡地理空间数据"
                / "镇龙乡及周边地理数据"
                / "数字高程模型数据（DEM）"
                / "镇龙乡及周边30米DEM.tif"
            )
        self.src = rasterio.open(str(dem_path))
        self.dem = self.src.read(1)
        self._validate_crs()

    def _validate_crs(self):
        crs_str = str(self.src.crs).upper()
        if "4326" not in crs_str:
            raise ValueError(
                f"DEM坐标系为 {self.src.crs}，期望 EPSG:4326。"
                f"当前输入的经纬度坐标需要先用 pyproj 转换到 {self.src.crs}"
            )
        self._crs_is_4326 = True

    @property
    def resolution(self):
        return (abs(self.src.transform[0]), abs(self.src.transform[4]))

    @property
    def shape(self):
        return self.dem.shape

    @property
    def crs(self):
        return self.src.crs

    @staticmethod
    def _haversine_distance(lon1, lat1, lon2, lat2):
        R = 6371000.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def _grid_cells_along_line(self, p1, p2):
        # 在连续像素坐标中逐一跨越网格边界；角点同时纳入两侧像元。
        inverse = ~self.src.transform
        c0, r0 = inverse @ p1
        c1, r1 = inverse @ p2
        if not all(np.isfinite(v) for v in (r0, c0, r1, c1)):
            raise ValueError("航线坐标无效")

        r, c = int(np.floor(r0)), int(np.floor(c0))
        end_r, end_c = int(np.floor(r1)), int(np.floor(c1))
        nrows, ncols = self.dem.shape
        if not (0 <= r < nrows and 0 <= c < ncols
                and 0 <= end_r < nrows and 0 <= end_c < ncols):
            raise ValueError("航线端点超出DEM范围")

        cells = []
        seen = set()

        def add_cell(rr, cc):
            if 0 <= rr < nrows and 0 <= cc < ncols and (rr, cc) not in seen:
                cells.append((rr, cc))
                seen.add((rr, cc))

        dc, dr = c1 - c0, r1 - r0
        step_c = (dc > 0) - (dc < 0)
        step_r = (dr > 0) - (dr < 0)
        delta_c = np.inf if step_c == 0 else 1.0 / abs(dc)
        delta_r = np.inf if step_r == 0 else 1.0 / abs(dr)
        next_c = ((c + 1) - c0) / dc if step_c > 0 else (
            (c0 - c) / -dc if step_c < 0 else np.inf
        )
        next_r = ((r + 1) - r0) / dr if step_r > 0 else (
            (r0 - r) / -dr if step_r < 0 else np.inf
        )

        add_cell(r, c)
        # 恰好沿网格线飞行时，两侧像元都与航线相交。
        on_col_edge = step_c == 0 and np.isclose(c0, round(c0), atol=1e-12, rtol=0)
        on_row_edge = step_r == 0 and np.isclose(r0, round(r0), atol=1e-12, rtol=0)

        def add_edge_neighbors():
            if on_col_edge:
                add_cell(r, c - 1)
            if on_row_edge:
                add_cell(r - 1, c)
            if on_col_edge and on_row_edge:
                add_cell(r - 1, c - 1)

        add_edge_neighbors()
        limit = 3 * (abs(end_r - r) + abs(end_c - c) + 10)
        for _ in range(limit):
            if (r, c) == (end_r, end_c):
                break
            if next_c < next_r - 1e-12:
                c += step_c
                next_c += delta_c
            elif next_r < next_c - 1e-12:
                r += step_r
                next_r += delta_r
            else:
                add_cell(r, c + step_c)
                add_cell(r + step_r, c)
                c += step_c
                r += step_r
                next_c += delta_c
                next_r += delta_r
            add_cell(r, c)
            add_edge_neighbors()
        else:
            raise RuntimeError("DEM栅格穿越未到达终点")

        return cells

    def _valid_route_elevations(self, cells):
        elevations = np.asarray([self.dem[r, c] for r, c in cells], dtype=float)
        valid = np.isfinite(elevations)
        if self.src.nodata is not None:
            valid &= elevations != self.src.nodata
        if not np.any(valid):
            raise ValueError("航线经过的DEM栅格全部为NoData")
        return elevations, valid

    def get_single_point_height(self, lon, lat):
        r, c = self.src.index(lon, lat)
        if 0 <= r < self.dem.shape[0] and 0 <= c < self.dem.shape[1]:
            return float(self.dem[r, c])
        raise ValueError(f"坐标 ({lon}, {lat}) 超出DEM范围")

    def get_node_parameter(self, lon, lat, h_ground=None, h_operation=None,
                           is_service_area=False):
        if h_ground is None:
            h_ground = self.get_single_point_height(lon, lat)
        if h_operation is None:
            h_operation = h_ground + 30 if is_service_area else h_ground
        return {
            "lon": lon,
            "lat": lat,
            "h_ground": float(h_ground),
            "h_operation": float(h_operation),
        }

    def get_horizontal_distance(self, p1, p2):
        return self._haversine_distance(p1[0], p1[1], p2[0], p2[1])

    def get_max_dem_along_route(self, p1, p2):
        cells = self._grid_cells_along_line(p1, p2)
        if not cells:
            raise ValueError("航线上无有效DEM栅格")
        elevations, valid = self._valid_route_elevations(cells)
        return float(np.max(elevations[valid]))

    def get_cruise_height(self, p1, p2, safety_margin=50.0):
        h_max = self.get_max_dem_along_route(p1, p2)
        return h_max + safety_margin

    def get_route_profile(self, p1, p2):
        cells = self._grid_cells_along_line(p1, p2)
        if not cells:
            raise ValueError("航线上无有效DEM栅格")

        rows, cols = zip(*cells)
        elevations, valid = self._valid_route_elevations(cells)
        xs, ys = zip(*[self.src.xy(r, c) for r, c in cells])

        return {
            "cells": cells,
            "elevations": elevations.tolist(),
            "lons": list(xs),
            "lats": list(ys),
            "h_max": float(np.max(elevations[valid])),
            "h_min": float(np.min(elevations[valid])),
        }

    def get_route_parameter(self, p1, p2, node1_ground=None, node1_op=None,
                            node2_ground=None, node2_op=None,
                            from_node=None, to_node=None, safety_margin=50.0):
        distance = self.get_horizontal_distance(p1, p2)
        h_max = self.get_max_dem_along_route(p1, p2)
        cruise_height = h_max + safety_margin

        h1_g = node1_ground if node1_ground is not None else self.get_single_point_height(*p1)
        h2_g = node2_ground if node2_ground is not None else self.get_single_point_height(*p2)
        h1_op = node1_op if node1_op is not None else h1_g
        h2_op = node2_op if node2_op is not None else h2_g

        climb_height = cruise_height - h1_op
        descent_height = cruise_height - h2_op

        return {
            "from_node": from_node,
            "to_node": to_node,
            "distance": distance,
            "h_max": h_max,
            "cruise_height": cruise_height,
            "start_ground": float(h1_g),
            "end_ground": float(h2_g),
            "start_operation": float(h1_op),
            "end_operation": float(h2_op),
            "climb_height": climb_height,
            "descent_height": descent_height,
        }

    def close(self):
        self.src.close()

def demo():
    dem = DEMRouteAnalyzer()

    print(f"DEM CRS: {dem.crs}")
    print(f"DEM 尺寸: {dem.shape}")
    print(f"DEM 分辨率: {dem.resolution}")
    print()

    O01 = (109.230852, 23.008509)
    o01_ground = dem.get_single_point_height(*O01)
    o01_op = o01_ground

    test_points = [
        ("S001", (109.243232, 23.033593), 154.0),
        ("S002", (109.258552, 23.070327), 189.5),
        ("S005", (109.228300, 23.056134), 165.0),
        ("S010", (109.276017, 23.019401), 321.6),
        ("S015", (109.192379, 23.049455), 444.5),
    ]

    print(f"{'航线':<12} {'距离(m)':>10} {'H_max(m)':>10} {'H_cr(m)':>8} {'H_up(m)':>8} {'H_down(m)':>10}")
    print("-" * 62)

    for name, coord, s_ground in test_points:
        s_op = s_ground + 30
        rp = dem.get_route_parameter(
            O01, coord,
            node1_ground=o01_ground, node1_op=o01_op,
            node2_ground=s_ground, node2_op=s_op,
            from_node="O01", to_node=name,
        )
        print(
            f"O01->{name:<4} "
            f"{rp['distance']:>10.0f} "
            f"{rp['h_max']:>10.1f} "
            f"{rp['cruise_height']:>8.1f} "
            f"{rp['climb_height']:>8.1f} "
            f"{rp['descent_height']:>10.1f}"
        )

    print()
    print("O01 节点:", dem.get_node_parameter(*O01, is_service_area=False))
    print("S015 节点:", dem.get_node_parameter(109.192379, 23.049455, h_ground=444.5,
                                                is_service_area=True))

    dem.close()


if __name__ == "__main__":
    demo()
