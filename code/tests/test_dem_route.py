"""DEM 航线穿越回归检查：主方向、顺序、NoData 和真实节点对。"""

import sys
import unittest
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_route_matrices import load_nodes  # noqa: E402
from dem_route import DEMRouteAnalyzer  # noqa: E402


class DemRouteTests(unittest.TestCase):
    def test_supercover_directions_and_nodata(self):
        values = np.zeros((12, 12), dtype=np.float32)
        values[5, 6] = 99
        values[1, 1] = -9999
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff", height=12, width=12, count=1,
                dtype="float32", transform=from_origin(0, 12, 1, 1),
                crs="EPSG:4326", nodata=-9999,
            ) as dataset:
                dataset.write(values, 1)
                analyzer = DEMRouteAnalyzer.__new__(DEMRouteAnalyzer)
                analyzer.src = dataset
                analyzer.dem = values
                cases = [
                    ((0.2, 6.5), (10.8, 6.5), 11),
                    ((6.5, 11.8), (6.5, 1.2), 11),
                    ((0.2, 11.8), (10.8, 1.2), 11),
                ]
                for start, end, minimum in cases:
                    with self.subTest(start=start, end=end):
                        cells = analyzer._grid_cells_along_line(start, end)
                        self.assertGreaterEqual(len(cells), minimum)
                        self.assertEqual(cells[0], dataset.index(*start))
                        self.assertEqual(cells[-1], dataset.index(*end))
                        self.assertEqual(
                            set(cells), set(analyzer._grid_cells_along_line(end, start))
                        )
                boundary = analyzer._grid_cells_along_line((6.0, 11.8), (6.0, 1.2))
                self.assertEqual({column for _, column in boundary}, {5, 6})
                self.assertEqual(
                    set(boundary),
                    set(analyzer._grid_cells_along_line((6.0, 1.2), (6.0, 11.8))),
                )
                self.assertEqual(
                    analyzer.get_max_dem_along_route(*cases[0][:2]), 99
                )
                with self.assertRaisesRegex(ValueError, "NoData"):
                    analyzer.get_max_dem_along_route((1.2, 10.8), (1.2, 10.8))

    def test_real_node_pairs_against_dense_sampling(self):
        nodes = load_nodes()
        points = {name: (item["lon"], item["lat"]) for name, item in nodes.items()}
        analyzer = DEMRouteAnalyzer()
        try:
            for first, p1 in points.items():
                for second, p2 in points.items():
                    if first >= second:
                        continue
                    with self.subTest(first=first, second=second):
                        cells = analyzer._grid_cells_along_line(p1, p2)
                        self.assertEqual(
                            set(cells), set(analyzer._grid_cells_along_line(p2, p1))
                        )
                        dense = {
                            analyzer.src.index(
                                p1[0] + (p2[0] - p1[0]) * t,
                                p1[1] + (p2[1] - p1[1]) * t,
                            )
                            for t in np.linspace(0, 1, 5000)
                        }
                        self.assertFalse(dense - set(cells))
        finally:
            analyzer.close()


if __name__ == "__main__":
    unittest.main()
