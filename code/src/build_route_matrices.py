import pandas as pd
import numpy as np
from pathlib import Path
from dem_route import DEMRouteAnalyzer

r"""
航段参数矩阵生成模块
====================

功能：基于服务区数据与DEM，计算所有节点对之间的航段参数，生成5个16×16矩阵和1个完整长表。

输入
----
- ``data/服务区数据.csv`` : 16个节点（1调度中心O01 + 15服务区S001~S015）的编号、经纬度、地面高程、需保障人口
- DEM : 镇龙乡30m分辨率数字高程模型（由 ``DEMRouteAnalyzer`` 自动加载）

处理流程
--------

1. 加载节点数据，区分调度中心（作业高度 = 地面高程）与服务区（作业高度 = 地面高程 + 30m）
2. 对每一对有序节点 (i, j), i≠j，调用 ``DEMRouteAnalyzer.get_route_parameter()`` 计算航段参数
3. 输出矩阵和长表到 ``data/`` 目录

输出文件
--------
============ ============================== ======
文件名        内容                           维度
============ ============================== ======
distance_matrix.csv         水平距离 L_ij (m)       16×16
h_max_matrix.csv            最高地面高程 (m)         16×16
cruise_height_matrix.csv    巡航高度 H_cr (m)        16×16
climb_height_matrix.csv     爬升高度 H_up (m)        16×16
descent_height_matrix.csv   下降高度 H_down (m)      16×16
route_parameter_all.csv     全部航段参数长表          240×8
============ ============================== ======

对应模型变量
------------

.. math::

    \mathcal{A}_{ij} = (L_{ij},\; H_{ij}^{cr},\; H_{ij}^{up},\; H_{ij}^{down})

使用方式
--------

>>> from build_route_matrices import main
>>> main()

或直接运行：

.. code-block:: bash

    python build_route_matrices.py
"""


def load_nodes():
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "服务区数据.csv"
    df = pd.read_csv(csv_path)
    nodes = {}
    for _, row in df.iterrows():
        name = row["V"]
        is_base = (name == "O01")
        nodes[name] = {
            "lon": row["x"],
            "lat": row["y"],
            "h_ground": row["h"],
            "is_base": is_base,
            "h_operation": row["h"] if is_base else row["h"] + 30,
        }
    return nodes


def build_route_matrices(nodes, dem):
    names = list(nodes.keys())
    n = len(names)

    distance_mat = np.zeros((n, n))
    h_max_mat = np.zeros((n, n))
    cruise_mat = np.zeros((n, n))
    climb_mat = np.zeros((n, n))
    descent_mat = np.zeros((n, n))

    total = n * (n - 1)
    count = 0

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                continue
            count += 1

            pi = (nodes[ni]["lon"], nodes[ni]["lat"])
            pj = (nodes[nj]["lon"], nodes[nj]["lat"])

            rp = dem.get_route_parameter(
                pi, pj,
                node1_ground=nodes[ni]["h_ground"],
                node1_op=nodes[ni]["h_operation"],
                node2_ground=nodes[nj]["h_ground"],
                node2_op=nodes[nj]["h_operation"],
                from_node=ni, to_node=nj,
            )

            distance_mat[i, j] = rp["distance"]
            h_max_mat[i, j] = rp["h_max"]
            cruise_mat[i, j] = rp["cruise_height"]
            climb_mat[i, j] = rp["climb_height"]
            descent_mat[i, j] = rp["descent_height"]

            if count % 40 == 0:
                print(f"  进度: {count}/{total}")

    matrices = {
        "distance": distance_mat,
        "h_max": h_max_mat,
        "cruise_height": cruise_mat,
        "climb_height": climb_mat,
        "descent_height": descent_mat,
    }
    return names, matrices


def save_matrices(names, matrices):
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_labels = {
        "distance": "距离矩阵_L (m)",
        "h_max": "最高地面高程矩阵_Hmax (m)",
        "cruise_height": "巡航高度矩阵_Hcr (m)",
        "climb_height": "爬升高度矩阵_Hup (m)",
        "descent_height": "下降高度矩阵_Hdown (m)",
    }

    for key, label in matrix_labels.items():
        mat = matrices[key]
        df = pd.DataFrame(mat, index=names, columns=names)
        filepath = output_dir / f"{key}_matrix.csv"
        df.to_csv(filepath, encoding="utf-8-sig")
        print(f"  已保存: {filepath.name}")

    summary_file = output_dir / "route_parameter_all.csv"
    rows = []
    n = len(names)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                continue
            rows.append({
                "from": ni,
                "to": nj,
                "distance": matrices["distance"][i, j],
                "h_max": matrices["h_max"][i, j],
                "cruise_height": matrices["cruise_height"][i, j],
                "climb_height": matrices["climb_height"][i, j],
                "descent_height": matrices["descent_height"][i, j],
            })
    df_all = pd.DataFrame(rows)
    df_all.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"  已保存: {summary_file.name}")


def main():
    print("=" * 55)
    print("Q1: 生成16×16航段参数矩阵")
    print("=" * 55)

    print("\n[1/3] 加载节点数据...")
    nodes = load_nodes()
    print(f"  共 {len(nodes)} 个节点 (1 调度中心 + {len(nodes)-1} 服务区)")

    print("\n[2/3] 加载DEM并计算航段参数...")
    dem = DEMRouteAnalyzer()
    names, matrices = build_route_matrices(nodes, dem)
    dem.close()

    print("\n[3/3] 保存矩阵到 data/ ...")
    save_matrices(names, matrices)

    print("\n" + "=" * 55)
    print("完成! 输出文件:")
    print("  data/distance_matrix.csv       - 水平距离矩阵 (m)")
    print("  data/h_max_matrix.csv           - 最高地面高程矩阵 (m)")
    print("  data/cruise_height_matrix.csv   - 巡航高度矩阵 (m)")
    print("  data/climb_height_matrix.csv    - 爬升高度矩阵 (m)")
    print("  data/descent_height_matrix.csv  - 下降高度矩阵 (m)")
    print("  data/route_parameter_all.csv    - 全部航段参数 (长表)")
    print("=" * 55)


if __name__ == "__main__":
    main()