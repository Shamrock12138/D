import pandas as pd
import numpy as np
from pathlib import Path
from dem_route import DEMRouteAnalyzer





















































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
    for key in ("distance", "h_max", "cruise_height"):
        if not np.allclose(matrices[key], matrices[key].T, rtol=0, atol=1e-8):
            mismatches = np.argwhere(
                ~np.isclose(matrices[key], matrices[key].T, rtol=0, atol=1e-8)
            )
            i, j = mismatches[0]
            raise ValueError(
                f"{key} 正反向不对称: {names[i]}->{names[j]}="
                f"{matrices[key][i, j]}, 反向={matrices[key][j, i]}"
            )
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
