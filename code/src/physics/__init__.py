r"""
运输无人机能耗与作业时间基础模型
================================

机型集合 G = {A, B, C}，航段能耗按水平航程能耗与爬升附加能耗之和计算，
下降能耗效率为 0。

参数记号
--------
============= ====== ============================================
M_g^0         kg     含电池空载总质量
Q_g           kg     最大载荷
V_g           m³     可用装载体积
L_g^0         m      空载标准航程（满载 0 kg 时的最大水平航程）
L_g^F         m      满载标准航程（满载 Q_g kg 时的最大水平航程）
E_g^use       kWh    电池可用能量
ρ_g           %      返航电量下限
η_g^up               爬升能耗效率
v^up          m/s    最大爬升速度
v^cr          m/s    计划巡航速度
v^down        m/s    最大下降速度
T_setup       s      工位固定准备时间
T_load        s      每箱装载时间
T_handover    s      接收点基础交接时间
T_handover_p  s      每箱增加交接时间
============= ====== ============================================

统一接口
--------
:math:`\\mathcal{E}(g, i, j, q)` — 航段运输能耗
:math:`\\mathcal{T}(g, i, j)`   — 航段飞行时间
"""

import pandas as pd
from pathlib import Path

G = 9.81

PROJECT = Path(__file__).resolve().parent.parent.parent


class TransportPhysicsModel:
    u"""运输无人机物理模型

    提供两个统一接口：
        E(g, i, j, q) — 从 i 到 j 携带载荷 q 的单航段能耗 (kWh)
        T(g, i, j)   — 从 i 到 j 的单航段飞行时间 (s)
    """

    def __init__(self, uav_row, route_params):
        u"""
        Parameters
        ----------
        uav_row : pd.Series
            单行机型参数，含 M_g0, Q_g, V_g, L_0, L_F, E_use, ρ_g, η_up,
            v_k^up, v_k^cr, v_k^down
        route_params : dict of pd.DataFrame
            distance_matrix, climb_height_matrix, descent_height_matrix
        """
        self.u = uav_row
        self.routes = route_params

    # ── 等效航程 ──────────────────────────────────────────
    def equivalent_range(self, q):
        u"""L_g(q) = L_g^0 - (L_g^0 - L_g^F) * (q / Q_g)^{3/2}"""
        u = self.u
        return u["L_0"] - (u["L_0"] - u["L_F"]) * (q / u["Q_g"]) ** 1.5

    # ── 水平巡航能耗 ──────────────────────────────────────
    def horizontal_energy(self, q, distance):
        u"""E^{hor} = E^{use} * d / L_g(q)   [kWh]"""
        Lq = self.equivalent_range(q)
        return self.u["E_use"] * distance / Lq

    # ── 爬升附加能耗 ──────────────────────────────────────
    def climb_energy(self, q, H_up):
        u"""E^{up} = (M_g^0 + q) * g * H_up / (3.6e6 * η^{up})   [kWh]"""
        return (
            (self.u["M_g0"] + q) * G * H_up
            / (3.6e6 * self.u["η_up"])
        )

    # ── 下降能耗 ──────────────────────────────────────────
    @staticmethod
    def descent_energy():
        return 0.0

    # ── 单航段总能耗 ──────────────────────────────────────
    def leg_energy(self, q, distance, H_up):
        u"""E(i→j, q) = E^{hor}(d, q) + E^{up}(H_up, q)   [kWh]"""
        return self.horizontal_energy(q, distance) + self.climb_energy(q, H_up)

    # ── 航段飞行时间 ──────────────────────────────────────
    def leg_time(self, origin, destination):
        u"""t^{fly} = H_up/v_up + d/v_cr + H_down/v_down   [s]"""
        u = self.u
        H_up = float(self.routes["climb_height"].loc[origin, destination])
        H_down = float(self.routes["descent_height"].loc[origin, destination])
        d = float(self.routes["distance"].loc[origin, destination])

        return H_up / u["v_k^up"] + d / u["v_k^cr"] + H_down / u["v_k^down"]

    # ── 统一接口 ──────────────────────────────────────────
    def energy(self, origin, destination, q):
        u"""E(g, i, j, q): 航段运输能耗 [kWh]"""
        H_up = float(self.routes["climb_height"].loc[origin, destination])
        d = float(self.routes["distance"].loc[origin, destination])
        return self.leg_energy(q, d, H_up)

    def time(self, origin, destination):
        u"""T(g, i, j): 航段飞行时间 [s]"""
        return self.leg_time(origin, destination)

    # ── 往返能耗 ──────────────────────────────────────────
    def round_trip_energy(self, q, service_area):
        u"""O01→Si→O01 往返总能耗: 去程载荷 q, 返程空载 0 [kWh]"""
        E_out = self.energy("O01", service_area, q)
        E_back = self.energy(service_area, "O01", 0.0)
        return E_out + E_back

    # ── 可用能量 ──────────────────────────────────────────
    @property
    def available_energy(self):
        u"""(1 - ρ_g) * E_g^{use}   [kWh]"""
        return (1.0 - self.u["ρ_g"] / 100.0) * self.u["E_use"]

    # ── 最大安全载荷（二分搜索）──────────────────────────
    def max_safe_payload(self, service_area):
        u"""二分搜索: max q s.t. E_round(q) ≤ (1-ρ)*E_use, q ≤ Q_g"""
        E_avail = self.available_energy

        def surplus(m):
            return E_avail - self.round_trip_energy(m, service_area)

        if surplus(0.0) < 0:
            return 0.0

        lo, hi = 0.0, float(self.u["Q_g"])
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if surplus(mid) >= 0:
                lo = mid
            else:
                hi = mid
        return lo

    # ── 架次总作业时间 ────────────────────────────────────
    def sortie_total_time(self, n_boxes, service_area):
        u"""T_total = T_setup + T_load + T_fly(往返) + T_handover"""
        u = self.u
        t_fly_out = self.time("O01", service_area)
        t_fly_back = self.time(service_area, "O01")

        return (
            u["T_setup"]
            + u["T_load"] * n_boxes
            + t_fly_out
            + t_fly_back
            + u["T_handover"]
            + u["T_handover_p"] * n_boxes
        )


def load_models():
    u"""加载所有机型的 TransportPhysicsModel 实例"""
    uav_df = pd.read_csv(PROJECT / "data" / "运输无人机_机型参数.csv")
    data_dir = PROJECT / "data"

    route_params = {}
    for key in ["distance", "climb_height", "descent_height"]:
        route_params[key] = pd.read_csv(data_dir / f"{key}_matrix.csv", index_col=0)

    models = {}
    for _, row in uav_df.iterrows():
        g = row["type"]
        models[g] = TransportPhysicsModel(row, route_params)

    return models


if __name__ == "__main__":
    models = load_models()
    for g, m in models.items():
        print(f"机型 {g}: Q_g={m.u['Q_g']}kg  E_use={m.u['E_use']}kWh  "
              f"ρ={m.u['ρ_g']}%")
        for si in ["S001", "S002", "S008"]:
            msp = m.max_safe_payload(si)
            E_rt = m.round_trip_energy(msp, si)
            print(f"  {si}: max_payload={msp:.1f}kg  E_round={E_rt:.4f}kWh  "
                  f"E_avail={m.available_energy:.4f}kWh")
        print()