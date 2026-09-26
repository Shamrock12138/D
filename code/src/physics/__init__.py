
































import pandas as pd
from pathlib import Path

G = 9.81

PROJECT = Path(__file__).resolve().parent.parent.parent


class TransportPhysicsModel:
    






    def __init__(self, uav_row, route_params):
        








        self.u = uav_row
        self.routes = route_params


    def equivalent_range(self, q):

        u = self.u
        return u["L_0"] - (u["L_0"] - u["L_F"]) * (q / u["Q_g"]) ** 1.5


    def horizontal_energy(self, q, distance):

        Lq = self.equivalent_range(q)
        return self.u["E_use"] * distance / Lq


    def climb_energy(self, q, H_up):

        return (
            (self.u["M_g0"] + q) * G * H_up
            / (3.6e6 * self.u["η_up"])
        )


    @staticmethod
    def descent_energy():
        return 0.0


    def leg_energy(self, q, distance, H_up):

        return self.horizontal_energy(q, distance) + self.climb_energy(q, H_up)


    def leg_time(self, origin, destination):

        u = self.u
        H_up = float(self.routes["climb_height"].loc[origin, destination])
        H_down = float(self.routes["descent_height"].loc[origin, destination])
        d = float(self.routes["distance"].loc[origin, destination])

        return H_up / u["v_k^up"] + d / u["v_k^cr"] + H_down / u["v_k^down"]


    def energy(self, origin, destination, q):

        H_up = float(self.routes["climb_height"].loc[origin, destination])
        d = float(self.routes["distance"].loc[origin, destination])
        return self.leg_energy(q, d, H_up)

    def time(self, origin, destination):

        return self.leg_time(origin, destination)


    def round_trip_energy(self, q, service_area):

        E_out = self.energy("O01", service_area, q)
        E_back = self.energy(service_area, "O01", 0.0)
        return E_out + E_back


    @property
    def available_energy(self):

        return (1.0 - self.u["ρ_g"] / 100.0) * self.u["E_use"]


    def max_safe_payload(self, service_area):

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


    def sortie_total_time(self, n_boxes, service_area):

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