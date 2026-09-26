import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.session_resources import (
    assign_session_energy_components,
    build_relay_session_table,
)


PARAMS = SimpleNamespace(energy_capacity_kwh=3.2, safety_margin=0.2,
                         full_charge_time_s=3600)


def relay_rows(count=3, energy=.01):
    return pd.DataFrame([{
        "relay_session_id": "R01-RS001", "relay_uav_id": "R01",
        "candidate_id": "C1", "dispatch_time_s": 0,
        "return_time_s": 100, "uav_release_time_s": 100,
        "outbound_energy_kWh": .1, "return_energy_kWh": .1,
        "service_energy_kWh": energy, "gap_id": f"G{i}",
    } for i in range(count)])


class SessionResourcesTests(unittest.TestCase):
    def test_three_gap_rows_are_one_session_with_flight_counted_once(self):
        sessions = build_relay_session_table(relay_rows(), PARAMS)
        self.assertEqual(len(sessions), 1)
        self.assertAlmostEqual(sessions.iloc[0].relay_session_energy_kWh, .23)
        self.assertAlmostEqual(sessions.iloc[0].outbound_energy_kWh, .1)
        self.assertAlmostEqual(sessions.iloc[0].return_energy_kWh, .1)

    def test_session_has_one_energy_component(self):
        sessions = build_relay_session_table(relay_rows(), PARAMS)
        assigned, fits = assign_session_energy_components(sessions, 6)
        self.assertTrue(fits)
        self.assertEqual(assigned.iloc[0].energy_component_id, "E01")

    def test_session_energy_above_2_56_kwh_violates_safety_soc(self):
        sessions = build_relay_session_table(relay_rows(1, energy=2.4), PARAMS)
        self.assertGreater(sessions.iloc[0].relay_session_energy_kWh, 2.56)
        self.assertLess(sessions.iloc[0].end_soc, PARAMS.safety_margin)

    def test_six_concurrent_sessions_fit_and_seven_do_not(self):
        base = build_relay_session_table(relay_rows(1), PARAMS).iloc[0].to_dict()
        six = pd.DataFrame([{**base, "relay_session_id": f"S{i}",
                             "relay_uav_id": f"R{i:02d}"} for i in range(6)])
        seven = pd.concat([six, pd.DataFrame([{**base, "relay_session_id": "S6",
                                               "relay_uav_id": "R07"}])], ignore_index=True)
        self.assertTrue(assign_session_energy_components(six, 6)[1])
        self.assertFalse(assign_session_energy_components(seven, 6)[1])


if __name__ == "__main__":
    unittest.main()
