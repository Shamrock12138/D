import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Q4 import relay_sessions


class Q4RelaySessionTests(unittest.TestCase):
    def test_sharing_rows_collapse_to_one_physical_session(self):
        q3 = {"relay": [
            {"relay_uav_id": "R01", "relay_session_id": "R01-RS001", "_sorties": ("A",),
             "dispatch_time_s": "0", "uav_release_time_s": "30", "energy_release_time_s": "20"},
            {"relay_uav_id": "R01", "relay_session_id": "R01-RS001", "_sorties": ("B",),
             "dispatch_time_s": "10", "uav_release_time_s": "25", "energy_release_time_s": "25"},
        ]}
        sessions = relay_sessions(q3)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["_sorties"], ("A", "B"))
        self.assertEqual(sessions[0]["dispatch_time_s"], 0)
        self.assertEqual(sessions[0]["uav_release_time_s"], 30)

    def test_legacy_rows_remain_distinct_sessions(self):
        q3 = {"relay": [
            {"relay_uav_id": "R01", "_sorties": ("A",),
             "dispatch_time_s": "0", "uav_release_time_s": "10", "energy_release_time_s": "10"},
            {"relay_uav_id": "R01", "_sorties": ("B",),
             "dispatch_time_s": "10", "uav_release_time_s": "20", "energy_release_time_s": "20"},
        ]}
        self.assertEqual(len(relay_sessions(q3)), 2)


if __name__ == "__main__":
    unittest.main()
