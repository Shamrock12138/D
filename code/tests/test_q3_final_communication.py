import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.q3.final_communication_validator as validator


def sample(direct):
    return SimpleNamespace(tau=1.0, direct=direct, x=10.0, y=20.0, z=30.0,
                           phase="cruise")


class FineCommunicationTests(unittest.TestCase):
    def validate(self, occurrences, transport_rows, relay_rows, covered=True):
        cache = SimpleNamespace(build_nodes=lambda: None,
                                 build_segments=lambda *args, **kwargs: None,
                                 close=lambda: None)
        sites = pd.DataFrame([{
            "candidate_id": "C1", "relay_g01_margin_db": 3.0,
            "dem_row": 0, "dem_col": 0, "lon": 0.0, "lat": 0.0,
            "absolute_height": 20.0,
        }])
        packed = np.array([[1 if covered else 0]], dtype=np.uint8)
        with patch.object(validator, "DirectProfileCache", return_value=cache), \
             patch.object(validator, "required_segment_keys", return_value=[]), \
             patch.object(validator, "_assemble_pattern_profile_with_sources",
                          side_effect=lambda occ, _cache: [sample(occ.direct)]), \
             patch.object(validator, "_actual_coverage", return_value=(packed, {})), \
             patch.object(validator, "load_relay_link_parameters", return_value=object()), \
             patch.object(validator, "DemTerrain", return_value=SimpleNamespace(close=lambda: None)), \
             patch.object(validator.pd, "read_csv", return_value=sites):
            return validator.validate_fine_communication(
                {"occurrences": occurrences}, pd.DataFrame(transport_rows),
                pd.DataFrame(relay_rows), dt=1.0)

    def test_repeated_pattern_occurrences_are_isolated_by_sortie_id(self):
        occurrences = [SimpleNamespace(sortie_id="P-1", pattern_id="P", direct=False),
                       SimpleNamespace(sortie_id="P-2", pattern_id="P", direct=False)]
        transport = [{"sortie_id": "P-1", "start_time_s": 0.0},
                     {"sortie_id": "P-2", "start_time_s": 100.0}]
        relay = [{"sortie_id": "P-1", "candidate_id": "C1",
                  "service_start_s": 0.0, "service_end_s": 5.0}]
        result = self.validate(occurrences, transport, relay)
        self.assertEqual(result["unserved_time_samples"], 1)
        self.assertFalse(result["all_pass"])

    def test_direct_sample_passes_without_relay(self):
        result = self.validate(
            [SimpleNamespace(sortie_id="S1", pattern_id="P", direct=True)],
            [{"sortie_id": "S1", "start_time_s": 0.0}], [],
        )
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["direct_samples"], 1)
        self.assertEqual(result["relay_required_samples"], 0)

    def test_outage_without_active_relay_fails(self):
        result = self.validate(
            [SimpleNamespace(sortie_id="S1", pattern_id="P", direct=False)],
            [{"sortie_id": "S1", "start_time_s": 0.0}], [],
        )
        self.assertFalse(result["all_pass"])
        self.assertEqual(result["unserved_time_samples"], 1)

    def test_active_but_uncovered_relay_fails(self):
        result = self.validate(
            [SimpleNamespace(sortie_id="S1", pattern_id="P", direct=False)],
            [{"sortie_id": "S1", "start_time_s": 0.0}],
            [{"sortie_id": "S1", "candidate_id": "C1",
              "service_start_s": 0.0, "service_end_s": 5.0}],
            covered=False,
        )
        self.assertFalse(result["all_pass"])
        self.assertEqual(result["uncovered_link_samples"], 1)


if __name__ == "__main__":
    unittest.main()
