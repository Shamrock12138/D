import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.relay_master import intrinsic_status, prefix_workload, relay_conflict_core


class RelayMasterTests(unittest.TestCase):
    def test_intrinsic_overlap_and_nonoverlap(self):
        options = {str(i): [(0, 10)] for i in range(3)}
        self.assertEqual(intrinsic_status(list(options), options), "INFEASIBLE")
        options['2'].append((10, 20))
        self.assertIn(intrinsic_status(list(options), options), ('OPTIMAL', 'FEASIBLE'))

    def test_prefix_bound_exhaustive(self):
        options = {'g': [(-3, 4), (2, 8)]}
        for latest in range(10):
            for t in range(20):
                bound = prefix_workload(['g'], options, latest, t)
                actual = [max(0, min(s+b, t)-max(s+a, 0))
                          for a, b in options['g'] for s in range(latest+1)
                          if s+a >= 0]
                self.assertEqual(bound, min(actual))

    def test_core_does_not_require_full_class_cover(self):
        occurrences = [SimpleNamespace(sortie_id=str(i), gap_ids=[str(i)],
                        latest_start_s=0, duration_s=10, class_counts={},
                        delivery_offsets={}) for i in range(4)]
        problem = {'occurrences': occurrences, 'horizon_s': 100,
                   'class_params': {}, 'relay': pd.DataFrame([
                       {'gap_id': str(i), 'dispatch_offset_s': 0 if i < 3 else 20,
                        'relay_uav_occupancy_s': 10} for i in range(4)])}
        result = relay_conflict_core(problem, ['0', '1', '2', '3'])
        self.assertEqual(result['status'], 'INFEASIBLE')
        self.assertTrue({'0', '1', '2'}.issubset(result['sortie_ids']))
        self.assertNotEqual(relay_conflict_core(problem, ['0', '1'])['status'], 'INFEASIBLE')


if __name__ == '__main__':
    unittest.main()
