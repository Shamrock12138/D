import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.alns_search import TransportSearch


class AlnsTests(unittest.TestCase):
    def make_search(self):
        occurrences = [SimpleNamespace(sortie_id=name, pattern_id=name, gap_ids=[],
                        latest_start_s=90, duration_s=10, class_counts=counts,
                        delivery_offsets={}, energy_kWh=1, visit_order=['S1'])
                       for name, counts in [('A', {'c1': 1}), ('B', {'c2': 1}),
                                            ('AB', {'c1': 1, 'c2': 1})]]
        return TransportSearch({'occurrences': occurrences,
            'class_supply': {'c1': 1, 'c2': 1}, 'horizon_s': 100,
            'class_params': {c: {'hard_deadline_s': float('inf')} for c in ('c1','c2')},
            'relay': pd.DataFrame(columns=['gap_id','dispatch_offset_s','relay_uav_occupancy_s'])})

    def test_repair_conservation(self):
        search = self.make_search()
        result = search.repair([0], 2)
        self.assertEqual(result, (0, 1))
        self.assertTrue(search.eligible(result))

    def test_proven_core_excluded(self):
        search = self.make_search()
        search.cores.append(frozenset([2]))
        self.assertEqual(search.repair([], 2), (0, 1))

    def test_destroy_all_operators_preserve_subset(self):
        search = self.make_search()
        for op in search.operators:
            kept = search.destroy((0, 1), op)
            self.assertLess(len(kept), 2)
            self.assertTrue(set(kept) <= {0, 1})


if __name__ == '__main__':
    unittest.main()
