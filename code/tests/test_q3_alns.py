import sys
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.alns_search import TransportSearch
from src.q3.pareto import dominates, update_archive
from src.q3.step8_acceptance import OUTPUTS, _output_hashes, _verify_solver_inputs


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

    def test_existing_exact_solution_is_excluded(self):
        search = self.make_search()
        search.excluded_sets.append(frozenset([0, 1]))
        self.assertEqual(search.repair([], 2), (2,))

    def test_destroy_all_operators_preserve_subset(self):
        search = self.make_search()
        for op in search.operators:
            kept = search.destroy((0, 1), op)
            self.assertLess(len(kept), 2)
            self.assertTrue(set(kept) <= {0, 1})

    def test_weighted_proxy_uses_selected_objective_direction(self):
        search = self.make_search()
        for occurrence in search.occ:
            occurrence.uav_type = 'small'
        search.problem['uav_ids'] = {'small': ['U1']}
        search.objective_weights = search._validate_objective_weights((0, 0, 0, 1))
        self.assertLess(search.score([2]), search.score([0, 1]))
        search.objective_weights = search._validate_objective_weights((0, 0, 1, 0))
        self.assertLess(search.score([2]), search.score([0, 1]))

    def test_pareto_archive_keeps_only_nondominated_solutions(self):
        a = {'objectives': {'f1': 1, 'cmax': 5, 'energy': 5, 'n': 3}}
        b = {'objectives': {'f1': 2, 'cmax': 6, 'energy': 6, 'n': 4}}
        c = {'objectives': {'f1': 0, 'cmax': 7, 'energy': 4, 'n': 2}}
        archive = update_archive([], a)
        archive = update_archive(archive, b)
        self.assertEqual(archive, [a])
        archive = update_archive(archive, c)
        self.assertEqual(len(archive), 2)
        self.assertTrue(dominates(a['objectives'].values(), b['objectives'].values()))

    def test_acceptance_hashes_use_solver_and_candidate_directories(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            solver_inputs = root / 'solver_inputs'
            candidate = root / 'candidate'
            solver_inputs.mkdir()
            candidate.mkdir()
            (solver_inputs / 'patterns.csv').write_bytes(b'solver input')
            recorded = hashlib.sha256(b'solver input').hexdigest()
            _verify_solver_inputs({'patterns.csv': recorded}, solver_inputs)

            for name in OUTPUTS:
                (candidate / name).write_bytes(('candidate:' + name).encode())
            hashes = _output_hashes(candidate)
            self.assertEqual(set(hashes), set(OUTPUTS))
            self.assertEqual(
                hashes[OUTPUTS[0]],
                hashlib.sha256(('candidate:' + OUTPUTS[0]).encode()).hexdigest(),
            )


if __name__ == '__main__':
    unittest.main()
