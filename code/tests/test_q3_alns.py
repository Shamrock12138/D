import sys
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.alns_search import (TransportSearch, accept_candidate,
                                alns_temperature, integer_repair_cost)
from src.q3.pareto import dominates, update_archive
import src.q3.step8_acceptance as step8_acceptance
from src.q3.step8_acceptance import OUTPUTS, _output_hashes, _verify_solver_inputs


class AlnsTests(unittest.TestCase):
    def make_search(self):
        occurrences = [SimpleNamespace(sortie_id=name, pattern_id=name, gap_ids=[],
                        latest_start_s=90, duration_s=10, class_counts=counts,
                        delivery_offsets={}, energy_kWh=1 if name != 'AB' else .5,
                        uav_type='small', visit_order=['S1'])
                       for name, counts in [('A', {'c1': 1}), ('B', {'c2': 1}),
                                            ('AB', {'c1': 1, 'c2': 1})]]
        return TransportSearch({'occurrences': occurrences,
            'class_supply': {'c1': 1, 'c2': 1}, 'horizon_s': 100,
            'class_params': {c: {'hard_deadline_s': float('inf'),
                                 'expected_time_s': 1.0, 'priority': 1}
                             for c in ('c1','c2')},
            'uav_ids': {'small': ['U1']},
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

    def test_excluded_baseline_remains_structurally_feasible_current(self):
        search = self.make_search()
        baseline = (0, 1)
        search.excluded_sets.append(frozenset(baseline))
        self.assertTrue(search.structurally_feasible(baseline))
        self.assertFalse(search.eligible(baseline))

    def test_local_destroy_removes_one_to_three_jobs(self):
        search = self.make_search()
        kept = search.destroy((0, 1), 'local')
        self.assertEqual(len(kept), 1)

    def test_annealing_temperature_cools_and_acceptance_uses_it(self):
        self.assertGreater(alns_temperature(1), alns_temperature(100))
        import random
        self.assertFalse(accept_candidate(1.01, 1.0, True, random.Random(1), 1000))

    def test_destroy_all_operators_preserve_subset(self):
        search = self.make_search()
        for op in search.operators:
            kept = search.destroy((0, 1), op)
            self.assertLess(len(kept), 2)
            self.assertTrue(set(kept) <= {0, 1})

    def test_weighted_proxy_uses_selected_objective_direction(self):
        search = self.make_search()
        search.objective_weights = search._validate_objective_weights((0, 0, 0, 1))
        self.assertLess(search.score([2]), search.score([0, 1]))
        search.objective_weights = search._validate_objective_weights((0, 0, 1, 0))
        self.assertLess(search.score([2]), search.score([0, 1]))
        search.objective_weights = search._validate_objective_weights((1, 0, 0, 0))
        self.assertLess(search.score([2]), search.score([0, 1]))

    def test_scaled_repair_cost_preserves_small_score_differences(self):
        self.assertGreater(integer_repair_cost(0.012), integer_repair_cost(0.003))
        self.assertGreater(integer_repair_cost(0.003), 1)

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

    def test_frozen_resolver_requires_complete_v2(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            old_data = step8_acceptance.DATA
            try:
                step8_acceptance.DATA = root
                v1 = root / 'q3_step8_frozen'
                v2 = root / 'q3_step8_frozen_v2'
                v1.mkdir()
                (v1 / OUTPUTS[0]).touch()
                (v1 / OUTPUTS[1]).touch()
                with self.assertRaises(FileNotFoundError):
                    step8_acceptance.resolve_frozen_dir()
                v2.mkdir()
                (v2 / OUTPUTS[0]).touch()
                (v2 / OUTPUTS[1]).touch()
                (v2 / step8_acceptance.SESSION_OUTPUT).touch()
                self.assertEqual(step8_acceptance.resolve_frozen_dir(), v2)
            finally:
                step8_acceptance.DATA = old_data


if __name__ == '__main__':
    unittest.main()
