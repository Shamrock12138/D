"""Adaptive transport-neighbourhood search with exact joint CP-SAT validation.

Repair uses a small time-limited class-cover CP model, not the timed transport
Master. Congestion estimates guide search only and never certify feasibility.
"""
import json
import math
import random
import time
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from src.q3.bootstrap import subset_problem
from src.q3.cp_sat_scheduler import DATA, prepare_q3_problem, solve_q3_joint, write_step8_outputs
from src.q3.decomposition import _input_hashes
from src.q3.relay_master import latest_start, relay_intervals


REPAIR_COST_SCALE = 1_000_000


def integer_repair_cost(value, jitter=1.0):
    return max(1, round(REPAIR_COST_SCALE * max(0.0, float(value)) * float(jitter)))


class TransportSearch:
    operators = ('random', 'relay_load', 'gaps', 'congestion', 'deadline', 'related')

    def __init__(self, problem, seed=42, objective_weights=None):
        self.problem = problem
        self.rng = random.Random(seed)
        self.seed = seed
        self.occ = problem['occurrences']
        self.by_id = {o.sortie_id: i for i, o in enumerate(self.occ)}
        self.classes = list(problem['class_supply'])
        self.supply = np.array([problem['class_supply'][c] for c in self.classes], dtype=int)
        self.counts = np.array([[o.class_counts.get(c, 0) for c in self.classes]
                                for o in self.occ], dtype=int)
        options = relay_intervals(problem)
        self.allowed, self.base, self.loads, self.latest = [], [], [], []
        self.occ_energy = np.asarray([float(o.energy_kWh) for o in self.occ], dtype=float)
        self.occ_duration = np.asarray([float(o.duration_s) for o in self.occ], dtype=float)
        self.uav_types = sorted({str(o.uav_type) for o in self.occ})
        self.uav_type_index = {typ: index for index, typ in enumerate(self.uav_types)}
        self.occ_type_index = np.asarray(
            [self.uav_type_index[str(o.uav_type)] for o in self.occ], dtype=int)
        self.uav_count_by_type = np.asarray([
            max(1, len(problem['uav_ids'].get(typ, ()))) for typ in self.uav_types
        ], dtype=float)
        self.occ_soft_terms = []
        self.grid = np.arange(0, problem['horizon_s'], 250) + 125
        self.footprint = np.zeros((len(self.occ), len(self.grid)), dtype=float)
        self.copies = defaultdict(list)
        params = problem['class_params']
        for i, o in enumerate(self.occ):
            self.copies[o.pattern_id].append(i)
            upper = latest_start(problem, o)
            self.latest.append(upper)
            # Relay feasibility belongs to the exact shared-session joint model.
            usable = upper >= 0
            load = 0
            for gap in o.gap_ids:
                valid = [(a, b) for a, b in options.get(gap, ()) if upper+a >= 0]
                if not valid:
                    continue
                a, b = min(valid, key=lambda ab: ab[1]-ab[0])
                load += b-a
                self.footprint[i] += ((self.grid >= upper+a) & (self.grid < upper+b))
            self.allowed.append(usable)
            self.loads.append(load)
            self.base.append(1000 + 2000*len(o.gap_ids) + 2*load + o.energy_kWh)
            terms = []
            for class_id, amount in o.class_counts.items():
                item = params[class_id]
                expected = float(item.get('expected_time_s', float('nan')))
                if math.isfinite(item.get('hard_deadline_s', float('inf'))) or not math.isfinite(expected):
                    continue
                delta = float(o.delivery_offsets.get(class_id, 0.0)) - expected
                terms.append((int(amount) * int(item.get('priority', 0)), delta))
            self.occ_soft_terms.append(tuple(terms))
        self.allowed = np.array(self.allowed)
        self.base = np.array(self.base)
        self.weights = {op: 1.0 for op in self.operators}
        self.objective_weights = self._validate_objective_weights(objective_weights)
        self.objective_scales = self._build_objective_scales()
        relay = problem.get('relay')
        self.min_relay_energy_by_gap = {}
        if relay is not None and len(relay) and 'relay_energy_kWh' in relay:
            self.min_relay_energy_by_gap = (
                relay.assign(_gap_id=relay['gap_id'].astype(str))
                .groupby('_gap_id')['relay_energy_kWh'].min().astype(float).to_dict())
        self.occ_gap_ids = [tuple(map(str, o.gap_ids)) for o in self.occ]
        self.occ_relay_load = np.asarray(self.loads, dtype=float)
        self.cores = []
        self.excluded_sets = []

    @staticmethod
    def _validate_objective_weights(weights):
        if weights is None:
            return None
        values = tuple(float(value) for value in weights)
        if len(values) != 4 or any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError("Objective weights must be four finite nonnegative values")
        if sum(values) <= 0:
            raise ValueError("At least one objective weight must be positive")
        total = sum(values)
        return tuple(value / total for value in values)

    def _build_objective_scales(self):
        f1_scale = 0.0
        for class_id, amount in self.problem['class_supply'].items():
            item = self.problem['class_params'][class_id]
            deadline = item.get('hard_deadline_s', float('inf'))
            expected = item.get('expected_time_s', float('nan'))
            if math.isfinite(deadline) or not math.isfinite(expected):
                continue
            f1_scale += int(amount) * int(item.get('priority', 0)) * max(
                1.0, self.problem['horizon_s'] - float(expected))
        return (max(1.0, f1_scale), max(1.0, float(self.problem['horizon_s'])),
                100.0, 40.0)

    def canonical(self, selected):
        multiplicities = Counter(self.occ[i].pattern_id for i in selected)
        return tuple(sorted(i for p, n in multiplicities.items() for i in self.copies[p][:n]))

    def exact_cover(self, selected):
        return (len(selected) == len(set(selected)) and
                np.array_equal(self.counts[list(selected)].sum(axis=0), self.supply))

    def eligible(self, selected):
        signature = frozenset(selected)
        return (self.exact_cover(selected) and all(self.allowed[list(selected)])
                and signature not in self.excluded_sets)

    def objective_estimates(self, selected):
        """Normalized transport-set proxies; never used to certify feasibility."""
        if not selected:
            return (0.0, 0.0, 0.0, 0.0)
        indices = np.asarray(selected, dtype=int)
        f1 = 0.0
        workload = np.bincount(self.occ_type_index[indices],
                               weights=self.occ_duration[indices],
                               minlength=len(self.uav_types))
        predicted_start = 0.5 * workload / self.uav_count_by_type
        for i in selected:
            start_hat = predicted_start[self.occ_type_index[i]]
            f1 += sum(weight * max(0.0, start_hat + offset_minus_expected)
                      for weight, offset_minus_expected in self.occ_soft_terms[i])
        transport_cmax = max((workload / self.uav_count_by_type), default=0.0)
        relay_work = float(self.occ_relay_load[indices].sum())
        relay_cmax = relay_work / 2.0
        transport_energy = float(self.occ_energy[indices].sum())
        gaps = {gap for i in selected for gap in self.occ_gap_ids[i]}
        relay_energy = sum(self.min_relay_energy_by_gap.get(gap, 0.0) for gap in gaps)
        estimates = (f1, max(transport_cmax, relay_cmax), transport_energy + relay_energy,
                    len(selected) + len(gaps))
        return tuple(value / scale for value, scale in zip(estimates, self.objective_scales))

    def score(self, selected, objective_weights=None):
        profile = self.footprint[list(selected)].sum(axis=0)
        weights = self.objective_weights if objective_weights is None else self._validate_objective_weights(objective_weights)
        if weights is None:
            return float(self.base[list(selected)].sum() + 5000*np.maximum(0, profile-2).sum())
        objective_proxy = float(np.dot(weights, self.objective_estimates(selected)))
        congestion = float(np.maximum(0, profile-2).sum() / max(1, len(self.grid)))
        return objective_proxy + 0.1 * congestion

    def destroy(self, selected, operator):
        selected = list(selected)
        if not selected:
            return []
        n = max(1, math.ceil(len(selected)*self.rng.uniform(.2, .55)))
        if operator == 'random':
            removed = self.rng.sample(selected, n)
        else:
            profile = self.footprint[selected].sum(axis=0)
            pivot = self.rng.choice(selected)
            services = set(self.occ[pivot].visit_order)
            def priority(i):
                if operator == 'relay_load': return self.loads[i]
                if operator == 'gaps': return len(self.occ[i].gap_ids)
                if operator == 'deadline': return -self.latest[i]
                if operator == 'related': return len(services & set(self.occ[i].visit_order))
                return float(np.dot(self.footprint[i], np.maximum(0, profile-2)))
            removed = sorted(selected, key=lambda i: (priority(i), self.rng.random()), reverse=True)[:n]
        return [i for i in selected if i not in removed]

    def repair(self, kept, seconds=1.0):
        residual = self.supply-self.counts[kept].sum(axis=0)
        if np.any(residual < 0): return None
        candidates = np.flatnonzero(self.allowed & np.all(self.counts <= residual, axis=1))
        candidates = [int(i) for i in candidates if i not in kept]
        model = cp_model.CpModel()
        variables = {i: model.NewBoolVar(f'x{i}') for i in candidates}
        for c, demand in enumerate(residual):
            model.Add(sum(int(self.counts[i, c])*x for i, x in variables.items()
                          if self.counts[i, c]) == int(demand))
        kept_set = set(kept)
        for forbidden in self.excluded_sets:
            if kept_set <= forbidden:
                remaining = forbidden - kept_set
                if not remaining:
                    model.AddBoolOr([])
                elif remaining <= set(variables):
                    model.Add(sum(variables[i] for i in remaining) <= len(remaining)-1)
        profile = self.footprint[kept].sum(axis=0)
        if self.objective_weights is None:
            raw_costs = {i: self.base[i]+5000*np.maximum(
                0, profile+self.footprint[i]-2).sum() for i in candidates}
        else:
            base_score = self.score(kept)
            raw_costs = {i: max(1e-6, self.score(kept+[i])-base_score)
                         for i in candidates}
        costs = {i: integer_repair_cost(value, self.rng.uniform(.95, 1.05))
                 for i, value in raw_costs.items()}
        model.Minimize(sum(costs[i]*x for i, x in variables.items()))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = self.rng.randrange(2**30)
        status = solver.Solve(model)
        if status == cp_model.MODEL_INVALID: raise RuntimeError(solver.ResponseStats())
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None
        candidate = self.canonical(kept+[i for i, x in variables.items() if solver.Value(x)])
        return candidate if self.eligible(candidate) else None

    def initial_seeds(self):
        seeds = []
        manifest = DATA/'q3_step8_decomposition_manifest.json'
        if manifest.exists():
            for attempt in json.loads(manifest.read_text(encoding='utf-8')).get('attempts', []):
                ids = attempt.get('sortie_ids', [])
                if ids and all(s in self.by_id for s in ids):
                    seeds.append(tuple(self.by_id[s] for s in ids))
        for path in sorted(DATA.glob('Q2_anchor_*_selected.csv')):
            frame = pd.read_csv(path)
            col = 'sortie_id' if 'sortie_id' in frame else 'task_id'
            if col in frame:
                ids = frame[col].astype(str).tolist()
                if ids and all(s in self.by_id for s in ids):
                    seeds.append(tuple(self.by_id[s] for s in ids))
        return [self.canonical(s) for s in seeds if self.exact_cover(s)]


def run_alns(iterations=200, wall_time_s=300, repair_time_s=1, joint_time_s=20,
             workers=4, seed=42, batch_size=5, max_solutions=3,
             objective_weights=None, excluded_sortie_sets=(), polish_time_s=0):
    started = time.monotonic()
    hashes = _input_hashes()
    problem = prepare_q3_problem(tier='all')
    search = TransportSearch(problem, seed, objective_weights=objective_weights)
    seeds = search.initial_seeds()
    baseline_ids = []
    frozen_transport = DATA/'q3_step8_frozen'/'q3_joint_transport_schedule.csv'
    if frozen_transport.is_file():
        baseline_ids = pd.read_csv(frozen_transport, encoding='utf-8-sig')['sortie_id'].astype(str).tolist()
    elif (DATA/'q3_joint_transport_schedule.csv').is_file():
        baseline_ids = pd.read_csv(DATA/'q3_joint_transport_schedule.csv', encoding='utf-8-sig')['sortie_id'].astype(str).tolist()
    if baseline_ids and all(s in search.by_id for s in baseline_ids):
        baseline = search.canonical([search.by_id[s] for s in baseline_ids])
        search.excluded_sets.append(frozenset(baseline))
    for excluded_ids in excluded_sortie_sets:
        if excluded_ids and all(str(s) in search.by_id for s in excluded_ids):
            excluded = frozenset(search.canonical(
                [search.by_id[str(s)] for s in excluded_ids]))
            if excluded not in search.excluded_sets:
                search.excluded_sets.append(excluded)
    admissible_seeds = [s for s in seeds if search.eligible(s)]
    current = min(admissible_seeds or seeds, key=search.score) if seeds else ()
    run_dir = DATA/'q3_alns_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {'status': 'SEARCHING', 'algorithm': 'ALNS_CP_SAT', 'seed': seed,
              'input_sha256': hashes, 'iterations': 0, 'generated': 0,
              'unique_candidates': 0, 'repair_wall_time_s': 0, 'attempts': [],
              'config': dict(iterations=iterations, wall_time_s=wall_time_s,
                             repair_time_s=repair_time_s, joint_time_s=joint_time_s,
                             workers=workers, batch_size=batch_size,
                             objective_weights=search.objective_weights,
                             polish_time_s=polish_time_s),
              'initial_exact_cover_seeds': len(seeds),
              'forbidden_occurrences': int((~search.allowed).sum()),
              'max_additional_solutions': max_solutions,
              'baseline_excluded': bool(search.excluded_sets),
              'run_directory': str(run_dir), 'solutions': [], 'archive_errors': []}
    pending, visited = {}, set()
    path = run_dir/'manifest.json'
    def save():
        report['wall_time_s'] = time.monotonic()-started
        report['operator_weights'] = search.weights
        path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    save()
    for iteration in range(1, iterations+1):
        remaining = wall_time_s-(time.monotonic()-started)
        if remaining <= 0: break
        op = search.rng.choices(search.operators, weights=list(search.weights.values()))[0]
        kept = search.destroy(current, op) if current else []
        kept = [i for i in kept if search.allowed[i]]
        if iteration % 10 == 0: kept = []  # diversify beyond the current basin
        before = time.monotonic()
        candidate = search.repair(kept, min(repair_time_s, remaining))
        report['repair_wall_time_s'] += time.monotonic()-before
        report['iterations'] = iteration
        reward = .2
        if candidate is not None:
            report['generated'] += 1
            value = search.score(candidate)
            previous = search.score(current) if current else float('inf')
            if (not search.eligible(current) or value < previous
                    or search.rng.random() < math.exp(min(0, (previous-value)/max(100, previous*.05)))):
                current = candidate
                reward = 2 if value < previous else 1
            if candidate not in visited and candidate not in pending:
                pending[candidate] = value
                report['unique_candidates'] += 1
        search.weights[op] = .8*search.weights[op]+.2*reward
        if pending and (iteration % batch_size == 0 or iteration == iterations):
            candidate = min(pending, key=pending.get)
            del pending[candidate]
            if not search.eligible(candidate):
                save()
                continue
            visited.add(candidate)
            remaining = wall_time_s-(time.monotonic()-started)
            if remaining <= 0: break
            ids = [search.occ[i].sortie_id for i in candidate]
            solution = solve_q3_joint(tier='all', time_limit_s=min(joint_time_s, remaining),
                        workers=workers, random_seed=seed, problem=subset_problem(problem, ids),
                        feasibility_only=True, allow_relay_sharing=True,
                        fixed_sortie_ids=ids)
            attempt = {'iteration': iteration, 'sortie_ids': ids, 'status': solution['status'],
                       'wall_time_s': solution['wall_time_s'], 'score': search.score(candidate)}
            report['attempts'].append(attempt)
            print(f"ALNS #{iteration}: {len(ids)} sorties, joint {solution['status']}, "
                  f"generated={report['generated']}", flush=True)
            if solution['status'] == 'MODEL_INVALID': raise RuntimeError('Invalid joint model')
            if solution['status'] in ('FEASIBLE', 'OPTIMAL'):
                if polish_time_s > 0:
                    remaining = wall_time_s-(time.monotonic()-started)
                    if remaining > 0:
                        polished = solve_q3_joint(
                            tier='all', time_limit_s=min(polish_time_s, remaining),
                            workers=workers, random_seed=seed + iteration,
                            problem=subset_problem(problem, ids), feasibility_only=False,
                            hint=solution, allow_relay_sharing=True,
                            objective_weights=(search.objective_weights or (0.25,)*4),
                            fixed_sortie_ids=ids)
                        attempt['polish_status'] = polished['status']
                        attempt['polish_wall_time_s'] = polished.get('wall_time_s', 0.0)
                        if polished['status'] in ('FEASIBLE', 'OPTIMAL'):
                            solution = polished
                    else:
                        attempt['polish_status'] = 'SKIPPED_WALL_BUDGET'
                if _input_hashes() != hashes: raise RuntimeError('Inputs changed during ALNS')
                solution.update(input_sha256=hashes, alns_iteration=iteration)
                if solution.get('solve_mode') != 'weighted_polish':
                    solution.update(solve_mode='ALNS_CP_SAT', optimization_status='NOT_RUN')
                else:
                    solution['optimization_status'] = solution['status']
                ordinal = len(report['solutions']) + 1
                solution_dir = run_dir/f'solution_{iteration:04d}'
                try:
                    write_step8_outputs(solution, output_dir=solution_dir)
                    from src.q3.step8_acceptance import accept_step8
                    acceptance = accept_step8(freeze=False, data_dir=solution_dir)
                    (solution_dir/'acceptance.json').write_text(
                        json.dumps(acceptance, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                except Exception as exc:
                    attempt['archive_error'] = f'{type(exc).__name__}: {exc}'
                    report['archive_errors'].append({
                        'iteration': iteration, 'solution_directory': str(solution_dir),
                        'error': attempt['archive_error'],
                    })
                    print(f"ALNS archive failed at iteration {iteration}: {attempt['archive_error']}",
                          flush=True)
                    save()
                    continue
                chosen_set = frozenset(candidate)
                search.excluded_sets.append(chosen_set)
                report['solutions'].append({
                    'index': ordinal, 'iteration': iteration,
                    'sortie_ids': ids, 'solution_directory': str(solution_dir),
                    'status': acceptance['status'],
                    'validation_all_pass': acceptance['validation']['all_pass'],
                    'objectives': acceptance['objectives'],
                    'transport_sorties': len(solution['transport']),
                    'relay_gap_jobs': len(solution['relay']),
                    'relay_sessions': int(solution['relay']['relay_session_id'].nunique())
                        if 'relay_session_id' in solution['relay'] else len(solution['relay']),
                    'search_source': 'weighted_alns',
                })
                report['distinct_additional_solutions'] = len(report['solutions'])
                print(f"ALNS archived distinct solution #{ordinal} at iteration {iteration}", flush=True)
                if len(report['solutions']) >= max_solutions:
                    report['status'] = 'SOLUTION_LIMIT_REACHED'
                    save()
                    return report
            if solution['status'] == 'INFEASIBLE':
                attempt['core'] = {
                    'status': 'DISABLED',
                    'reason': 'exclusive relay core is invalid under same-site sharing',
                }
            # UNKNOWN is only visited within this run, never a proven exclusion.
        save()
    report['status'] = 'FEASIBLE_SOLUTIONS' if report['solutions'] else 'UNKNOWN'
    report['reason'] = 'Search budget exhausted before requested distinct solution count'
    save()
    return report
