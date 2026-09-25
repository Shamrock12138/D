"""Adaptive transport-neighbourhood search with exact joint CP-SAT validation.

Repair uses a small time-limited class-cover CP model, not the timed transport
Master. Congestion estimates guide search only and never certify feasibility.
"""
import json
import math
import random
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from src.q3.bootstrap import subset_problem
from src.q3.cp_sat_scheduler import DATA, prepare_q3_problem, solve_q3_joint, write_step8_outputs
from src.q3.decomposition import _input_hashes
from src.q3.relay_master import latest_start, relay_intervals


class TransportSearch:
    operators = ('random', 'relay_load', 'gaps', 'congestion', 'deadline', 'related')

    def __init__(self, problem, seed=42):
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
        self.grid = np.arange(0, problem['horizon_s'], 250) + 125
        self.footprint = np.zeros((len(self.occ), len(self.grid)), dtype=float)
        self.copies = defaultdict(list)
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
        self.allowed = np.array(self.allowed)
        self.base = np.array(self.base)
        self.weights = {op: 1.0 for op in self.operators}
        self.cores = []

    def canonical(self, selected):
        multiplicities = Counter(self.occ[i].pattern_id for i in selected)
        return tuple(sorted(i for p, n in multiplicities.items() for i in self.copies[p][:n]))

    def exact_cover(self, selected):
        return (len(selected) == len(set(selected)) and
                np.array_equal(self.counts[list(selected)].sum(axis=0), self.supply))

    def eligible(self, selected):
        return self.exact_cover(selected) and all(self.allowed[list(selected)])

    def score(self, selected):
        profile = self.footprint[list(selected)].sum(axis=0)
        return float(self.base[list(selected)].sum() + 5000*np.maximum(0, profile-2).sum())

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
        profile = self.footprint[kept].sum(axis=0)
        costs = {i: int((self.base[i]+5000*np.maximum(0, profile+self.footprint[i]-2).sum())
                        * self.rng.uniform(.5, 1.5)) for i in candidates}
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
             workers=4, seed=42, batch_size=5):
    started = time.monotonic()
    hashes = _input_hashes()
    problem = prepare_q3_problem(tier='all')
    search = TransportSearch(problem, seed)
    seeds = search.initial_seeds()
    admissible_seeds = [s for s in seeds if search.eligible(s)]
    current = min(admissible_seeds or seeds, key=search.score) if seeds else ()
    report = {'status': 'UNKNOWN', 'algorithm': 'ALNS_CP_SAT', 'seed': seed,
              'input_sha256': hashes, 'iterations': 0, 'generated': 0,
              'unique_candidates': 0, 'repair_wall_time_s': 0, 'attempts': [],
              'config': dict(iterations=iterations, wall_time_s=wall_time_s,
                             repair_time_s=repair_time_s, joint_time_s=joint_time_s,
                             workers=workers, batch_size=batch_size),
              'initial_exact_cover_seeds': len(seeds),
              'forbidden_occurrences': int((~search.allowed).sum())}
    pending, visited = {}, set()
    path = DATA/'q3_alns_manifest.json'
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
                        feasibility_only=True, allow_relay_sharing=True)
            attempt = {'iteration': iteration, 'sortie_ids': ids, 'status': solution['status'],
                       'wall_time_s': solution['wall_time_s'], 'score': search.score(candidate)}
            report['attempts'].append(attempt)
            print(f"ALNS #{iteration}: {len(ids)} sorties, joint {solution['status']}, "
                  f"generated={report['generated']}", flush=True)
            if solution['status'] == 'MODEL_INVALID': raise RuntimeError('Invalid joint model')
            if solution['status'] in ('FEASIBLE', 'OPTIMAL'):
                if _input_hashes() != hashes: raise RuntimeError('Inputs changed during ALNS')
                solution.update(input_sha256=hashes, solve_mode='ALNS_CP_SAT',
                                optimization_status='NOT_RUN', alns_iteration=iteration)
                write_step8_outputs(solution)
                report['status'] = 'FEASIBLE'
                save()
                return report
            if solution['status'] == 'INFEASIBLE':
                attempt['core'] = {
                    'status': 'DISABLED',
                    'reason': 'exclusive relay core is invalid under same-site sharing',
                }
            # UNKNOWN is only visited within this run, never a proven exclusion.
        save()
    report['reason'] = 'Search budget exhausted; no validated joint feasible schedule'
    save()
    return report
