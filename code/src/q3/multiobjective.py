"""Weighted ALNS runs with exact joint CP-SAT validation and a Pareto archive."""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.q3.alns_search import run_alns
from src.q3.cp_sat_scheduler import DATA
from src.q3.objectives import OBJECTIVE_NAMES
from src.q3.pareto import update_archive


WEIGHT_VECTORS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.4, 0.2, 0.2, 0.2),
    (0.2, 0.4, 0.2, 0.2),
    (0.2, 0.2, 0.4, 0.2),
    (0.2, 0.2, 0.2, 0.4),
    (0.25, 0.25, 0.25, 0.25),
)


def _summary_row(solution_id, candidate, weights):
    solution_dir = Path(candidate['solution_directory'])
    resource = pd.read_csv(solution_dir / 'q3_joint_resource_summary.csv', encoding='utf-8-sig').iloc[0]
    objectives = candidate['objectives']
    return {
        'solution_id': solution_id,
        'weight_F1': weights[0], 'weight_Cmax': weights[1],
        'weight_E': weights[2], 'weight_N': weights[3],
        **{name: objectives[name] for name in OBJECTIVE_NAMES},
        'transport_sorties': int(resource['transport_sorties']),
        'relay_sessions': int(candidate['relay_sessions']),
        'transport_energy_kWh': float(resource['transport_energy_kWh']),
        'relay_energy_kWh': float(resource['relay_energy_kWh']),
        'validation_all_pass': bool(candidate['validation_all_pass']),
        'solution_directory': str(solution_dir),
    }


def run_multiobjective(iterations=200, wall_time_per_weight_s=180,
                        repair_time_s=1, joint_time_s=20, workers=4,
                        seed=2026, max_solutions_per_weight=5,
                        weight_vectors=WEIGHT_VECTORS, output_dir=None):
    """Search each weight direction and save all strictly feasible Pareto points.

    ``wall_time_per_weight_s`` is an independent budget for each weight vector;
    exact joint CP-SAT validation is run for every unique ALNS candidate by
    defaulting to ``batch_size=1``.
    """
    started = time.monotonic()
    weights = [tuple(map(float, row)) for row in weight_vectors]
    if not weights or any(len(row) != 4 or sum(row) <= 0 for row in weights):
        raise ValueError('Provide nonempty four-objective weight vectors')
    run_root = Path(output_dir) if output_dir else (
        DATA / 'q3_multiobjective' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run_root.mkdir(parents=True, exist_ok=False)
    manifest_path = run_root / 'manifest.json'
    summary_path = run_root / 'pareto_summary.csv'
    archive = []
    all_runs = []
    all_candidates = []
    excluded = []

    def save(status='SEARCHING'):
        rows = []
        for index, entry in enumerate(archive, start=1):
            rows.append(_summary_row(f'Q3-P{index:03d}', entry['candidate'], entry['weights']))
        pd.DataFrame(rows, columns=[
            'solution_id', 'weight_F1', 'weight_Cmax', 'weight_E', 'weight_N',
            *OBJECTIVE_NAMES, 'transport_sorties', 'relay_sessions',
            'transport_energy_kWh', 'relay_energy_kWh', 'validation_all_pass',
            'solution_directory',
        ]).to_csv(summary_path, index=False, encoding='utf-8-sig')
        manifest = {
            'status': status, 'algorithm': 'weighted_ALNS_exact_joint_CP_SAT_Pareto',
            'objective_names': list(OBJECTIVE_NAMES), 'weight_vectors': [list(w) for w in weights],
            'config': {'iterations_per_weight': int(iterations),
                       'wall_time_per_weight_s': float(wall_time_per_weight_s),
                       'repair_time_s': float(repair_time_s),
                       'joint_time_s': float(joint_time_s), 'workers': int(workers),
                       'seed': int(seed), 'max_solutions_per_weight': int(max_solutions_per_weight),
                       'batch_size': 1},
            'completed_weight_runs': len(all_runs), 'candidate_count': len(all_candidates),
            'pareto_size': len(archive), 'pareto_solution_ids': [f'Q3-P{i:03d}' for i in range(1, len(archive)+1)],
            'elapsed_wall_time_s': time.monotonic() - started,
            'run_directory': str(run_root), 'weight_runs': all_runs,
            'candidates': all_candidates,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    save()
    for weight_index, vector in enumerate(weights):
        report = run_alns(
            iterations=iterations, wall_time_s=wall_time_per_weight_s,
            repair_time_s=repair_time_s, joint_time_s=joint_time_s,
            workers=workers, seed=seed + weight_index, batch_size=1,
            max_solutions=max_solutions_per_weight, objective_weights=vector,
            excluded_sortie_sets=excluded,
        )
        all_runs.append({'weight_index': weight_index + 1, 'weights': list(vector),
                         'status': report['status'], 'run_directory': report['run_directory'],
                         'iterations': report['iterations'],
                         'unique_candidates': report['unique_candidates'],
                         'joint_attempts': len(report['attempts']),
                         'accepted_solutions': len(report['solutions'])})
        for candidate in report['solutions']:
            item = {'candidate': candidate, 'weights': vector,
                    'objectives': candidate['objectives']}
            all_candidates.append({
                'weight_index': weight_index + 1, 'weights': list(vector),
                'sortie_ids': candidate['sortie_ids'], 'status': candidate['status'],
                'validation_all_pass': candidate['validation_all_pass'],
                'objectives': candidate['objectives'],
                'solution_directory': candidate['solution_directory'],
            })
            excluded.append(candidate['sortie_ids'])
            if candidate['validation_all_pass'] is True:
                archive = update_archive(archive, item, objective_key='objectives')
        save()
    save(status='COMPLETE')
    return json.loads(manifest_path.read_text(encoding='utf-8'))
