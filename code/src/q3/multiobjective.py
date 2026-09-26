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
from src.q3.step8_acceptance import accept_step8


WEIGHT_LABELS = (
    'F1', 'Cmax', 'E', 'N', 'F1-heavy', 'Cmax-heavy', 'E-heavy', 'N-heavy', 'balanced',
)
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
    transport_energy = float(resource['transport_energy_kWh'])
    relay_session_energy = max(0.0, float(objectives['F3_total_energy_kWh']) - transport_energy)
    weight_values = tuple(weights) if weights is not None else (None, None, None, None)
    return {
        'solution_id': solution_id,
        'weight_F1': weight_values[0], 'weight_Cmax': weight_values[1],
        'weight_E': weight_values[2], 'weight_N': weight_values[3],
        **{name: objectives[name] for name in OBJECTIVE_NAMES},
        'transport_sorties': int(resource['transport_sorties']),
        'relay_sessions': int(candidate['relay_sessions']),
        'transport_energy_kWh': transport_energy,
        'relay_energy_kWh': relay_session_energy,
        'validation_all_pass': bool(candidate['validation_all_pass']),
        'search_source': candidate.get('search_source', 'weighted_alns'),
        'solution_directory': str(solution_dir),
    }


def run_multiobjective(iterations=200, wall_time_per_weight_s=180,
                        repair_time_s=1, joint_time_s=20, polish_time_s=30, workers=4,
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
    generated_total = unique_total = attempted_total = feasible_total = 0

    frozen_v2 = DATA / 'q3_step8_frozen_v2'
    frozen_dir = (frozen_v2 if (frozen_v2 / 'q3_joint_transport_schedule.csv').is_file()
                  else DATA / 'q3_step8_frozen')
    if (frozen_dir / 'q3_joint_transport_schedule.csv').is_file():
        baseline_acceptance = accept_step8(freeze=False, data_dir=frozen_dir)
        baseline_transport = pd.read_csv(
            frozen_dir / 'q3_joint_transport_schedule.csv', encoding='utf-8-sig')
        baseline_relay = pd.read_csv(
            frozen_dir / 'q3_joint_relay_schedule.csv', encoding='utf-8-sig')
        baseline = {
            'status': baseline_acceptance['status'],
            'validation_all_pass': baseline_acceptance['validation']['all_pass'],
            'objectives': baseline_acceptance['objectives'],
            'sortie_ids': baseline_transport['sortie_id'].astype(str).tolist(),
            'solution_directory': str(frozen_dir),
            'transport_sorties': len(baseline_transport),
            'relay_sessions': (int(baseline_relay['relay_session_id'].nunique())
                               if len(baseline_relay) and 'relay_session_id' in baseline_relay
                               else len(baseline_relay)),
            'search_source': 'frozen_baseline',
        }
        baseline_item = {'candidate': baseline, 'weights': None,
                         'objectives': baseline['objectives']}
        if baseline['validation_all_pass'] is True:
            archive = update_archive(archive, baseline_item,
                                     objective_key='objectives',
                                     objective_names=OBJECTIVE_NAMES)
            excluded.append(baseline['sortie_ids'])

    def save(status='SEARCHING'):
        rows = []
        for index, entry in enumerate(archive, start=1):
            rows.append(_summary_row(f'Q3-P{index:03d}', entry['candidate'], entry['weights']))
        pd.DataFrame(rows, columns=[
            'solution_id', 'weight_F1', 'weight_Cmax', 'weight_E', 'weight_N',
            *OBJECTIVE_NAMES, 'transport_sorties', 'relay_sessions',
            'transport_energy_kWh', 'relay_energy_kWh', 'validation_all_pass',
            'search_source', 'solution_directory',
        ]).to_csv(summary_path, index=False, encoding='utf-8-sig')
        manifest = {
            'status': status, 'algorithm': 'weighted_ALNS_exact_joint_CP_SAT_Pareto',
            'objective_names': list(OBJECTIVE_NAMES), 'weight_vectors': [list(w) for w in weights],
            'config': {'iterations_per_weight': int(iterations),
                       'wall_time_per_weight_s': float(wall_time_per_weight_s),
                       'repair_time_s': float(repair_time_s),
                       'joint_time_s': float(joint_time_s), 'workers': int(workers),
                       'polish_time_s': float(polish_time_s),
                       'seed': int(seed), 'max_solutions_per_weight': int(max_solutions_per_weight),
                       'batch_size': 1},
            'completed_weight_runs': len(all_runs), 'candidate_count': unique_total,
            'generated_candidates': generated_total,
            'accepted_solution_count': len(all_candidates),
            'joint_attempted_candidates': attempted_total,
            'joint_feasible_candidates': feasible_total,
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
            excluded_sortie_sets=excluded, polish_time_s=polish_time_s,
        )
        all_runs.append({'weight_index': weight_index + 1, 'weights': list(vector),
                         'status': report['status'], 'run_directory': report['run_directory'],
                         'iterations': report['iterations'],
                         'unique_candidates': report['unique_candidates'],
                         'joint_attempts': len(report['attempts']),
                         'accepted_solutions': len(report['solutions'])})
        generated_total += int(report.get('generated', 0))
        unique_total += int(report.get('unique_candidates', 0))
        attempted_total += len(report.get('attempts', []))
        feasible_total += sum(bool(candidate.get('validation_all_pass'))
                              for candidate in report.get('solutions', []))
        for candidate in report['solutions']:
            candidate['search_source'] = 'weighted_alns'
            item = {'candidate': candidate, 'weights': vector,
                    'objectives': candidate['objectives']}
            all_candidates.append({
                'weight_index': weight_index + 1, 'weights': list(vector),
                'sortie_ids': candidate['sortie_ids'], 'status': candidate['status'],
                'validation_all_pass': candidate['validation_all_pass'],
                'objectives': candidate['objectives'],
                'solution_directory': candidate['solution_directory'],
                'search_source': 'weighted_alns',
            })
            if candidate['validation_all_pass'] is True:
                archive = update_archive(archive, item, objective_key='objectives',
                                         objective_names=OBJECTIVE_NAMES)
        save()
    save(status='COMPLETE')
    return json.loads(manifest_path.read_text(encoding='utf-8'))
