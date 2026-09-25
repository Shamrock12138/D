"""Test compact CP-SAT model building."""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.cp_sat_scheduler import prepare_q3_problem, _build_q3_model

warnings.filterwarnings("ignore")

print("=== Phase 1: prepare_q3_problem ===")
problem = prepare_q3_problem(tier="tier2")

print(f"occurrences: {len(problem['occurrences'])}")
print(f"classes: {len(problem['class_supply'])}")
print(f"gap_option_map entries: {len(problem['gap_option_map'])}")
print(f"relay options: {problem['relay'].shape[0] if problem['relay'] is not None else 0}")
print(f"uav types: {sorted(problem['uav_ids'].keys())}")
print(f"total class supply: {sum(problem['class_supply'].values())}")

# Check: occurrence_gaps coverage
total_gaps = sum(len(v) for v in problem["occurrence_gaps"].values())
print(f"total occurrence gaps: {total_gaps}")

# Check: gap_option_map coverage
gap_ids_with_rolein_relay = 0
for occ in problem["occurrences"]:
    for gid in occ.gap_ids:
        if gid in problem["gap_option_map"]:
            gap_ids_with_rolein_relay += 1
print(f"occurrence gaps WITH relay options: {gap_ids_with_rolein_relay}")
print(f"occurrence gaps WITHOUT relay options: {total_gaps - gap_ids_with_rolein_relay}")

print("\n=== Phase 2: _build_q3_model ===")
try:
    built = _build_q3_model(problem)
    model, select, starts, relay_select, relay_starts, tc, rc, jc, meta = built
    print(f"Transport variables: {len(select)}")
    print(f"Relay variables: {len(relay_select)}")
    print(f"Transport meta entries: {len(meta['transport'])}")
    print(f"Relay meta entries: {len(meta['relay'])}")
    print("\nSUCCESS: Model built!")

    print("\n=== Phase 3: Quick feasibility solve (30s limit) ===")
    from ortools.sat.python import cp_model as cpm
    solver = cpm.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    print(f"Solver status: {solver.StatusName(status)}")
    if status in (cpm.OPTIMAL, cpm.FEASIBLE):
        obj = solver.ObjectiveValue()
        print(f"Objective value: {obj}")
        # Count selected occurrences
        n_selected = sum(1 for v in select.values() if solver.Value(v))
        print(f"Selected transport occurrences: {n_selected}")
        n_relay_selected = sum(1 for v in relay_select.values() if solver.Value(v))
        print(f"Selected relay options: {n_relay_selected}")
        print("\nFEASIBLE: Model has valid solutions!")
    elif status == cpm.INFEASIBLE:
        print("\nINFEASIBLE: Model has no solution.")
    else:
        print(f"\nNo solution found within 30s (status={solver.StatusName(status)}).")

except Exception as e:
    print(f"\nFAIL: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()