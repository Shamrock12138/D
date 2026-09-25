"""Freeze a finally selected, accepted Q3 schedule as Q4 input.

Example after final Q3 selection and validation:
python code/Q3_step13_freeze_for_q4.py --source code/data/q3_selected_final \
  --acceptance code/data/q3_selected_final/acceptance.json \
  --selection-id selected-pareto-solution
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.q4_export import freeze_q3_for_q4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--selection-id", required=True)
    args = parser.parse_args()
    accepted = json.loads(args.acceptance.read_text(encoding="utf-8"))
    result = freeze_q3_for_q4(args.source, accepted, args.selection_id)
    print(json.dumps({"status": result["status"],
                      "final_selection_id": result["final_selection_id"],
                      "output": "code/data/q3_final_frozen/"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
