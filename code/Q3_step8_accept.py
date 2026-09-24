"""Validate and freeze the completed Step8 joint schedule."""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.step8_acceptance import accept_step8


if __name__ == "__main__":
    print(json.dumps(accept_step8(), ensure_ascii=False, indent=2))
