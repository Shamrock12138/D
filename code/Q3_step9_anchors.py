

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.anchors import run_anchors


if __name__ == "__main__":
    print(run_anchors().to_string(index=False))
