





import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.q3.transport.candidate_filter import filter_compact_candidates

if __name__ == "__main__":
    filter_compact_candidates()