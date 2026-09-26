










import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.q3.transport.comm_gap import extract_pattern_gap_templates

if __name__ == "__main__":
    extract_pattern_gap_templates()