import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.communication.link_budget import free_space_loss_db


def test_colocated_gateway_uses_finite_one_meter_reference():
    assert free_space_loss_db(0.0, 2400.0) == free_space_loss_db(1.0, 2400.0)
