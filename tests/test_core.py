import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud_pipeline.core import Policy, realized_cost


def test_threshold_boundaries():
    assert Policy(.2,.8).decide([.199,.2,.799,.8]).tolist() == ["APPROVE","REVIEW","REVIEW","BLOCK"]
    with pytest.raises(ValueError):
        Policy(.8,.2)
    with pytest.raises(ValueError):
        Policy(.2,.8).decide([np.nan])


def test_cost_accounting():
    cfg={"fraud_loss_multiplier":1,"review_cost":3,"legitimate_review_friction":2,"review_success":.8,"fraud_block_cost":1,"legitimate_block_cost":25}
    actual=realized_cost([1,1,0,0],[100,100,100,100],np.array(["APPROVE","REVIEW","REVIEW","BLOCK"]),cfg)
    assert actual == pytest.approx([100,23,5,25])
