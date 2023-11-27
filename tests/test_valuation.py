from agent.constraint import LinearConstraint
from agent.item import ScheduleItem
from agent.valuation import ConstraintSatifactionValuation


def test_valid_constraint_valuation(
    bundle: list[ScheduleItem], linear_constraint: LinearConstraint
):
    valuation = ConstraintSatifactionValuation([linear_constraint])

    assert valuation.independent(bundle) == 1
