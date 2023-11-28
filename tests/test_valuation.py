from agent.constraint import LinearConstraint
from agent.item import ScheduleItem
from agent.valuation import ConstraintSatifactionValuation


def test_valid_constraint_valuation(
    bundle_250_301: list[ScheduleItem], linear_constraint: LinearConstraint
):
    valuation = ConstraintSatifactionValuation([linear_constraint])

    assert valuation.independent(bundle_250_301) == 1
    assert valuation.value(bundle_250_301) == 2
