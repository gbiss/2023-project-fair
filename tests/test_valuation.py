from fair.constraint import LinearConstraint
from fair.item import ScheduleItem
from fair.valuation import ConstraintSatifactionValuation


def test_valid_constraint_valuation(
    bundle_250_301: list[ScheduleItem], linear_constraint: LinearConstraint
):
    valuation = ConstraintSatifactionValuation([linear_constraint])

    assert valuation.independent(bundle_250_301) == 1
    assert valuation.value(bundle_250_301) == 2


def test_asymmetric_constraint_valuation(
    bundle_250_301: list[ScheduleItem],
    bundle_301_611: list[ScheduleItem],
    linear_constraint_250_301: LinearConstraint,
):
    valuation = ConstraintSatifactionValuation([linear_constraint_250_301])

    assert valuation.value(bundle_250_301) == 1
    assert valuation.value(bundle_301_611) == 2
