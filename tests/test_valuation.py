from fair.constraint import LinearConstraint
from fair.item import ScheduleItem
from fair.valuation import ConstraintSatifactionValuation, UniqueItemsValuation


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


def test_unique_item_adapter(
    schedule_item250: ScheduleItem, linear_constraint_250_301: LinearConstraint
):
    original_valuation = ConstraintSatifactionValuation([linear_constraint_250_301])
    unique_valuation = UniqueItemsValuation(original_valuation)
    bundle = [schedule_item250, schedule_item250]

    assert original_valuation.independent(bundle) == True
    assert original_valuation.value(bundle) == 2
    assert unique_valuation.independent(bundle) == True
    assert unique_valuation.value(bundle) == 1
