from typing import List

from fair.constraint import LinearConstraint, PreferenceConstraint
from fair.feature import Course, Section, Slot, Weekday
from fair.item import ScheduleItem
from fair.valuation import ConstraintSatifactionValuation, UniqueItemsValuation, is_mrf


def test_valid_constraint_valuation(
    bundle_250_301: list[ScheduleItem], all_courses_constraint: LinearConstraint
):
    valuation = ConstraintSatifactionValuation([all_courses_constraint])

    assert valuation.independent(bundle_250_301) == 1
    assert valuation.value(bundle_250_301) == 2


def test_asymmetric_constraint_valuation(
    bundle_250_301: list[ScheduleItem],
    bundle_301_611: list[ScheduleItem],
    linear_constraint_250_301: LinearConstraint,
    all_courses_constraint: LinearConstraint,
):
    valuation = ConstraintSatifactionValuation([linear_constraint_250_301])
    assert valuation.value(bundle_250_301) == 1

    valuation = ConstraintSatifactionValuation([all_courses_constraint])
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


def test_memoization(
    schedule_item250: ScheduleItem, all_items: List[ScheduleItem], course: Course
):
    constraint = PreferenceConstraint.from_item_lists(
        all_items, [["250", "301"]], [1], course
    )
    valuation = ConstraintSatifactionValuation([constraint])
    valuation.value([schedule_item250])
    valuation.value([schedule_item250])

    assert valuation._value_ct == 2
    assert valuation._unique_value_ct == 1

    constraint = PreferenceConstraint.from_item_lists(
        all_items, [["250", "301", "611"]], [2], course
    )
    valuation = ConstraintSatifactionValuation([constraint])
    valuation.value(all_items)
    before_value = valuation._unique_value_ct
    before_independent = valuation._unique_independent_ct
    valuation.value(all_items)

    assert valuation._unique_value_ct == before_value
    assert valuation._unique_independent_ct == before_independent


def test_valuation_compilation(
    bundle_250_301: list[ScheduleItem], all_courses_constraint: LinearConstraint
):
    valuation = ConstraintSatifactionValuation([all_courses_constraint])
    compiled = valuation.compile()

    assert valuation.independent(bundle_250_301) == compiled.independent(bundle_250_301)
    assert valuation.value(bundle_250_301) == compiled.value(bundle_250_301)


def test_mrf(course: Course, slot: Slot, weekday: Weekday, section: Section):
    features = [course, slot, weekday, section]
    schedule = [
        ScheduleItem(features, ["250", (1, 2), ("Mon",), 1], 0),
        ScheduleItem(features, ["250", (4, 5), ("Mon",), 2], 1),
        ScheduleItem(features, ["301", (4, 5), ("Mon",), 1], 2),
    ]
    # is not MRF
    A = [[1, 1, 0], [0, 1, 1]]
    b = [1, 1]
    constraint = LinearConstraint(A, b, 3)
    valuation = ConstraintSatifactionValuation([constraint])
    assert not is_mrf(schedule, valuation.value)

    # add example where is it MRF
