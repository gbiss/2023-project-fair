import math

from fair.constraint import LinearConstraint
from fair.feature import Course, Section, Slot, Weekday
from fair.item import ScheduleItem
from fair.set_tools import (
    is_monotonic_non_decreasing,
    is_mrf,
    is_submodular,
    nonnegative_rank_value,
    powerset,
    rank_value_leq_cardinality,
)
from fair.valuation import ConstraintSatifactionValuation


def length_marginal_increasing(vals: list[int]):
    """FAIL submodular, PASS monotone non-decreasing"""
    return len(vals) ** 2


def length_marginal_decreasing(vals: list[int]):
    """PASS submodular, PASS monotone non-decreasing"""
    return math.sqrt(len(vals))


def additive(vals: list[int]):
    """PASS submodular, PASS monotone non-decreasing"""
    return sum(vals)


def budget_additive(vals: list[int]):
    """PASS submodular, PASS monotone non-decreasing"""
    return min(sum(vals), 3)


def additive_divide_by_rank(vals: list[int]):
    """FAIL submodular, FAIL monotone non-decreasing"""
    return sum(vals) / (len(vals) + 1)


def test_powerset():
    test_set1 = [0, 1, 2]
    power_set1 = powerset(test_set1)
    power_set1 = list(power_set1)
    assert power_set1 == [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    test_set2 = [0, 1, 2, 3]
    power_set2 = powerset(test_set2)
    power_set2 = list(power_set2)
    assert power_set2 == [
        (),
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
        (0, 1, 2, 3),
    ]


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

    # is MRF
    values = [0, 1, 2, 3, 4, 5]
    assert is_mrf(values, length_marginal_decreasing)

    # is MRF
    features = [course, slot, weekday, section]
    schedule = [
        ScheduleItem(features, ["250", (1, 2), ("Mon",), 1], 0),
        ScheduleItem(features, ["250", (1, 2), ("Mon",), 2], 1),
        ScheduleItem(features, ["301", (4, 5), ("Mon",), 1], 2),
    ]
    A = [[1, 0, 0], [0, 1, 1]]
    b = [1, 2]
    constraint = LinearConstraint(A, b, 3)
    valuation = ConstraintSatifactionValuation([constraint])
    assert is_mrf(schedule, valuation.value)


def test_submodular():
    values = [0, 1, 2, 3, 4, 5]

    assert not is_submodular(values, length_marginal_increasing)
    assert is_submodular(values, length_marginal_decreasing)
    assert is_submodular(values, additive)
    assert is_submodular(values, budget_additive)
    assert not is_submodular(values, additive_divide_by_rank)


def test_monotonic_non_decreasing():
    values = [0, 1, 2, 3, 4, 5]

    assert is_monotonic_non_decreasing(values, length_marginal_increasing)
    assert is_monotonic_non_decreasing(values, length_marginal_decreasing)
    assert is_monotonic_non_decreasing(values, additive)
    assert is_monotonic_non_decreasing(values, budget_additive)
    assert not is_monotonic_non_decreasing(values, additive_divide_by_rank)


def test_nonnegative_rank_value():
    values = [0, 1, 2, 3, 4, 5]

    assert nonnegative_rank_value(values, length_marginal_increasing)
    assert nonnegative_rank_value(values, length_marginal_decreasing)
    assert nonnegative_rank_value(values, additive)
    assert nonnegative_rank_value(values, budget_additive)
    assert nonnegative_rank_value(values, additive_divide_by_rank)


def test_rank_value_leq_cardinality():
    values = [0, 1, 2, 3, 4, 5]

    assert not rank_value_leq_cardinality(values, length_marginal_increasing)
    assert rank_value_leq_cardinality(values, length_marginal_decreasing)
    assert not rank_value_leq_cardinality(values, additive)
    assert not rank_value_leq_cardinality(values, budget_additive)
    assert not rank_value_leq_cardinality(values, additive_divide_by_rank)
