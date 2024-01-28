import math

from fair.constraint import LinearConstraint
from fair.feature import Course, Section, Slot, Weekday
from fair.item import ScheduleItem
from fair.set_tools import (
    is_monotonic_non_decreasing,
    is_mrf,
    is_submodular,
    nonnegative_rank_value,
    rank_value_leq_cardinality,
)
from fair.valuation import ConstraintSatifactionValuation


def length_marginal_increasing(vals):
    """FAIL submodular, PASS monotone non-decreasing"""
    return len(vals) ** 2


def length_marginal_decreasing(vals):
    """PASS submodular, PASS monotone non-decreasing"""
    return math.sqrt(len(vals))


def additive(vals):
    """PASS submodular, PASS monotone non-decreasing"""
    return sum(vals)


def budget_additive(vals):
    """PASS submodular, PASS monotone non-decreasing"""
    return min(sum(vals), 3)


def additive_divide_by_rank(vals):
    """FAIL submodular, FAIL monotone non-decreasing"""
    return sum(vals) / (len(vals) + 1)


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


def test_submodular():
    values = [0, 1, 2, 3, 4, 5]

    assert not is_submodular(values, length_marginal_increasing)
    assert is_submodular(values, length_marginal_decreasing)
    assert is_submodular(values, additive)
    assert is_submodular(values, budget_additive)
    assert not is_submodular(values, additive_divide_by_rank)


def test_monotonic_non_decreasing():
    vals_1 = [0, 1, 2, 3, 4, 5]

    assert is_monotonic_non_decreasing(vals_1, length_marginal_increasing)
    assert is_monotonic_non_decreasing(vals_1, length_marginal_decreasing)
    assert is_monotonic_non_decreasing(vals_1, additive)
    assert is_monotonic_non_decreasing(vals_1, budget_additive)
    assert not is_monotonic_non_decreasing(vals_1, additive_divide_by_rank)


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
