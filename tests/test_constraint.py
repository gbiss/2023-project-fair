import numpy as np

from agent.constraint import (
    CoursePreferrenceConstraint,
    CourseTimeConstraint,
    indicator,
)
from agent.feature import Course, Slot
from agent.item import ScheduleItem


def test_indicator(course: Course, bundle_250_301: list[ScheduleItem]):
    ind = indicator([course], bundle_250_301)

    np.testing.assert_array_equal(ind.toarray().flatten(), [1, 1, 0])


def test_linear_constraint(
    course: Course, bundle_250_301: list[ScheduleItem], all_items: list[ScheduleItem]
):
    constraint = CoursePreferrenceConstraint.from_course_lists(
        [["250", "301", "611"]], [1], course
    )

    assert not constraint.satisfies(bundle_250_301)


def test_linear_constraint_250_301(
    linear_constraint_250_301: CoursePreferrenceConstraint,
    bundle_250_301: list[ScheduleItem],
    bundle_301_611: list[ScheduleItem],
):
    assert not linear_constraint_250_301.satisfies(bundle_250_301)
    assert linear_constraint_250_301.satisfies(bundle_301_611)


def test_time_constraint(
    all_items: list[ScheduleItem],
    bundle_250_301: list[ScheduleItem],
    bundle_301_611: list[ScheduleItem],
    course: Course,
    slot: Slot,
):
    constraint = CourseTimeConstraint.mutually_exclusive_slots(all_items, course, slot)

    assert not constraint.satisfies(bundle_250_301)
    assert constraint.satisfies(bundle_301_611)
