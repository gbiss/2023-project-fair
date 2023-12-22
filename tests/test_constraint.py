from typing import List

import numpy as np

from fair.constraint import (
    CourseTimeConstraint,
    MutualExclusivityConstraint,
    PreferenceConstraint,
    indicator,
)
from fair.feature import BaseFeature, Course, Section, Slot
from fair.item import ScheduleItem


def test_indicator(bundle_250_301: list[ScheduleItem]):
    ind = indicator(bundle_250_301, 3)

    np.testing.assert_array_equal(ind.toarray().flatten(), [1, 0, 1])


def test_linear_constraint(
    course: Course,
    bundle_250_301_2: list[ScheduleItem],
    schedule: list[ScheduleItem],
):
    constraint = PreferenceConstraint.from_item_lists(
        schedule, [["250", "301", "611"]], [1], course
    )

    assert not constraint.satisfies(bundle_250_301_2)


def test_linear_constraint_250_301(
    linear_constraint_250_301: PreferenceConstraint,
    schedule_item250: ScheduleItem,
    bundle_250_301: list[ScheduleItem],
):
    assert not linear_constraint_250_301.satisfies(bundle_250_301)
    assert linear_constraint_250_301.satisfies([schedule_item250])


def test_time_constraint(
    all_items: list[ScheduleItem],
    bundle_250_301: list[ScheduleItem],
    bundle_301_611: list[ScheduleItem],
    slot: Slot,
):
    constraint = CourseTimeConstraint.from_items(all_items, slot)

    assert not constraint.satisfies(bundle_250_301)
    assert constraint.satisfies(bundle_301_611)


def test_section_constraint(
    items_repeat_section: list[ScheduleItem],
    bundle_250_301: list[ScheduleItem],
    bundle_250_250_2: list[ScheduleItem],
    course: Course,
    section: Section,
):
    constraint = MutualExclusivityConstraint.from_items(items_repeat_section, course)

    assert constraint.satisfies(bundle_250_301)
    assert not constraint.satisfies(bundle_250_250_2)


def test_constrained_items(
    all_items: list[ScheduleItem],
    schedule_item250: ScheduleItem,
    schedule_item301: ScheduleItem,
    course_time_constraint: CourseTimeConstraint,
):
    assert schedule_item250 in course_time_constraint.constrained_items(all_items)

    ct_active = course_time_constraint.constrained_items(all_items)
    assert (
        len(set(ct_active[schedule_item250]).intersection(ct_active[schedule_item301]))
        > 0
    )
