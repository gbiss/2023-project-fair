import numpy as np

from agent.constraint import LinearConstraint, indicator
from agent.feature import Course
from agent.item import ScheduleItem


def test_indicator(course: Course, bundle: list[ScheduleItem]):
    ind = indicator(course, bundle)

    np.testing.assert_array_equal(ind.toarray().flatten(), [1, 1, 0])


def test_linear_constraint(
    course: Course, bundle: list[ScheduleItem], all_items: list[ScheduleItem]
):
    constraint = LinearConstraint.from_lists([all_items], [1], course)

    assert not constraint.satisfies(bundle)
