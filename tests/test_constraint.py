from agent.constraint import indicator, LinearConstraint
from agent.feature import Course
from agent.item import ScheduleItem
import numpy as np
from scipy.sparse import dok_array


def test_indicator(course: Course, bundle: list[ScheduleItem]):
    ind = indicator(course, bundle)

    np.testing.assert_array_equal(ind.toarray().flatten(), [1, 1, 0])


def test_linear_constraint(
    domain: list[int], course: Course, bundle: list[ScheduleItem]
):
    A = dok_array((1, len(domain)), dtype=np.int_)
    A = A + True
    b = dok_array((1, 1), dtype=np.bool_)
    b = b + True

    assert not LinearConstraint(A, b, course).violates(bundle)
