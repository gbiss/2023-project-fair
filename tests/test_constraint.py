import numpy as np
from valuation.constraint import indicator
from valuation.feature import Course
from valuation.item import ScheduleItem


def test_indicator():
    domain = [250, 301, 611]
    course = Course(domain)
    bundle = [
        ScheduleItem([course], [250]),
        ScheduleItem([course], [301]),
    ]
    ind = indicator(course, bundle)

    np.testing.assert_array_equal(ind.toarray().flatten(), [1, 1, 0])
