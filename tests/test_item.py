import pytest
from valuation.item import ScheduleItem, DomainError, FeatureError
from valuation.feature import Course


def test_good_schedule():
    domain = [250, 301, 611]
    course = Course(domain)
    schedule = ScheduleItem([course], [250])

    assert schedule.value(course) == 250
    assert schedule.index(course, 250) == domain.index(250)


def test_bad_feature_list_schedule():
    with pytest.raises(FeatureError):
        ScheduleItem([Course([250, 301, 611])], [250, 301])


def test_bad_feature_list_schedule():
    with pytest.raises(DomainError):
        ScheduleItem([Course([250, 301, 611])], [101])
