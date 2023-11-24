import pytest
from valuation.item import Schedule, DomainError, FeatureError
from valuation.feature import Course


def test_good_schedule():
    course = Course([250, 301, 611])
    schedule = Schedule([course], [250])

    assert schedule.value(course) == 250


def test_bad_feature_list_schedule():
    with pytest.raises(FeatureError):
        Schedule([Course([250, 301, 611])], [250, 301])


def test_bad_feature_list_schedule():
    with pytest.raises(DomainError):
        Schedule([Course([250, 301, 611])], [101])
