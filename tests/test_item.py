import pytest

from agent.feature import Course
from agent.item import DomainError, FeatureError, ScheduleItem


def test_good_schedule(
    domain: list[int], course: Course, schedule_item250: ScheduleItem
):
    assert schedule_item250.value(course) == 250
    assert schedule_item250.index(course, 250) == domain.index(250)


def test_bad_feature_list_schedule(course: Course):
    # more values than features
    with pytest.raises(FeatureError):
        ScheduleItem([course], [250, 301])


def test_bad_value_schedule(course: Course):
    # value outside feature domain
    with pytest.raises(DomainError):
        ScheduleItem([course], [101])
