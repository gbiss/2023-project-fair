import pytest

from agent.feature import Course
from agent.item import DomainError, FeatureError, ScheduleItem


def test_item_hash(schedule_item250: ScheduleItem):
    hash(schedule_item250)


def test_item_lt(schedule_item250: ScheduleItem, schedule_item301: ScheduleItem):
    h250 = hash(schedule_item250)
    h301 = hash(schedule_item301)
    assert (schedule_item250 < schedule_item301) == (h250 < h301)


def test_item_eq(course: Course):
    sch1 = ScheduleItem([course], ["250"])
    sch2 = ScheduleItem([course], ["250"])
    assert sch1 == sch2


def test_good_schedule(
    course_domain: list[int], course: Course, schedule_item250: ScheduleItem
):
    assert schedule_item250.value(course) == "250"
    assert schedule_item250.index([course]) == course_domain.index("250")


def test_bad_feature_list_schedule(course: Course):
    # more values than features
    with pytest.raises(FeatureError):
        ScheduleItem([course], ["250", "301"])


def test_bad_value_schedule(course: Course):
    # value outside feature domain
    with pytest.raises(DomainError):
        ScheduleItem([course], ["101"])
