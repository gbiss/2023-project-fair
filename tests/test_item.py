import pytest

from fair.feature import Course
from fair.item import DomainError, FeatureError, ScheduleItem


def test_item_hash(schedule_item250: ScheduleItem):
    hash(schedule_item250)


def test_item_lt(schedule_item250: ScheduleItem, schedule_item301: ScheduleItem):
    h250 = hash(schedule_item250)
    h301 = hash(schedule_item301)
    assert (schedule_item250 < schedule_item301) == (h250 < h301)


def test_item_eq(course: Course):
    sch1 = ScheduleItem([course], ["250"], 1)
    sch2 = ScheduleItem([course], ["250"], 1)
    assert sch1 == sch2


def test_good_schedule(
    course_domain: list[int], course: Course, schedule_item250: ScheduleItem
):
    assert schedule_item250.value(course) == "250"


def test_bad_feature_list_schedule(course: Course):
    # more values than features
    with pytest.raises(FeatureError):
        ScheduleItem([course], ["250", "301"], 1)


def test_bad_value_schedule(course: Course):
    # value outside feature domain
    with pytest.raises(DomainError):
        ScheduleItem([course], ["101"], 1)


def test_schedule_excel_import(excel_schedule_path_with_cats: str):
    schedule_items = ScheduleItem.parse_excel(excel_schedule_path_with_cats)

    assert None not in [sched.category for sched in schedule_items]
