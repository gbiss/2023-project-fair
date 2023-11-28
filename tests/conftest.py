import pytest

from agent.constraint import LinearConstraint
from agent.feature import Course
from agent.item import ScheduleItem
from agent.valuation import ConstraintSatifactionValuation


@pytest.fixture
def domain():
    return [250, 301, 611]


@pytest.fixture
def course(domain: list[int]):
    return Course(domain)


@pytest.fixture
def schedule_item250(course: Course):
    return ScheduleItem([course], [250])


@pytest.fixture
def schedule_item301(course: Course):
    return ScheduleItem([course], [301])


@pytest.fixture
def schedule_item611(course: Course):
    return ScheduleItem([course], [611])


@pytest.fixture
def bundle_250_301(schedule_item250: ScheduleItem, schedule_item301: ScheduleItem):
    return [schedule_item250, schedule_item301]


@pytest.fixture
def bundle_301_611(schedule_item301: ScheduleItem, schedule_item611: ScheduleItem):
    return [schedule_item301, schedule_item611]


@pytest.fixture
def linear_constraint(
    course: Course, bundle_250_301: list[ScheduleItem], all_items: list[ScheduleItem]
):
    return LinearConstraint.from_lists([all_items], [2], course)


@pytest.fixture
def linear_constraint_250_301(
    course: Course, bundle_250_301: list[ScheduleItem], all_items: list[ScheduleItem]
):
    return LinearConstraint.from_lists([all_items[:-1]], [1], course)


@pytest.fixture
def all_items(
    schedule_item250: ScheduleItem,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return [schedule_item250, schedule_item301, schedule_item611]


@pytest.fixture
def course_valuation(
    linear_constraint_250_301: LinearConstraint, all_items: list[ScheduleItem]
):
    return ConstraintSatifactionValuation([linear_constraint_250_301])
