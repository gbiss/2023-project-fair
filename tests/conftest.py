import pytest

from agent.constraint import CoursePreferrenceConstraint
from agent.feature import Course, Slot
from agent.item import ScheduleItem
from agent.valuation import ConstraintSatifactionValuation


@pytest.fixture
def course_domain():
    return [250, 301, 611]


@pytest.fixture
def course(course_domain: list[int]):
    return Course(course_domain)


@pytest.fixture
def slot():
    return Slot(["10am", "12pm", "2pm"])


@pytest.fixture
def schedule_item250(course: Course, slot: Slot):
    return ScheduleItem([course, slot], [250, "10am"])


@pytest.fixture
def schedule_item301(course: Course, slot: Slot):
    return ScheduleItem([course, slot], [301, "10am"])


@pytest.fixture
def schedule_item611(course: Course, slot: Slot):
    return ScheduleItem([course, slot], [611, "12pm"])


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
    return CoursePreferrenceConstraint.from_lists([all_items], [2], course)


@pytest.fixture
def linear_constraint_250_301(
    course: Course, bundle_250_301: list[ScheduleItem], all_items: list[ScheduleItem]
):
    return CoursePreferrenceConstraint.from_lists([all_items[:-1]], [1], course)


@pytest.fixture
def all_items(
    schedule_item250: ScheduleItem,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return [schedule_item250, schedule_item301, schedule_item611]


@pytest.fixture
def course_valuation(linear_constraint_250_301: CoursePreferrenceConstraint):
    return ConstraintSatifactionValuation([linear_constraint_250_301])
