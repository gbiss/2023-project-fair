import pytest

from agent.constraint import CoursePreferrenceConstraint
from agent.feature import Course, Section, Slot
from agent.item import ScheduleItem
from agent.valuation import ConstraintSatifactionValuation


@pytest.fixture
def course_domain():
    return ["250", "301", "611"]


@pytest.fixture
def course(course_domain: list[int]):
    return Course(course_domain)


@pytest.fixture
def slot():
    return Slot(["10am", "12pm", "2pm"])


@pytest.fixture
def section():
    return Section([1, 2, 3])


@pytest.fixture
def schedule_item250(course: Course, slot: Slot, section: Section):
    return ScheduleItem([course, slot, section], ["250", "10am", 1])


@pytest.fixture
def schedule_item250_2(course: Course, slot: Slot, section: Section):
    return ScheduleItem([course, slot, section], ["250", "12pm", 2])


@pytest.fixture
def schedule_item301(course: Course, slot: Slot, section: Section):
    return ScheduleItem([course, slot, section], ["301", "10am", 1])


@pytest.fixture
def schedule_item611(course: Course, slot: Slot, section: Section):
    return ScheduleItem([course, slot, section], ["611", "12pm", 1])


@pytest.fixture
def bundle_250_301(schedule_item250: ScheduleItem, schedule_item301: ScheduleItem):
    return [schedule_item250, schedule_item301]


@pytest.fixture
def bundle_250_250_2(schedule_item250: ScheduleItem, schedule_item250_2: ScheduleItem):
    return [schedule_item250, schedule_item250_2]


@pytest.fixture
def bundle_301_611(schedule_item301: ScheduleItem, schedule_item611: ScheduleItem):
    return [schedule_item301, schedule_item611]


@pytest.fixture
def linear_constraint(
    course: Course, bundle_250_301: list[ScheduleItem], all_items: list[ScheduleItem]
):
    return CoursePreferrenceConstraint.from_course_lists(
        [["250", "301", "611"]], [2], course
    )


@pytest.fixture
def linear_constraint_250_301(
    course: Course, bundle_250_301: list[ScheduleItem], all_items: list[ScheduleItem]
):
    return CoursePreferrenceConstraint.from_course_lists([["250", "301"]], [1], course)


@pytest.fixture
def all_items(
    schedule_item250: ScheduleItem,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return [schedule_item250, schedule_item301, schedule_item611]


@pytest.fixture
def items_repeat_section(
    schedule_item250: ScheduleItem,
    schedule_item250_2: ScheduleItem,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return [schedule_item250, schedule_item250_2, schedule_item301, schedule_item611]


@pytest.fixture
def course_valuation(linear_constraint_250_301: CoursePreferrenceConstraint):
    return ConstraintSatifactionValuation([linear_constraint_250_301])


@pytest.fixture
def schedule(course: Course, slot: Slot, section: Section):
    features = [course, slot, section]
    items = [
        ScheduleItem(features, ["250", "10am", 1]),
        ScheduleItem(features, ["250", "12pm", 2]),
        ScheduleItem(features, ["301", "12pm", 1]),
        ScheduleItem(features, ["301", "2pm", 2]),
        ScheduleItem(features, ["611", "2pm", 1]),
    ]
    return items
