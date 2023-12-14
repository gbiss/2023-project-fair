import os
from typing import List

import pytest

from fair.constraint import (
    CoursePreferrenceConstraint,
    CourseSectionConstraint,
    CourseTimeConstraint,
    LinearConstraint,
)
from fair.feature import Course, Section, Slot
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan
from fair.valuation import ConstraintSatifactionValuation


@pytest.fixture
def course_domain():
    return ["250", "301", "611"]


@pytest.fixture
def course(course_domain: list[int]):
    return Course(course_domain)


@pytest.fixture
def slot():
    return Slot([1, 2, 3, 4, 5, 6, 7], [(1, 2), (2, 3), (4, 5), (6, 7)])


@pytest.fixture
def section():
    return Section([1, 2, 3])


@pytest.fixture
def schedule_item250(course: Course, slot: Slot, section: Section):
    return ScheduleItem([course, slot, section], ["250", (1, 2), 1])


@pytest.fixture
def schedule_item250_2(course: Course, slot: Slot, section: Section):
    return ScheduleItem([course, slot, section], ["250", (4, 5), 2])


@pytest.fixture
def schedule_item301(course: Course, slot: Slot, section: Section):
    return ScheduleItem([course, slot, section], ["301", (2, 3), 1])


@pytest.fixture
def schedule_item611(course: Course, slot: Slot, section: Section):
    return ScheduleItem([course, slot, section], ["611", (4, 5), 1])


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
        ScheduleItem(features, ["250", (1, 2), 1]),
        ScheduleItem(features, ["250", (4, 5), 2]),
        ScheduleItem(features, ["301", (4, 5), 1]),
        ScheduleItem(features, ["301", (6, 7), 2]),
        ScheduleItem(features, ["611", (6, 7), 1]),
    ]
    return items


@pytest.fixture
def excel_schedule_path():
    return os.path.join(
        os.path.dirname(__file__), "../resources/fall2023schedule-2.xlsx"
    )


@pytest.fixture
def excel_schedule_path_with_cats():
    return os.path.join(
        os.path.dirname(__file__), "../resources/fall2023schedule-2-cat.xlsx"
    )


@pytest.fixture
def global_constraints(
    schedule: list[ScheduleItem], course: Course, section: Section, slot: Slot
):
    course_time_constr = CourseTimeConstraint.mutually_exclusive_slots(
        schedule, course, slot
    )
    course_sect_constr = CourseSectionConstraint.one_section_per_course(
        schedule, course, section
    )

    return [course_time_constr, course_sect_constr]


@pytest.fixture
def renaissance1(global_constraints: List[LinearConstraint], course: Course):
    return RenaissanceMan(
        [["250", "301"], ["611"]],
        [1, 1],
        course,
        global_constraints,
        0,
    )


@pytest.fixture
def renaissance2(global_constraints: List[LinearConstraint], course: Course):
    return RenaissanceMan(
        [["250", "301"], ["611"]],
        [1, 1],
        course,
        global_constraints,
        1,
    )
