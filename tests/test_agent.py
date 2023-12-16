from fair.agent import Student, exchange_contribution, marginal_contribution
from fair.constraint import (
    CourseSectionConstraint,
    CourseTimeConstraint,
    PreferrenceConstraint,
)
from fair.feature import Course, Section, Slot
from fair.item import ScheduleItem
from fair.valuation import ConstraintSatifactionValuation, StudentValuation


def test_exchange_contribution(
    course_valuation: ConstraintSatifactionValuation, all_items: list[ScheduleItem]
):
    assert (
        exchange_contribution(
            course_valuation, [all_items[0]], all_items[0], all_items[1]
        )
        == True
    )

    assert (
        exchange_contribution(
            course_valuation, [all_items[0], all_items[2]], all_items[2], all_items[1]
        )
        == False
    )


def test_marginal_contribution(
    course_valuation: ConstraintSatifactionValuation, all_items: list[ScheduleItem]
):
    assert marginal_contribution(course_valuation, [all_items[0]], all_items[1]) == 0
    assert marginal_contribution(course_valuation, [all_items[0]], all_items[2]) == 1


def test_student(
    course: Course,
    slot: Slot,
    section: Section,
    schedule: list[ScheduleItem],
    all_items: list[ScheduleItem],
):
    preferred_constr = PreferrenceConstraint.from_item_lists([all_items], [2], [course])
    course_time_constr = CourseTimeConstraint.mutually_exclusive_slots(
        schedule, course, slot
    )
    course_sect_constr = CourseSectionConstraint.one_section_per_course(
        schedule, course, section
    )
    student = Student(
        StudentValuation([preferred_constr, course_time_constr, course_sect_constr])
    )

    # two non-conflicting courses
    assert student.value([schedule[0], schedule[2]]) == 2
    # two courses at the same time
    assert student.value([schedule[1], schedule[2]]) == 1
    # two sections of the same course
    assert student.value([schedule[0], schedule[1]]) == 1
