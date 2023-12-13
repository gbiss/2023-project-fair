from fair.agent import LegacyStudent
from fair.allocation import original_yankee_swap
from fair.constraint import CourseSectionConstraint, CourseTimeConstraint
from fair.feature import Course, Section, Slot
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan


def test_legacy_simulated_student(
    course: Course, slot: Slot, section: Section, schedule: list[ScheduleItem]
):
    course_time_constr = CourseTimeConstraint.mutually_exclusive_slots(
        schedule, course, slot
    )
    course_sect_constr = CourseSectionConstraint.one_section_per_course(
        schedule, course, section
    )
    student1 = RenaissanceMan(
        [["250", "301"], ["611"]],
        [1, 1],
        course,
        [course_time_constr, course_sect_constr],
        0,
    )
    student2 = RenaissanceMan(
        [["250", "301"], ["611"]],
        [1, 1],
        course,
        [course_time_constr, course_sect_constr],
        1,
    )
    leg_student1 = LegacyStudent(student1)
    leg_student2 = LegacyStudent(student2)

    original_yankee_swap([leg_student1, leg_student2], schedule)
