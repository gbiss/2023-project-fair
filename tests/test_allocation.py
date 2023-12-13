from fair.agent import LegacyStudent
from fair.allocation import original_yankee_swap
from fair.constraint import CourseSectionConstraint, CourseTimeConstraint
from fair.feature import Course, Section, Slot
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan


def test_original_yankee_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1)
    leg_student2 = LegacyStudent(renaissance2)

    original_yankee_swap([leg_student1, leg_student2], schedule)
