from fair.constraint import CourseSectionConstraint, CourseTimeConstraint
from fair.feature import Course, Section, Slot
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan


def test_renaissance_man(
    course: Course, section: Section, slot: Slot, schedule: ScheduleItem
):
    topic_list = [["250", "301"], ["611"]]
    quantities = [1, 1]
    global_constraints = [
        CourseTimeConstraint.mutually_exclusive_slots(schedule, course, slot),
        CourseSectionConstraint.one_section_per_course(schedule, course, section),
    ]
    RenaissanceMan(topic_list, quantities, course, global_constraints, 1)
