from fair.constraint import CourseSectionConstraint, CourseTimeConstraint
from fair.feature import Course, Section, Slot
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan


def test_renaissance_man(
    course: Course, section: Section, slot: Slot, schedule: ScheduleItem
):
    topic_list = [
        [ScheduleItem([course], ["250"]), ScheduleItem([course], ["301"])],
        [ScheduleItem([course], ["611"])],
    ]
    quantities = [1, 1]
    global_constraints = [
        CourseTimeConstraint.mutually_exclusive_slots(schedule, slot, [course, slot]),
        CourseSectionConstraint.one_section_per_course(schedule, course, section),
    ]

    # preferred course list does not exceed max quantitity for multiple random configurations
    for i in range(10):
        student = RenaissanceMan(
            topic_list, quantities, [course], global_constraints, i
        )
        for j in range(len(quantities)):
            assert len(student.preferred_courses[j]) <= student.quantities[j]

    # student without global constraints can always be fully satisfied
    student = RenaissanceMan(topic_list, quantities, [course], [], 0)
    items = []
    for i, quant in enumerate(student.quantities):
        assert student.value(student.preferred_courses[i]) == quant
