from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan


def test_renaissance_man(
    course: Course, section: Section, slot: Slot, schedule: ScheduleItem
):
    topic_list = [["250", "301"], ["611"]]
    quantities = [1, 1]
    features = [course, section, slot]
    global_constraints = [
        CourseTimeConstraint.from_items(schedule, slot, [course, slot]),
        MutualExclusivityConstraint.from_items(schedule, course, [course, section]),
    ]

    # preferred course list does not exceed max quantitity for multiple random configurations
    for i in range(10):
        student = RenaissanceMan(
            topic_list, quantities, course, global_constraints, schedule, features, i
        )
        for j in range(len(quantities)):
            assert len(student.preferred_topics[j]) <= student.quantities[j]

    # student without global constraints can always be fully satisfied
    student = RenaissanceMan(topic_list, quantities, course, [], schedule, features, 0)
    for i, quant in enumerate(student.quantities):
        items = [
            item
            for item in schedule
            if item.value(course) in student.preferred_topics[i]
        ]
        assert student.value(items) == quant
