from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, Weekday
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan


def test_renaissance_man(
    course: Course,
    section: Section,
    slot: Slot,
    weekday: Weekday,
    schedule: ScheduleItem,
):
    topic_list = [["250", "301"], ["611"]]
    quantities = [1, 1]
    min_courses = 1
    max_courses = 2
    global_constraints = [
        CourseTimeConstraint.from_items(schedule, slot, weekday),
        MutualExclusivityConstraint.from_items(schedule, course),
    ]

    # preferred course list does not exceed max quantitity for multiple random configurations
    for i in range(10):
        student = RenaissanceMan(
            topic_list,
            quantities,
            min_courses,
            max_courses,
            course,
            global_constraints,
            schedule,
            i,
        )
        for j in range(len(quantities)):
            assert len(student.preferred_topics[j]) <= student.quantities[j]

    # student without global constraints can always be fully satisfied
    student = RenaissanceMan(
        topic_list, quantities, min_courses, max_courses, course, [], schedule, 0
    )
    for i, quant in enumerate(student.quantities):
        items = [
            item
            for item in schedule
            if item.value(course) in student.preferred_topics[i]
        ]
        assert student.value(items) == quant


def test_renaissance_man_memoing(
    course: Course,
    slot: Slot,
    weekday: Weekday,
    schedule: ScheduleItem,
):
    topic_list = [["250", "301"], ["611"]]
    quantities = [1, 1]
    min_courses = 1
    max_courses = 2
    global_constraints = [
        CourseTimeConstraint.from_items(schedule, slot, weekday),
        MutualExclusivityConstraint.from_items(schedule, course),
    ]

    student_no_memo = RenaissanceMan(
        topic_list,
        quantities,
        min_courses,
        max_courses,
        course,
        global_constraints,
        schedule,
        memoize=False,
    )
    student_no_memo.value(schedule)

    assert student_no_memo.valuation._value_memo == {}

    student_with_memo = RenaissanceMan(
        topic_list,
        quantities,
        min_courses,
        max_courses,
        course,
        global_constraints,
        schedule,
    )
    student_with_memo.value(schedule)

    assert len(student_with_memo.valuation._value_memo) > 0

    student_with_memo.valuation.reset()

    assert len(student_with_memo.valuation._value_memo) == 0
