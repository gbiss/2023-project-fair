import os
from collections import defaultdict

import pandas as pd

from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap_E, serial_dictatorship, round_robin
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.metrics import (
    EF_1_agents,
    EF_1_count,
    EF_agents,
    EF_count,
    EF_X_agents,
    EF_X_count,
    leximin,
    nash_welfare,
    utilitarian_welfare,
)
from fair.optimization import StudentAllocationProgram
from fair.simulation import RenaissanceMan, SubStudent

NUM_STUDENTS = 50
MAX_COURSES_PER_TOPIC = 15
LOWER_MAX_COURSES_TOTAL = 5
UPPER_MAX_COURSES_TOTAL = 10
EXCEL_SCHEDULE_PATH = os.path.join(
    os.path.dirname(__file__), "../resources/fall2023schedule-2-cat.xlsx"
)
SPARSE = False
FIND_OPTIMAL = True

# load schedule as DataFrame
with open(EXCEL_SCHEDULE_PATH, "rb") as fd:
    df = pd.read_excel(fd)
df = df[8:15]
# construct features from DataFrame
course = Course(df["Catalog"].astype(str).unique().tolist())


time_ranges = df["Mtg Time"].dropna().unique()
slot = Slot.from_time_ranges(time_ranges, "15T")
weekday = Weekday()

section = Section(df["Section"].dropna().unique().tolist())
features = [course, slot, weekday, section]

# construct schedule
schedule = []
topic_map = defaultdict(set)
for idx, (_, row) in enumerate(df.iterrows()):
    crs = str(row["Catalog"])
    topic_map[row["Categories"]].add(crs)
    slt = slots_for_time_range(row["Mtg Time"], slot.times)
    sec = row["Section"]
    # capacity = row["CICScapacity"]
    capacity = 2
    dys = tuple([day.strip() for day in row["zc.days"].split(" ")])
    schedule.append(
        ScheduleItem(features, [crs, slt, dys, sec], index=idx, capacity=capacity)
    )

topics = sorted([sorted(list(courses)) for courses in topic_map.values()])

# global constraints
course_time_constr = CourseTimeConstraint.from_items(schedule, slot, weekday, SPARSE)
course_sect_constr = MutualExclusivityConstraint.from_items(schedule, course, SPARSE)

# randomly generate students
students = []
for i in range(NUM_STUDENTS):
    student = RenaissanceMan(
        topics,
        [min(len(topic), MAX_COURSES_PER_TOPIC) for topic in topics],
        LOWER_MAX_COURSES_TOTAL,
        UPPER_MAX_COURSES_TOTAL,
        course,
        [course_time_constr, course_sect_constr],
        schedule,
        seed=i,
        sparse=SPARSE,
    )
    legacy_student = LegacyStudent(student, student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    students.append(legacy_student)

X = general_yankee_swap_E(students, schedule)
print("YS utilitarian welfare: ", utilitarian_welfare(X[0], students, schedule))
print("YS nash welfare: ", nash_welfare(X[0], students, schedule))
print("YS leximin vector: ", leximin(X[0], students, schedule))

X_SD = serial_dictatorship(students, schedule)
print("SD utilitarian welfare: ", utilitarian_welfare(X_SD, students, schedule))
print("SD nash welfare: ", nash_welfare(X_SD, students, schedule))
print("SD leximin vector: ", leximin(X_SD, students, schedule))

X_RR = round_robin(students, schedule)
print("RR utilitarian welfare: ", utilitarian_welfare(X_RR, students, schedule))
print("RR nash welfare: ", nash_welfare(X_RR, students, schedule))
print("RR leximin vector: ", leximin(X_RR, students, schedule))


if FIND_OPTIMAL:
    orig_students = [student.student for student in students]
    program = StudentAllocationProgram(orig_students, schedule).compile()
    opt_alloc = program.formulateUSW().solve()
    opt_USW = sum(opt_alloc) / len(orig_students)
    print("optimal utilitarian welfare", opt_USW)

    opt_alloc = opt_alloc.reshape(len(students), len(schedule)).transpose()
    print(
        "ILP utilitarian welfare: ", utilitarian_welfare(opt_alloc, students, schedule)
    )
    print("ILP nash welfare: ", nash_welfare(opt_alloc, students, schedule))
    print("ILP leximin vector: ", leximin(opt_alloc, students, schedule))

from fair.allocation import get_bundle_from_allocation_matrix


def create_sub_schedule(bundle_1, bundle_2):

    sub_schedule = [*bundle_1, *bundle_2]
    set_sub_schedule = list(set(sub_schedule))

    course_strings = sorted([item.values[0] for item in set_sub_schedule])
    course = Course(course_strings)
    section = Section(sorted(list(set([item.values[3] for item in set_sub_schedule]))))
    features = [course, slot, weekday, section]

    new_schedule = []
    for i, item in enumerate(set_sub_schedule):
        new_schedule.append(
            ScheduleItem(
                features, item.values, index=i, capacity=sub_schedule.count(item)
            )
        )
    return new_schedule, course_strings, course


def create_sub_students(og_students, new_schedule, course_strings, course):
    course_time_constr = CourseTimeConstraint.from_items(
        new_schedule, slot, weekday, SPARSE
    )
    course_sect_constr = MutualExclusivityConstraint.from_items(
        new_schedule, course, SPARSE
    )
    new_students = []
    for student in og_students:
        preferred = student1.preferred_courses
        new_student = SubStudent(
            student.student.quantities,
            [
                [item for item in pref if item in course_strings]
                for pref in student.student.preferred_topics
            ],
            list(set(course_strings) & set(preferred)),
            student.student.total_courses,
            course,
            [course_time_constr, course_sect_constr],
            new_schedule,
            sparse=SPARSE,
        )
        legacy_student = LegacyStudent(
            new_student, new_student.preferred_courses, course
        )
        legacy_student.student.valuation.valuation = (
            legacy_student.student.valuation.compile()
        )
        new_students.append(legacy_student)
    return new_students


student1_idx = 2
student1 = students[student1_idx]

student2_idx = 49
student2 = students[student2_idx]

current_bundle_1 = get_bundle_from_allocation_matrix(X[0], schedule, student1_idx)
current_bundle_2 = get_bundle_from_allocation_matrix(X[0], schedule, student2_idx)

# create new schedule


new_schedule, course_strings, course = create_sub_schedule(
    current_bundle_1, current_bundle_2
)


new_students = create_sub_students(
    [student1, student2], new_schedule, course_strings, course
)

X_sub, _, _ = general_yankee_swap_E(new_students, new_schedule)
print(X_sub)
