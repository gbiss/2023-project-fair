import os
from collections import defaultdict

import pandas as pd
import numpy as np

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

NUM_STUDENTS = 4
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
    capacity = 1
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

students[3], students[2] = students[2], students[3]

X_YS, _, _ = general_yankee_swap_E(students, schedule)
print("YS utilitarian welfare: ", utilitarian_welfare(X_YS, students, schedule))
print("YS nash welfare: ", nash_welfare(X_YS, students, schedule))
print("YS leximin vector: ", leximin(X_YS, students, schedule))

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

    X_ILP = opt_alloc.reshape(len(students), len(schedule)).transpose()
    print("ILP utilitarian welfare: ", utilitarian_welfare(X_ILP, students, schedule))
    print("ILP nash welfare: ", nash_welfare(X_ILP, students, schedule))
    print("ILP leximin vector: ", leximin(X_ILP, students, schedule))

from fair.allocation import get_bundle_from_allocation_matrix


def create_sub_schedule(bundle_1, bundle_2):
    sub_schedule = [*bundle_1, *bundle_2]
    set_sub_schedule = sorted(list(set(sub_schedule)), key=lambda item: item.values[0])

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


def create_sub_student(student, new_schedule, course_strings, course):
    course_time_constr = CourseTimeConstraint.from_items(
        new_schedule, slot, weekday, SPARSE
    )
    course_sect_constr = MutualExclusivityConstraint.from_items(
        new_schedule, course, SPARSE
    )
    preferred = student.preferred_courses
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
    legacy_student = LegacyStudent(new_student, new_student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    return legacy_student


def yankee_swap_sub_problem(student, new_schedule, course_strings, course):

    sub_student = create_sub_student(student, new_schedule, course_strings, course)
    X_sub, _, _ = general_yankee_swap_E([sub_student, sub_student], new_schedule)
    print(X_sub)

    bundle_1 = get_bundle_from_allocation_matrix(X_sub, new_schedule, 0)
    bundle_2 = get_bundle_from_allocation_matrix(X_sub, new_schedule, 1)

    return min([sub_student.valuation(bundle_1), sub_student.valuation(bundle_2)])


def pairwise_maximin_share(student1, student2, current_bundle_1, current_bundle_2):

    PMMS = {}

    new_schedule, course_strings, course = create_sub_schedule(
        current_bundle_1, current_bundle_2
    )

    PMMS[student1] = yankee_swap_sub_problem(
        student1, new_schedule, course_strings, course
    )
    PMMS[student2] = yankee_swap_sub_problem(
        student2, new_schedule, course_strings, course
    )

    return PMMS


def PMMS_violations(X, students, schedule):
    PMMS_matrix = np.zeros((len(students), len(students)))
    for i, student_1 in enumerate(students):
        bundle_1 = get_bundle_from_allocation_matrix(X, schedule, i)

        for j in range(i + 1, len(students)):
            student_2 = students[j]
            bundle_2 = get_bundle_from_allocation_matrix(X, schedule, j)

            if len(bundle_1) == 0 and len(bundle_2) == 0:
                continue

            PMMS = pairwise_maximin_share(student_1, student_2, bundle_1, bundle_2)
            PMMS_matrix[i, j] = student_1.valuation(bundle_1) - PMMS[student_1]
            PMMS_matrix[j, i] = student_2.valuation(bundle_2) - PMMS[student_2]

    print(PMMS_matrix)

    return np.sum(PMMS_matrix < 0), np.sum(np.any(PMMS_matrix < 0, axis=1))


print(schedule)
print([student.preferred_courses for student in students])

print(X_YS)
print(X_SD)


print(PMMS_violations(X_SD, students, schedule))
