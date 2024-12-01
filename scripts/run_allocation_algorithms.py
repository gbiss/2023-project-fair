import os
from collections import defaultdict

import pandas as pd

from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap_E, serial_dictatorship, round_robin
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.metrics import (
    leximin,
    nash_welfare,
    utilitarian_welfare,
)
from fair.envy import (
    precompute_bundles_valuations,
    EF_violations,
    EF1_violations,
    EFX_violations,
)
from fair.optimization import StudentAllocationProgram
from fair.simulation import RenaissanceMan

NUM_STUDENTS = 300
MAX_COURSES_PER_TOPIC = 15
LOWER_MAX_COURSES_TOTAL = 10
UPPER_MAX_COURSES_TOTAL = 15
EXCEL_SCHEDULE_PATH = os.path.join(
    os.path.dirname(__file__), "../resources/fall2023schedule-2-cat.xlsx"
)
SPARSE = False
FIND_OPTIMAL = True

# load schedule as DataFrame
with open(EXCEL_SCHEDULE_PATH, "rb") as fd:
    df = pd.read_excel(fd)

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
    capacity = row["CICScapacity"]
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

X_YS, _, _ = general_yankee_swap_E(students, schedule)
print("YS utilitarian welfare: ", utilitarian_welfare(X_YS, students, schedule))
print("YS nash welfare: ", nash_welfare(X_YS, students, schedule))
print("YS leximin vector: ", leximin(X_YS, students, schedule))
bundles, valuations = precompute_bundles_valuations(X_YS, students, schedule)
print("YS EF_violations: ", EF_violations(X_YS, students, schedule, valuations))
print(
    "YS EF1_violations: ", EF1_violations(X_YS, students, schedule, bundles, valuations)
)
print(
    "YS EFX_violations: ", EFX_violations(X_YS, students, schedule, bundles, valuations)
)

X_SD = serial_dictatorship(students, schedule)
print("SD utilitarian welfare: ", utilitarian_welfare(X_SD, students, schedule))
print("SD nash welfare: ", nash_welfare(X_SD, students, schedule))
print("SD leximin vector: ", leximin(X_SD, students, schedule))
bundles, valuations = precompute_bundles_valuations(X_SD, students, schedule)
print("SD EF_violations: ", EF_violations(X_SD, students, schedule, valuations))
print(
    "SD EF1_violations: ", EF1_violations(X_SD, students, schedule, bundles, valuations)
)
print(
    "SD EFX_violations: ", EFX_violations(X_SD, students, schedule, bundles, valuations)
)

X_RR = round_robin(students, schedule)
print("RR utilitarian welfare: ", utilitarian_welfare(X_RR, students, schedule))
print("RR nash welfare: ", nash_welfare(X_RR, students, schedule))
print("RR leximin vector: ", leximin(X_RR, students, schedule))
bundles, valuations = precompute_bundles_valuations(X_RR, students, schedule)
print("RR EF_violations: ", EF_violations(X_RR, students, schedule, valuations))
print(
    "RR EF1_violations: ", EF1_violations(X_RR, students, schedule, bundles, valuations)
)
print(
    "RR EFX_violations: ", EFX_violations(X_RR, students, schedule, bundles, valuations)
)

if FIND_OPTIMAL:
    orig_students = [student.student for student in students]
    program = StudentAllocationProgram(orig_students, schedule).compile()
    opt_alloc = program.formulateUSW().solve()
    opt_USW = sum(opt_alloc) / len(orig_students)
    # print("optimal utilitarian welfare", opt_USW)

    opt_alloc = opt_alloc.reshape(len(students), len(schedule)).transpose()
    print(
        "ILP utilitarian welfare: ", utilitarian_welfare(opt_alloc, students, schedule)
    )
    print("ILP nash welfare: ", nash_welfare(opt_alloc, students, schedule))
    print("ILP leximin vector: ", leximin(opt_alloc, students, schedule))
    bundles, valuations = precompute_bundles_valuations(opt_alloc, students, schedule)
    print(
        "ILP EF_violations: ", EF_violations(opt_alloc, students, schedule, valuations)
    )
    print(
        "RR EF1_violations: ",
        EF1_violations(opt_alloc, students, schedule, bundles, valuations),
    )
    print(
        "RR EFX_violations: ",
        EFX_violations(opt_alloc, students, schedule, bundles, valuations),
    )
