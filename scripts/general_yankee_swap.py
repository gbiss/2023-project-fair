import os
from collections import defaultdict

import numpy as np
import pandas as pd

from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.metrics import leximin, nash_welfare, utilitarian_welfare
from fair.optimization import StudentAllocationProgram
from fair.simulation import RenaissanceMan

NUM_STUDENTS = 20
MAX_COURSES_PER_TOPIC = 5
MAX_COURSES_TOTAL = 6
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
        MAX_COURSES_TOTAL,
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

X = general_yankee_swap(students, schedule)
print("YS utilitarian welfare: ", utilitarian_welfare(X[0], students, schedule))
print("YS nash welfare: ", nash_welfare(X[0], students, schedule))
print("YS leximin vector: ", leximin(X[0], students, schedule))
print(
    "total bundles evaluated",
    sum([student.student.valuation._value_ct for student in students]),
)
print(
    "unique bundles evaluated",
    sum([student.student.valuation._unique_value_ct for student in students]),
)

if FIND_OPTIMAL:
    orig_students = [student.student for student in students]
    program = StudentAllocationProgram(orig_students, schedule).compile()
    ind = program.convert_allocation(X[0])
    opt_alloc = program.formulateUSW().solve()
    opt_USW = sum(opt_alloc) / len(orig_students)
    print("ILP Allocation count", opt_USW)

    # violoates no constraints
    assert not np.sum(program.A @ ind > program.b) > 0
    # solution is feasible
    assert not np.sum(program.A @ opt_alloc > program.b.T) > 0

    opt_alloc = opt_alloc.reshape(len(students), len(schedule)).transpose()
    total_value = sum(
        [
            orig_students[j].valuation.value(
                [schedule[i] for i in range(107) if opt_alloc[i, j]]
            )
            for j in range(20)
        ]
    )
    allocated_courses = sum(sum(opt_alloc))
    print("TOTAL VALUE:", total_value, "ALLOCATED COURSES:", allocated_courses)
    opt_USW = utilitarian_welfare(opt_alloc, students, schedule)
    opt_nash = nash_welfare(opt_alloc, students, schedule)
    opt_leximin = leximin(opt_alloc, students, schedule)
    print("ILP utilitarian welfare: ", opt_USW)
    print("ILP nash welfare: ", opt_nash)
    print("ILP leximin", opt_leximin)
