import os
from collections import defaultdict

import pandas as pd
import numpy as np
import random
import sys
import time

from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap_E, round_robin, SPIRE_algorithm
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
from fair.simulation import RenaissanceMan

seed = int(sys.argv[1])
seed_order = int(sys.argv[2])
print(sys.argv)
random.seed(seed)
NUM_STUDENTS = 100
MAX_COURSES_PER_TOPIC = 5
LOWER_MAX_COURSES_TOTAL = 2
UPPER_MAX_COURSES_TOTAL = 5
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
ii32 = np.iinfo(np.int32(10))
max_value = ii32.max

seeds = random.sample(range(1, max_value), NUM_STUDENTS)
random.seed(seed_order)
random.shuffle(seeds)

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
        seed=seeds[i],
        sparse=SPARSE,
    )
    legacy_student = LegacyStudent(student, student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    students.append(legacy_student)

X = general_yankee_swap_E(students, schedule)
print(X)
total_bundles = sum([student.student.valuation._value_ct for student in students])
unique_bundles = sum(
    [student.student.valuation._unique_value_ct for student in students]
)
print("total bundles evaluated", total_bundles)
print("unique bundles evaluated", unique_bundles)
print("YS utilitarian welfare: ", utilitarian_welfare(X[0], students, schedule))
print("YS nash welfare: ", nash_welfare(X[0], students, schedule))

if FIND_OPTIMAL:
    orig_students = [student.student for student in students]
    start = time.process_time()
    program = StudentAllocationProgram(orig_students, schedule).compile()
    opt_alloc = program.formulateUSW().solve()
    time_ILP = time.process_time() - start
    opt_USW = sum(opt_alloc) / len(orig_students)
    print("optimal utilitarian welfare", opt_USW)
    opt_alloc = opt_alloc.reshape(len(students), len(schedule)).transpose()
    print(
        "ILP utilitarian welfare: ", utilitarian_welfare(opt_alloc, students, schedule)
    )
    print("ILP nash welfare: ", nash_welfare(opt_alloc, students, schedule))

start = time.process_time()
X_SPIRE = SPIRE_algorithm(students, schedule)
time_SPIRE = time.process_time() - start

start = time.process_time()
X_RR = round_robin(students, schedule)
time_RR = time.process_time() - start


np.savez(
    f"./simulations/allocation_{NUM_STUDENTS}_{seed}_{seed_order}.npz",
    X_YS=X[0],
    time_steps=X[1],
    n_agents_involved=X[2],
    total_bundles=total_bundles,
    unique_bundles=unique_bundles,
    X_ILP=opt_alloc,
    time_ILP=time_ILP,
    X_RR=X_RR,
    time_RR=time_RR,
    X_SPIRE=X_SPIRE,
    time_SPIRE=time_SPIRE
)
