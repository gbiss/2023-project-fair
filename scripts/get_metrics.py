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


data=np.load(f'./simulations/allocation_{NUM_STUDENTS}_{seed}_{seed_order}.npz')
X_YS=data['X_YS']
time_steps=data['time_steps']
n_agents_involved=data['n_agents_involved']
total_bundles=data['total_bundles']
unique_bundles=data['unique_bundles']
X_ILP=data['X_ILP']
time_ILP=data['time_ILP']
X_RR=data['X_RR']
time_RR=data['time_RR']
X_SPIRE=data['X_SPIRE']
time_SPIRE=data['time_SPIRE']

print("YS utilitarian welfare: ", utilitarian_welfare(X_YS, students, schedule))
print("YS nash welfare: ", nash_welfare(X_YS, students, schedule))

print("ILP utilitarian welfare: ", utilitarian_welfare(X_ILP, students, schedule))
print("ILP nash welfare: ", nash_welfare(X_ILP, students, schedule))