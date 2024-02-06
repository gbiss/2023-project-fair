import os
from collections import defaultdict

import pandas as pd
import numpy as np

from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap, SPIRE_algorithm, round_robin
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.metrics import (
    leximin,
    nash_welfare,
    utilitarian_welfare,
    EF_count,
    EF_agents,
    EF_1_count,
    EF_1_agents,
    EF_X_count,
    EF_X_agents
)
from fair.optimization import StudentAllocationProgram
from fair.simulation import RenaissanceMan

import matplotlib.pyplot as plt

NUM_STUDENTS = 100
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
# df=df[10:13]
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
count = 0
for idx, row in df.iterrows():
    crs = str(row["Catalog"])
    topic_map[row["Categories"]].add(crs)
    slt = slots_for_time_range(row["Mtg Time"], slot.times)
    sec = row["Section"]
    capacity = row["CICScapacity"]
    dys = tuple([day.strip() for day in row["zc.days"].split(" ")])
    schedule.append(
        ScheduleItem(features, [crs, slt, dys, sec], index=count, capacity=capacity)
    )
    # print(crs,dys, slt)
    count += 1

topics = sorted([sorted(list(courses)) for courses in topic_map.values()])

# global constraints
course_time_constr = CourseTimeConstraint.from_items(schedule, slot, weekday, SPARSE)
course_sect_constr = MutualExclusivityConstraint.from_items(schedule, course, SPARSE)

# randomly generate students
students0 = []
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
    # print('prefered classes:', student.preferred_courses)
    students0.append(student)
    legacy_student = LegacyStudent(student, student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    students.append(legacy_student)


X_SP=SPIRE_algorithm(students, schedule)
print("SP utilitarian welfare: ", utilitarian_welfare(X_SP, students, schedule))
print("SP nash welfare: ", nash_welfare(X_SP, students, schedule))
print("SP leximin vector: ", leximin(X_SP, students, schedule))
print("SP EF_count: ", EF_count(X_SP, students, schedule))
print("SP EF_agents: ", EF_agents(X_SP, students, schedule))
print("SP EF_1_count: ", EF_1_count(X_SP, students, schedule))
print("SP EF_1_agents: ", EF_1_agents(X_SP, students, schedule))
print("SP EF_X_count: ", EF_X_count(X_SP, students, schedule))
print("SP EF_X_agents: ", EF_X_agents(X_SP, students, schedule))

X_RR=round_robin(students, schedule)
print("RR utilitarian welfare: ", utilitarian_welfare(X_RR, students, schedule))
print("RR nash welfare: ", nash_welfare(X_RR, students, schedule))
print("RR leximin vector: ", leximin(X_RR, students, schedule))
print("RR EF_count: ", EF_count(X_RR, students, schedule))
print("RR EF_agents: ", EF_agents(X_RR, students, schedule))
print("RR EF_1_count: ", EF_1_count(X_RR, students, schedule))
print("RR EF_1_agents: ", EF_1_agents(X_RR, students, schedule))
print("RR EF_X_count: ", EF_X_count(X_RR, students, schedule))
print("RR EF_X_agents: ", EF_X_agents(X_RR, students, schedule))

data=np.load(f'YS_ILP_{NUM_STUDENTS}_4.npz')
X_YS=data['X']
time_steps=data['time_steps']
num_agents_involved=data['num_agents_involved']
bundles_eval=data['bundles_eval']
unique_bundles_eval=data['unique_bundles_eval']
bundles_eval_aray=data['eval_bundles']
unique_bundles_eval_array=data['unique_eval_bundles']

print('total time',time_steps[-1])
print('total bundles:',sum(bundles_eval))
print('unique_bundles:', sum(unique_bundles_eval))

print("YS utilitarian welfare: ", utilitarian_welfare(X_YS, students, schedule))
print("YS nash welfare: ", nash_welfare(X_YS, students, schedule))
print("YS leximin vector: ", leximin(X_YS, students, schedule))
print("YS EF_count: ", EF_count(X_YS, students, schedule))
print("YS EF_agents: ", EF_agents(X_YS, students, schedule))
print("YS EF_1_count: ", EF_1_count(X_YS, students, schedule))
print("YS EF_1_agents: ", EF_1_agents(X_YS, students, schedule))
print("YS EF_X_count: ", EF_X_count(X_YS, students, schedule))
print("YS EF_X_agents: ", EF_X_agents(X_YS, students, schedule))

X_ILP=data['ilp_alloc']
print(
        "ILP utilitarian welfare: ", utilitarian_welfare(X_ILP, students, schedule)
    )
print("ILP nash welfare: ", nash_welfare(X_ILP, students, schedule))
print("ILP leximin vector: ", leximin(X_ILP, students, schedule))
print("ILP EF_count: ", EF_count(X_ILP, students, schedule))
print("ILP EF_agents: ", EF_agents(X_ILP, students, schedule))
print("ILP EF_1_count: ", EF_1_count(X_ILP, students, schedule))
print("ILP EF_1_agents: ", EF_1_agents(X_ILP, students, schedule))
print("ILP EF_X_count: ", EF_X_count(X_ILP, students, schedule))
print("ILP EF_X_agents: ", EF_X_agents(X_ILP, students, schedule))
