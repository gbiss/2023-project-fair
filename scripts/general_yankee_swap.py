import os
from collections import defaultdict

import pandas as pd
import numpy as np

from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.metrics import leximin, nash_welfare, utilitarian_welfare
from fair.optimization import StudentAllocationProgram
from fair.simulation import RenaissanceMan

NUM_STUDENTS = 50
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
    # capacity=20
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

X = general_yankee_swap(students, schedule)
bundles_eval=[student.valuation._value_ct for student in students0]
unique_bundles_eval=[student.valuation._unique_value_ct for student in students0]
YS_USW=utilitarian_welfare(X[0], students, schedule)
YS_nash=nash_welfare(X[0], students, schedule)
YS_leximin=leximin(X[0], students, schedule)

print("utilitarian welfare: ", YS_USW)
print("nash welfare: ", YS_nash)
print("leximin vector: ", YS_leximin)
print("total bundles evaluated", sum(bundles_eval))
print("unique bundles evaluated", sum(unique_bundles_eval ))

if FIND_OPTIMAL:
    orig_students = [student.student for student in students]
    program = StudentAllocationProgram(orig_students, schedule).compile()
    ind = program.convert_allocation(X[0])
    opt_alloc = program.formulateUSW().solve()
    opt_USW = sum(opt_alloc) / len(orig_students)
    print("optimal utilitarian welfare", opt_USW)
    opt_alloc=opt_alloc.reshape(len(students), len(schedule)).transpose()

    opt_USW=utilitarian_welfare(opt_alloc, students, schedule)
    opt_nash=nash_welfare(opt_alloc, students, schedule)
    opt_leximin=leximin(opt_alloc, students, schedule)
    print("optimal utilitarian welfare: ", opt_USW)
    print("ILP nash welfare: ", opt_nash)
    print("ILP leximin", opt_leximin)

print(YS_leximin)
print(opt_leximin)
# print('Ys allocation',X[0])
# print(opt_alloc)
print(X[0].shape, opt_alloc.shape)
np.savez(f'simulations/YS_ILP_{NUM_STUDENTS}_3.npz',X=X[0],time_steps=X[1],num_agents_involved=X[2], bundles_eval=bundles_eval,unique_bundles_eval=unique_bundles_eval,eval_bundles=X[3], unique_eval_bundles=X[4], YS_USW=YS_USW, YS_nash=YS_nash, YS_leximin=YS_leximin, ilp_alloc=opt_alloc,ilp_USW=opt_USW, ilp_nash=opt_nash, ilp_leximin=opt_leximin)