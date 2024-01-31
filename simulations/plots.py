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

import matplotlib.pyplot as plt

NUM_STUDENTS = 50
MAX_COURSES_PER_TOPIC = 5
MAX_COURSES_TOTAL = 6
# EXCEL_SCHEDULE_PATH = os.path.join(
#     os.path.dirname(__file__), "../resources/fall2023schedule-2-cat.xlsx"
# )
# SPARSE = False
# FIND_OPTIMAL = True

# # load schedule as DataFrame
# with open(EXCEL_SCHEDULE_PATH, "rb") as fd:
#     df = pd.read_excel(fd)
# # df=df[10:13]
# # construct features from DataFrame
# course = Course(df["Catalog"].astype(str).unique().tolist())

# time_ranges = df["Mtg Time"].dropna().unique()
# slot = Slot.from_time_ranges(time_ranges, "15T")
# weekday = Weekday()

# section = Section(df["Section"].dropna().unique().tolist())
# features = [course, slot, weekday, section]

# # construct schedule
# schedule = []
# topic_map = defaultdict(set)
# count = 0
# for idx, row in df.iterrows():
#     crs = str(row["Catalog"])
#     topic_map[row["Categories"]].add(crs)
#     slt = slots_for_time_range(row["Mtg Time"], slot.times)
#     sec = row["Section"]
#     capacity = row["CICScapacity"]
#     # capacity=20
#     dys = tuple([day.strip() for day in row["zc.days"].split(" ")])
#     schedule.append(
#         ScheduleItem(features, [crs, slt, dys, sec], index=count, capacity=capacity)
#     )
#     count += 1

# topics = sorted([sorted(list(courses)) for courses in topic_map.values()])

# # global constraints
# course_time_constr = CourseTimeConstraint.from_items(schedule, slot, weekday, SPARSE)
# course_sect_constr = MutualExclusivityConstraint.from_items(schedule, course, SPARSE)

# # randomly generate students
# students0 = []
# students = []
# for i in range(NUM_STUDENTS):
#     student = RenaissanceMan(
#         topics,
#         [min(len(topic), MAX_COURSES_PER_TOPIC) for topic in topics],
#         MAX_COURSES_TOTAL,
#         course,
#         [course_time_constr, course_sect_constr],
#         schedule,
#         seed=i,
#         sparse=SPARSE,
#     )
#     students0.append(student)
#     legacy_student = LegacyStudent(student, student.all_courses_constraint)
#     legacy_student.student.valuation.valuation = (
#         legacy_student.student.valuation.compile()
#     )
#     students.append(legacy_student)

#Plot times
Ns=[50,100,500,1000]
times=[]
bundles=[]
ubundles=[]
times1=[]
bundles1=[]
ubundles1=[]
for N in Ns:
    data=np.load(f'YS_ILP_{N}_2.npz')
    X=data['X']
    time_steps=data['time_steps']
    num_agents_involved=data['num_agents_involved']
    bundles_eval=data['bundles_eval']
    unique_bundles_eval=data['unique_bundles_eval']
    bundles_eval_aray=data['eval_bundles']
    unique_bundles_eval_array=data['unique_eval_bundles']
    YS_USW=data['YS_USW']
    YS_nash=data['YS_nash']
    YS_leximin=data['YS_leximin']
    ilp_alloc=data['ilp_alloc']
    ilp_USW=data['ilp_USW']
    ilp_nash=data['ilp_nash']
    ilp_leximin=data['ilp_leximin']
    times.append(time_steps[-1]/60)
    bundles.append(sum(bundles_eval))
    ubundles.append(sum(unique_bundles_eval))
    data=np.load(f'YS_ILP_{N}_3.npz')
    X=data['X']
    time_steps=data['time_steps']
    num_agents_involved=data['num_agents_involved']
    bundles_eval=data['bundles_eval']
    unique_bundles_eval=data['unique_bundles_eval']
    bundles_eval_aray=data['eval_bundles']
    unique_bundles_eval_array=data['unique_eval_bundles']
    YS_USW=data['YS_USW']
    YS_nash=data['YS_nash']
    YS_leximin=data['YS_leximin']
    ilp_alloc=data['ilp_alloc']
    ilp_USW=data['ilp_USW']
    ilp_nash=data['ilp_nash']
    ilp_leximin=data['ilp_leximin']
    times1.append(time_steps[-1]/60)
    bundles1.append(sum(bundles_eval))
    ubundles1.append(sum(unique_bundles_eval))
print(times)
print(bundles)
print(ubundles)
print(times1)
print(bundles1)
print(ubundles1)
plt.plot(Ns,times, marker='*')
plt.xticks(Ns)
plt.xlabel('NUM_STUDENTS')
plt.ylabel('Running time (min)')
plt.show()

data1 = times
data2 = bundles
data3=times1
data4=bundles1
# data2 = unique_bundles_eval_array


# data1=np.diff(time_steps)
# data2=np.diff(bundles_eval_aray)
# data2=np.diff(unique_bundles_eval_array)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('NUM_STUDENTS')
ax1.set_ylabel('Running time (min)', color=color)
ax1.plot(Ns, data1, color=color, alpha=0.8, label='Recursive valuation time')
ax1.plot(Ns, data3, color='C1', alpha=0.8, label='Linear valuation time')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(Ns)
plt.legend()
ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.set_ylabel('Total bundles evaluated', color=color)  
# ax2.set_ylabel('Unique bundles evaluated', color=color)  
ax2.plot(Ns, data2, color=color, alpha=0.8, label='Recursive valuation bundles')
ax2.plot(Ns, data4, color='C4', alpha=0.8, label='Linear valuation bundles')
ax2.tick_params(axis='y', labelcolor=color)
plt.legend(loc='center left')
fig.tight_layout() 
plt.show()



N=1000
data=np.load(f'YS_ILP_{N}_3.npz')
X=data['X']
time_steps=data['time_steps']
num_agents_involved=data['num_agents_involved']
bundles_eval=data['bundles_eval']
unique_bundles_eval=data['unique_bundles_eval']
bundles_eval_aray=data['eval_bundles']
unique_bundles_eval_array=data['unique_eval_bundles']
YS_USW=data['YS_USW']
YS_nash=data['YS_nash']
YS_leximin=data['YS_leximin']
ilp_alloc=data['ilp_alloc']
ilp_USW=data['ilp_USW']
ilp_nash=data['ilp_nash']
ilp_leximin=data['ilp_leximin']

#Plot evaluated bundles versus running time

data1 = time_steps
data2 = bundles_eval_aray
# data2 = unique_bundles_eval_array


# data1=np.diff(time_steps)
# data2=np.diff(bundles_eval_aray)
# data2=np.diff(unique_bundles_eval_array)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Step')
ax1.set_ylabel('Running time (s)', color=color)
ax1.plot(range(len(data1)), data1, color=color, alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.set_ylabel('Total bundles evaluated', color=color)  
# ax2.set_ylabel('Unique bundles evaluated', color=color)  
ax2.plot(range(len(data2)), data2, color=color, alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout() 
plt.show()
plt.close()

colors = np.diff(time_steps)
area = (2 * colors)**2
print(np.diff(time_steps))
# plt.set_xticks([0,2,4,6,8])
plt.scatter(num_agents_involved[1:],np.diff(time_steps),s=area,c=colors, alpha=0.5)
plt.ylabel('Running time (s)')
plt.xlabel('Num of involved agents')
plt.xticks([0,1,2])
plt.show()
area = (colors)**2
plt.scatter(np.diff(bundles_eval_aray),np.diff(time_steps),s=area,c=colors, alpha=0.5)
plt.ylabel('Running time (s)')
plt.xlabel('Total bundles evaluated')
plt.show()
plt.scatter(np.diff(unique_bundles_eval_array),np.diff(time_steps),s=area,c=colors, alpha=0.5)
plt.ylabel('Running time (s)')
plt.xlabel('Unique bundles evaluated')
plt.show()

N_STEPS=len(time_steps)
print(N_STEPS)
print(time_steps[-1])
plt.plot(range(len(time_steps)),time_steps)
plt.plot(range(len(time_steps)),unique_bundles_eval_array)
plt.show()

plt.plot(range(len(ilp_leximin)), ilp_leximin)
plt.plot(range(len(ilp_leximin)), YS_leximin)
plt.show()