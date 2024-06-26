import re
from datetime import datetime

import numpy as np
import pandas as pd

from fair.agent import LegacyStudent
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan

DEFAULT_CAPACITY = 30
SPARSE = False

survey_file = "resources/random_survey.csv"
mapping_file = "resources/survey_column_mapping.csv"
df = pd.read_csv(survey_file)
mapping = pd.read_csv(mapping_file, sep="|")
questions = ["1", "2", "3", "4"] + [f"5#1_{i}" for i in range(1, 12)]
cics_courses = [col for col in df.columns if re.match("7 _\d+$", col)]
compsci_courses = [col for col in df.columns if re.match("7_\d+$", col)]
info_courses = [col for col in df.columns if re.match("7 _\d\.", col)]
all_courses = cics_courses + compsci_courses + info_courses
df = df[questions + all_courses]


def extract_course_info(html):
    course_match = re.search(r"<strong>Course:.*?</strong>(.*?)\s*&nbsp;", html)
    parts = course_match.group(1).strip().split(" ")
    catalog = parts[0]
    try:
        course, section = parts[1].split("-")
    except ValueError:
        course, section = parts[1], "01"
    description = " ".join(parts[2:])

    return catalog, course, section, description


def extract_instructor_info(html):
    instructor_match = re.search(
        r"<strong>Instructor:.*?</strong>(.*?)(?:\s*&nbsp;|$)", html
    )

    return instructor_match.group(1).strip() if instructor_match else None


def extract_schedule_info(html):
    schedule_match = re.search(r"<strong>Schedule:.*?</strong>(.*?)\s*&nbsp;", html)
    schedule = schedule_match.group(1).strip() if schedule_match else None

    days_str, start_time, end_time = None, None, None
    if schedule:
        day_time_match = re.match(
            r"([A-Za-z]+)\s+(\d{2}:\d{2}\s+[AP]M)\s+-\s+(\d{2}:\d{2}\s+[AP]M)", schedule
        )
        if day_time_match:
            days_str, start_time_str, end_time_str = day_time_match.groups()
            start_time = datetime.strptime(start_time_str, "%I:%M %p").time()
            end_time = datetime.strptime(end_time_str, "%I:%M %p").time()

    return days_str, start_time, end_time


# construct course information map
course_map = {}
eliminated = []
for crs in all_courses:
    raw_description = mapping[mapping.question == crs].description.values[0]
    catalog, course_num, section, description = extract_course_info(raw_description)
    instructor = extract_instructor_info(raw_description)
    days, start_time, end_time = extract_schedule_info(raw_description)
    try:
        course_map[crs] = {
            "catalog": catalog,
            "course num": course_num,
            "section": section,
            "description": description,
            "instructor": instructor,
            "days": days,
            "time range": start_time.strftime("%I:%M %p")
            + " - "
            + end_time.strftime("%I:%M %p"),
        }
    except AttributeError:
        eliminated.append(crs)

all_courses = [crs for crs in all_courses if crs not in eliminated]

# construct features
course = Course([entry["course num"] for entry in course_map.values()])
slot = Slot.from_time_ranges(
    [entry["time range"] for entry in course_map.values()], "15T"
)
weekday = Weekday()
section = Section([entry["section"] for entry in course_map.values()])
features = [course, slot, weekday, section]

# construct schedule
schedule = []
days = Weekday().days
for idx, (crs, map) in enumerate(course_map.items()):
    crs = str(map["course num"])
    slt = slots_for_time_range(map["time range"], slot.times)
    sec = map["section"]
    capacity = DEFAULT_CAPACITY
    dys = tuple([day for day in days if day in map["days"]])
    schedule.append(
        ScheduleItem(features, [crs, slt, dys, sec], index=idx, capacity=capacity)
    )

topics = [sorted([item["course num"] for item in course_map.values()])]

# global constraints
course_time_constr = CourseTimeConstraint.from_items(schedule, slot, weekday, SPARSE)
course_sect_constr = MutualExclusivityConstraint.from_items(schedule, course, SPARSE)

# construct students
students = []
for idx, row in df.iterrows():
    preferred = [
        course_map[crs]["course num"]
        for crs in all_courses
        if not np.isnan(row[crs]) and row[crs] > 1
    ]
    total_num_courses = row["3"]
    student = RenaissanceMan(
        topics,
        [min(len(topic), total_num_courses) for topic in topics],
        total_num_courses,
        total_num_courses,
        course,
        [course_time_constr, course_sect_constr],
        schedule,
        seed=idx,
        sparse=SPARSE,
    )
    legacy_student = LegacyStudent(student, student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    students.append(legacy_student)
