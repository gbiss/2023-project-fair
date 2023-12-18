from fair.agent import LegacyStudent
from fair.allocation import (
    SPIRE_algorithm,
    bfs_yankee_swap,
    general_yankee_swap,
    original_yankee_swap,
    round_robin,
)
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan


def test_original_yankee_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1, renaissance1.all_courses_constraint)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.all_courses_constraint)

    original_yankee_swap([leg_student1, leg_student2], schedule)


def test_general_yankee_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1, renaissance1.all_courses_constraint)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.all_courses_constraint)

    general_yankee_swap([leg_student1, leg_student2], schedule)


def test_bfs_yankee_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1, renaissance1.all_courses_constraint)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.all_courses_constraint)

    bfs_yankee_swap([leg_student1, leg_student2], schedule)


def test_round_robin_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1, renaissance1.all_courses_constraint)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.all_courses_constraint)

    round_robin([leg_student1, leg_student2], schedule)


def test_spire_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1, renaissance1.all_courses_constraint)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.all_courses_constraint)

    SPIRE_algorithm([leg_student1, leg_student2], schedule)
