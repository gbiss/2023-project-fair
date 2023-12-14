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
    leg_student1 = LegacyStudent(renaissance1)
    leg_student2 = LegacyStudent(renaissance2)

    original_yankee_swap([leg_student1, leg_student2], schedule)


def test_general_yankee_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1)
    leg_student2 = LegacyStudent(renaissance2)

    general_yankee_swap([leg_student1, leg_student2], schedule)


def test_bfs_yankee_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1)
    leg_student2 = LegacyStudent(renaissance2)

    bfs_yankee_swap([leg_student1, leg_student2], schedule)


def test_round_robin_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1)
    leg_student2 = LegacyStudent(renaissance2)

    round_robin([leg_student1, leg_student2], schedule)


def test_spire_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    leg_student1 = LegacyStudent(renaissance1)
    leg_student2 = LegacyStudent(renaissance2)

    SPIRE_algorithm([leg_student1, leg_student2], schedule)
