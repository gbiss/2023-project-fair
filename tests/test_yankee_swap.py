import numpy as np

from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap
from fair.item import ScheduleItem
from fair.metrics import utilitarian_welfare
from fair.optimization import IntegerLinearProgram
from fair.simulation import RenaissanceMan


def test_ys(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
):
    agents = [renaissance1, renaissance2]
    leg_student1 = LegacyStudent(renaissance1, renaissance1.all_courses_constraint)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.all_courses_constraint)
    leg_agents = [leg_student1, leg_student2]

    X, _, _ = general_yankee_swap([leg_student1, leg_student2], schedule)
    util = utilitarian_welfare(X, leg_agents, schedule)
    print(util)

    program = IntegerLinearProgram(agents).compile()
    ind = program.convert_allocation(X)

    # ensure that allocation returned by YS violates no constraints
    assert not np.sum(program.A @ ind > program.b) > 0

