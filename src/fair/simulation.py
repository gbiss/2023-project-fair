from typing import List

import numpy as np

from fair.agent import BaseAgent
from fair.constraint import CoursePreferrenceConstraint, LinearConstraint
from fair.feature import Course
from fair.valuation import BaseValuation, ConstraintSatifactionValuation


class SimulatedAgent(BaseAgent):
    """A randomly generated agent"""

    def __init__(self, valuation: BaseValuation, seed: int | None = None):
        """
        Args:
            valuation (BaseValuation): Valuation object to apply to bundles
            seed (int | None, optional): _description_. Defaults to None.
        """
        super().__init__(valuation)
        self.seed = seed
        self.rng = np.random.default_rng(seed)


class RenaissanceMan(SimulatedAgent):
    @staticmethod
    def from_course_lists(
        topic_list: List[List[Course]],
        quantities: List[int],
        course: Course,
        global_constraints: List[LinearConstraint],
        seed: int | None = None,
    ):
        """Randomly generate a Renaissance man student

        The Renaissance man prefers courses from multiple topics, up to a maximum quantity per topic.

        Args:
            topic_list (List[List[Course]]): A list of lists of courses, one per topic
            quantities (List[int]): The maximum number of courses desired per topic
            course (Course): The feature corresponding to courses
            global_constraints (List[LinearConstraint]): Constraints not specific to this agent
            seed (int | None, optional): Random seed. Defaults to None.

        Returns:
            RenaissanceMan: Randomly generated Renaissance man
        """
        preferred_courses = [[], []]
        # fill in preferred_courses with random selection from topic_list subject to quantity constraints

        valuation = ConstraintSatifactionValuation(
            global_constraints
            + [
                CoursePreferrenceConstraint.from_course_lists(
                    preferred_courses, quantities, course
                )
            ]
        )

        return RenaissanceMan(valuation, seed)
