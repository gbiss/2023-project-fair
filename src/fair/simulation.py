from typing import List

import numpy as np

from fair.agent import BaseAgent
from fair.constraint import LinearConstraint, PreferenceConstraint
from fair.feature import BaseFeature, Course
from fair.item import ScheduleItem
from fair.valuation import ConstraintSatifactionValuation


class SimulatedAgent(BaseAgent):
    """A randomly generated agent"""

    def __init__(self, constraints: List[LinearConstraint]):
        """
        Args:
            constraints (List[LinearConstraint]): constraints to be used in defining valuation
        """
        super().__init__(ConstraintSatifactionValuation(constraints))


class RenaissanceMan(SimulatedAgent):
    """Randomly generate a Renaissance man student

    The Renaissance man prefers courses from multiple topics, up to a maximum quantity per topic.
    """

    def __init__(
        self,
        topic_list: List[List[ScheduleItem]],
        max_quantities: List[int],
        max_courses: int,
        course: Course,
        global_constraints: List[LinearConstraint],
        schedule: List[ScheduleItem] = None,
        features: List[BaseFeature] = None,
        seed: int | None = None,
    ):
        """
        Args:
            topic_list (List[List[ScheduleItem]]): A list of lists of course items, one per topic
            max_quantities (List[int]): The maximum number of courses desired per topic
            max_courses (int): Student can take no more than max_courses total
            course (Course): Feature for course
            global_constraints (List[LinearConstraint]): Constraints not specific to this agent
            schedule (List[ScheduleItem], optional): All possible items in the student's schedule. Defaults to None.
            features (List[BaseFeature], optional): The features implemented by items in schedule. Defaults to None.
            seed (int | None, optional): Random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)

        self.quantities = []
        for max_quant in max_quantities:
            self.quantities.append(rng.integers(0, max_quant + 1))

        self.preferred_topics = []
        self.preferred_courses = []
        for i, quant in enumerate(self.quantities):
            topic = rng.choice(topic_list[i], quant, replace=False).tolist()
            self.preferred_topics.append(topic)
            self.preferred_courses += topic

        self.total_courses = rng.integers(1, max_courses + 1)
        self.all_courses_constraint = PreferenceConstraint.from_item_lists(
            [self.preferred_courses],
            [self.total_courses],
            course,
            schedule,
            features,
        )
        self.topic_constraint = PreferenceConstraint.from_item_lists(
            self.preferred_topics, self.quantities, course, schedule, features
        )

        constraints = global_constraints + [
            self.all_courses_constraint,
            self.topic_constraint,
        ]

        super().__init__(constraints)
