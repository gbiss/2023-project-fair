from typing import List

import numpy as np

from fair.agent import BaseAgent
from fair.constraint import LinearConstraint, PreferrenceConstraint
from fair.feature import BaseFeature
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
        features: List[BaseFeature],
        global_constraints: List[LinearConstraint],
        seed: int | None = None,
    ):
        """
        Args:
            topic_list (List[List[ScheduleItem]]): A list of lists of course items, one per topic
            max_quantities (List[int]): The maximum number of courses desired per topic
            features (List[BaseFeature]): The features implemented by course items in topic_list
            global_constraints (List[LinearConstraint]): Constraints not specific to this agent
            seed (int | None, optional): Random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)

        self.quantities = []
        for max_quant in max_quantities:
            self.quantities.append(rng.integers(0, max_quant + 1))

        self.preferred_courses = []
        for i, quant in enumerate(self.quantities):
            self.preferred_courses.append(
                rng.choice(topic_list[i], quant, replace=False).tolist()
            )

        constraints = global_constraints + [
            PreferrenceConstraint.from_item_lists(
                self.preferred_courses, self.quantities, features
            )
        ]

        super().__init__(constraints)
