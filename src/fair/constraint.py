from collections import defaultdict
from typing import Any, List

import numpy as np
from scipy.sparse import dok_array

from .feature import BaseFeature, Course, Section, Slot
from .item import BaseItem, ScheduleItem


def indicator(bundle: List[BaseItem], extent: int):
    """Indicator vector for bundle over domain of features

    This method returns a vector of length extent with a 1 at the index of each item
    in the bundle

    Args:
        bundle (List[BaseItem]): Items whose index we would like to identify
        extent (int): Maximum index in domain

    Returns:
        scipy.sparse.dok_array: Indicator vector of bundle indices
    """
    rows = extent
    ind = dok_array((rows, 1), dtype=np.int_)
    for item in bundle:
        ind[item.index, 0] = True

    return ind.tocsr()


class BaseConstraint:
    pass


class LinearConstraint(BaseConstraint):
    """Constraints that can be expressed in the form A*x <= b"""

    def __init__(
        self,
        A: dok_array,
        b: dok_array,
        extent: int,
    ):
        """
        Args:
            A (dok_array): Constraint matrix
            b (dok_array): Row capacities
            extent (int): Largest possible index value
        """
        self.A = A.tocsr()
        self.b = b.tocsr()
        self.extent = extent

    def satisfies(self, bundle: List[BaseItem]):
        """Determine if bundle satisfies this constraint

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Returns:
            bool: True if the constraint is satisfied; False otherwise
        """
        ind = indicator(bundle, self.extent)
        product = self.A @ ind

        # apparently <= is much less efficient than using < and != separately
        less_than = (product < self.b).toarray().flatten()
        equal_to = ~(product != self.b).toarray().flatten()

        return np.prod([lt or eq for lt, eq in zip(less_than, equal_to)])

    def constrained_items(self, items: BaseItem):
        """Determine if, and for what constraint, each item is constrained

        Args:
            items (BaseItem): Items to evaluate

        Returns:
            Dict(BaseItem, List[int]): List of constraints (rows of A) where each item is constrained
        """
        active_map = defaultdict(list)
        for i in range(self.A.shape[0]):
            for item in items:
                if self.A[i, item.index] != 0:
                    active_map[item].append(i)

        return active_map


class PreferenceConstraint(LinearConstraint):
    @staticmethod
    def from_item_lists(
        schedule: List[BaseItem],
        preferred_values: List[List[Any]],
        limits: List[int],
        preferred_feature: BaseFeature,
    ):
        """A helper method for constructing preference constraints

        This constraint ensures that the bundle contains a limited number of pre-selected items from
        each provided category (e.g. Physics, Chemistry, etc. for course items).

        Args:
            schedule: (List[BaseItem], optional): Universe of all items under consideration
            preferred_values (List[List[Any]]): Each list is a category and values in that list are preferred
            limits (List[int]): The maximum number of items desired per category
            preferred_feature (BaseFeature): The feaure in terms of which preferred values are expressed

        Raises:
            IndexError: Number of categories must match among preferred_items and limits

        Returns:
            PreferrenceConstraint: A: (categories x features domain), b: (categories x 1)
        """
        if len(preferred_values) != len(limits):
            raise IndexError("item and limit lists must have the same length")

        rows = len(preferred_values)
        cols = max([item.index for item in schedule]) + 1
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i in range(rows):
            for value in preferred_values[i]:
                for item in schedule:
                    if item.value(preferred_feature) == value:
                        A[
                            i,
                            item.index,
                        ] = 1
            b[i, 0] = limits[i]

        return LinearConstraint(A, b, cols)


class CourseTimeConstraint(LinearConstraint):
    @staticmethod
    def from_items(items: List[ScheduleItem], slot: Slot):
        """Helper method for creating constraints that prevent course time overlap

        A bundle satisfies this constraint only if no two courses meet at the same time.

        Args:
            items (List[ScheduleItem]): Possibly time-conflicting items
            slot (Slot): Feature for time slots

        Returns:
            CourseTimeConstraint: A: (time slots x features domain), b: (time slots x 1)
        """
        rows = len(slot.times)
        cols = max([item.index for item in items]) + 1
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i, tm in enumerate(slot.times):
            items_at_time = [item for item in items if tm in item.value(slot)]
            for item in items_at_time:
                A[i, item.index] = 1
            b[i, 0] = 1

        return LinearConstraint(A, b, cols)


class MutualExclusivityConstraint(LinearConstraint):
    @staticmethod
    def from_items(
        items: List[ScheduleItem],
        exclusive_feature: BaseFeature,
    ):
        """Helper method for creating constraints that prevent scheduling multiple sections of the same class

        Args:
            items (List[ScheduleItem]): Items, possibly having same value for exclusive_feature
            exclusive_feature (BaseFeature): Feature that must remain exclusive

        Returns:
            MutualExclusivityConstraint: A: (exclusive_feature domain x features domain), b: (exclusive_feature domain x 1)
        """
        rows = len(exclusive_feature.domain)
        cols = max([item.index for item in items]) + 1
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i, excl in enumerate(exclusive_feature.domain):
            items_for_course = [
                item for item in items if item.value(exclusive_feature) == excl
            ]
            for item in items_for_course:
                A[i, item.index] = 1
            b[i, 0] = 1

        return LinearConstraint(A, b, cols)
