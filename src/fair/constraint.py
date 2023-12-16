from collections import defaultdict
from typing import List

import numpy as np
from scipy.sparse import dok_array

from .feature import BaseFeature, Course, Section, Slot
from .item import BaseItem, ScheduleItem


def indicator(features: List[BaseFeature], bundle: List[BaseItem]):
    """Indicator vector for bundle over domain of features

    Associated with each item in the bundle is a point in the cartesian product of
    domains associated with the features. This method returns a vector with a 1 at
    the index associated with each of these points, and a zero everywhere else.

    Args:
        features (List[BaseFeature]): Subset of features over which the bundle items are defined
        bundle (List[BaseItem]): Items whose index we would like to identify

    Returns:
        scipy.sparse.dok_array: Indicator vector of bundle indices
    """
    rows = np.prod([len(feature.domain) for feature in features])
    ind = dok_array((rows, 1), dtype=np.int_)
    for item in bundle:
        ind[item.index(features), 0] = True

    return ind


class BaseConstraint:
    pass


class LinearConstraint(BaseConstraint):
    """Constraints that can be expressed in the form A*x <= b"""

    def __init__(self, A: dok_array, b: dok_array, features: List[BaseFeature]):
        """
        Args:
            A (dok_array): Constraint matrix
            b (dok_array): Row capacities
            features (List[BaseFeature]): Features relevant for this constraint
        """
        self.A = A
        self.b = b
        self.features = features

    def satisfies(self, bundle: List[BaseItem]):
        """Determine if bundle satisfies this constraint

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Returns:
            bool: True if the constraint is satisfied; False otherwise
        """
        ind = indicator(self.features, bundle)
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
                if self.A[i, item.index(self.features)] != 0:
                    active_map[item].append(i)

        return active_map


class PreferenceConstraint(LinearConstraint):
    @staticmethod
    def from_item_lists(
        preferred_items: List[List[BaseItem]],
        limits: List[int],
        features: List[BaseFeature],
    ):
        """A helper method for constructing preference constraints

        This constraint ensures that the bundle contains a limited number of pre-selected items from
        each provided category (e.g. Physics, Chemistry, etc. for course items)

        Args:
            preferred_items (List[List[BaseItem]]): Each list is a category and items in that list are preferred
            limits (List[int]): The maximum number of items desired per category
            features (List[BaseFeature]): Feature to be used for items

        Raises:
            IndexError: Number of categories must match among preferred_items and limits

        Returns:
            PreferrenceConstraint: A: (categories x features domain), b: (categories x 1)
        """
        if len(preferred_items) != len(limits):
            raise IndexError("item and limit lists must have the same length")

        rows = len(preferred_items)
        cols = np.prod([len(feature.domain) for feature in features])
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i in range(rows):
            for item in preferred_items[i]:
                A[
                    i,
                    item.index(features),
                ] = 1
            b[i, 0] = limits[i]

        return LinearConstraint(A, b, features)

    def __init__(self, A: dok_array, b: dok_array, features: List[BaseFeature]):
        super().__init__(A, b, features)


class CourseTimeConstraint(LinearConstraint):
    @staticmethod
    def from_items(items: List[ScheduleItem], slot: Slot, features: List[BaseFeature]):
        """Helper method for creating constraints that prevent course time overlap

        A bundle satisfies this constraint only if no two courses meet at the same time.

        Args:
            items (List[ScheduleItem]): Possibly time-conflicting items
            slot (Slot): Feature for time slots
            features (List[BaseFeature]): Feature list for items

        Raises:
            AttributeError: Features list is required to contain the provided Slot feature

        Returns:
            CourseTimeConstraint: A: (time slots x features domain), b: (time slots x 1)
        """
        if slot not in features:
            raise AttributeError("features list must contain Slot feature")

        rows = len(slot.times)
        cols = np.prod([len(feature.domain) for feature in features])
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i, tm in enumerate(slot.times):
            items_at_time = [item for item in items if tm in item.value(slot)]
            for item in items_at_time:
                A[i, item.index(features)] = 1
            b[i, 0] = 1

        return LinearConstraint(A, b, features)


class MutualExclusivityConstraint(LinearConstraint):
    @staticmethod
    def from_items(
        items: List[ScheduleItem],
        exclusive_feature: BaseFeature,
        features: List[BaseFeature],
    ):
        """Helper method for creating constraints that prevent scheduling multiple sections of the same class

        Args:
            items (List[ScheduleItem]): Items, possibly having same value for exclusive_feature
            exclusive_feature (BaseFeature): Feature that must remain exclusive
            features (List[BaseFeatures]): Feature list for items

        Raises:
            AttributeError: Features list is required to contain exclusive_feature

        Returns:
            MutualExclusivityConstraint: A: (exclusive_feature domain x features domain), b: (exclusive_feature domain x 1)
        """
        if exclusive_feature not in features:
            raise AttributeError("features list must contain exclusive_feature")

        rows = len(exclusive_feature.domain)
        cols = np.prod([len(feature.domain) for feature in features])
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i, excl in enumerate(exclusive_feature.domain):
            items_for_course = [
                item for item in items if item.value(exclusive_feature) == excl
            ]
            for item in items_for_course:
                A[i, item.index(features)] = 1
            b[i, 0] = 1

        return LinearConstraint(A, b, features)
