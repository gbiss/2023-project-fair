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


class CoursePreferrenceConstraint(LinearConstraint):
    @staticmethod
    def from_course_lists(
        preferred_courses: List[List[str]], limits: List[int], course: Course
    ):
        """A helper method for constructing a course preference constraint

        This constraint ensures that the bundle contains a limited number of pre-selected courses from
        each provided topic (e.g. Physics, Chemistry, etc.)

        Args:
            preferred_courses (List[List[str]]): Each list is a topic and items in that list are courses
            limits (List[int]): The maximum number of courses desired per topic
            course (Course): Feature to be used for courses

        Raises:
            IndexError: Number of topics must match among preferred_courses and limits

        Returns:
            CoursePreferrenceConstraint: A: (topics x features domain), b: (topics x 1)
        """
        if len(preferred_courses) != len(limits):
            raise IndexError("item and limit lists must have the same length")

        constraint_ct = len(preferred_courses)
        domain = course.domain
        A = dok_array((constraint_ct, len(domain)), dtype=np.int_)
        b = dok_array((constraint_ct, 1), dtype=np.int_)

        for i in range(constraint_ct):
            for j in range(len(preferred_courses[i])):
                A[
                    i,
                    domain.index(preferred_courses[i][j]),
                ] = 1
            b[i, 0] = limits[i]

        return LinearConstraint(A, b, [course])

    def __init__(self, A: dok_array, b: dok_array, features: List[BaseFeature]):
        super().__init__(A, b, features)


class CourseTimeConstraint(LinearConstraint):
    @staticmethod
    def mutually_exclusive_slots(items: List[ScheduleItem], course: Course, slot: Slot):
        """Helper method for creating constraints that prevent course time overlap

        A bundle satisfies this constraint only if no two courses meet at the same time.

        Args:
            items (List[ScheduleItem]): Possibly time-conflicting items
            course (Course): Feature for course
            slot (Slot): Feature for time slots

        Returns:
            CourseTimeConstraint: A: (time slots x course domain), b: (time slots x 1)
        """
        rows = len(slot.times)
        cols = len(course.domain) * len(slot.domain)
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i, tm in enumerate(slot.times):
            items_at_time = [item for item in items if tm in item.value(slot)]
            for item in items_at_time:
                A[i, item.index([course, slot])] = 1
            b[i, 0] = 1

        return LinearConstraint(A, b, [course, slot])


class CourseSectionConstraint(LinearConstraint):
    @staticmethod
    def one_section_per_course(
        items: List[ScheduleItem], course: Course, section: Section
    ):
        """Helper method for creating constraints that prevent scheduling multiple sections of the same class

        Args:
            items (List[ScheduleItem]): Items, possibly from the same course
            course (Course): Feature for course
            section (Section): Feature for section

        Returns:
            CourseSectionConstraint: A: (course domain x features domain), b: (course domain x 1)
        """
        rows = len(course.domain)
        cols = len(course.domain) * len(section.domain)
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i, crs in enumerate(course.domain):
            items_for_course = [item for item in items if item.value(course) == crs]
            for item in items_for_course:
                A[i, item.index([course, section])] = 1
            b[i, 0] = 1

        return LinearConstraint(A, b, [course, section])
