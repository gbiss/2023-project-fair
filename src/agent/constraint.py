from typing import List

import numpy as np
from scipy.sparse import dok_array

from .feature import BaseFeature, Course, Section, Slot
from .item import BaseItem, ScheduleItem


def indicator(features: List[BaseFeature], bundle: List[BaseItem]):
    rows = np.prod([len(feature.domain) for feature in features])
    ind = dok_array((rows, 1), dtype=np.int_)
    for item in bundle:
        ind[item.index(features), 0] = True

    return ind


class BaseConstraint:
    pass


class LinearConstraint(BaseConstraint):
    def __init__(self, A: dok_array, b: dok_array, features: List[BaseFeature]):
        self.A = A
        self.b = b
        self.features = features

    def satisfies(self, bundle: List[BaseItem]):
        ind = indicator(self.features, bundle)
        product = self.A @ ind

        # apparently <= is much less efficient than using < and != separately
        less_than = np.prod((product < self.b).toarray().flatten())
        equal_to = not np.prod((product != self.b).toarray().flatten())

        return less_than or equal_to


class CoursePreferrenceConstraint(LinearConstraint):
    @staticmethod
    def from_course_lists(
        preferred_courses: List[List[str]], limits: List[int], course: Course
    ):
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
        domain = slot.domain
        rows = len(slot.domain)
        cols = len(course.domain) * len(slot.domain)
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i, slt in enumerate(slot.domain):
            items_in_slot = [item for item in items if item.value(slot) == slt]
            for item in items_in_slot:
                A[i, item.index([course, slot])] = 1
            b[i, 0] = 1

        return LinearConstraint(A, b, [course, slot])


class CourseSectionConstraint(LinearConstraint):
    @staticmethod
    def one_section_per_course(
        items: List[ScheduleItem], course: Course, section: Section
    ):
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
