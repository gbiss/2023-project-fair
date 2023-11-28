from typing import List

import numpy as np
from scipy.sparse import dok_array

from .feature import BaseFeature
from .item import BaseItem


def indicator(feature: BaseFeature, bundle: List[BaseItem]):
    ind = dok_array((len(feature.domain), 1), dtype=np.int_)
    for item in bundle:
        ind[item.index(feature, item.value(feature)), 0] = True

    return ind


class BaseConstraint:
    pass


class LinearConstraint(BaseConstraint):
    @staticmethod
    def from_lists(
        constraints: List[List[BaseItem]], limits: List[int], feature: BaseFeature
    ):
        if len(constraints) != len(limits):
            raise IndexError("item and limit lists must have the same length")

        constraint_ct = len(constraints)
        domain = feature.domain
        A = dok_array((constraint_ct, len(domain)), dtype=np.int_)
        b = dok_array((constraint_ct, 1), dtype=np.int_)

        for i in range(constraint_ct):
            for j in range(len(constraints[i])):
                A[
                    i,
                    constraints[i][j].index(feature, constraints[i][j].value(feature)),
                ] = 1
            b[i, 0] = limits[i]

        return LinearConstraint(A, b, feature)

    def __init__(self, A: dok_array, b: dok_array, feature: BaseFeature):
        self.A = A
        self.b = b
        self.feature = feature

    def satisfies(self, bundle: List[BaseItem]):
        ind = indicator(self.feature, bundle)
        product = self.A @ ind

        # apparently <= is much less efficient than using < and != separately
        less_than = np.prod((product < self.b).toarray().flatten())
        equal_to = not np.prod((product != self.b).toarray().flatten())

        return less_than or equal_to
