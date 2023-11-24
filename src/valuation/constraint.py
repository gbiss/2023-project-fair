from .feature import BaseFeature
from .item import BaseItem
import numpy as np
from scipy.sparse import dok_array
from typing import List


def indicator(feature: BaseFeature, bundle: List[BaseItem]):
    ind = dok_array((len(feature.domain), 1), dtype=np.bool_)
    for item in bundle:
        ind[item.index(feature, item.value(feature)), 0] = True

    return ind


class LinearConstraint:
    def __init__(self, A: dok_array, b: dok_array, feature: BaseFeature):
        self.A = A
        self.b = b
        self.feature = feature

    def violates(self, bundle: List[BaseItem]):
        ind = indicator(self.feature, bundle)

        return np.prod((self.A @ ind <= self.b).toarray().flatten())
