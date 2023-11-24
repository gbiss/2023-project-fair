from .feature import BaseFeature
from .item import BaseItem
import numpy as np
from scipy.sparse import dok_array
from typing import List


def indicator(feature: BaseFeature, bundle: List[BaseItem]):
    ind = dok_array((len(feature.domain), 1), dtype=np.bool_)
    for item in bundle:
        ind[item.index(feature, item.value(feature)), 0] = 1

    return ind
