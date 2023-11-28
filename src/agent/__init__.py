from typing import List

from .item import BaseItem
from .valuation import BaseValuation


def exchange_contribution(
    valuation: BaseValuation,
    bundle: List[BaseItem],
    og_item: BaseItem,
    new_item: BaseItem,
):
    """
    Determine whether the agent can exchange original_item for new_item and keep the same utility
    """
    og_val = valuation.value(bundle)
    print("og_value", og_val)

    for i in range(len(bundle)):
        if bundle[i] == new_item:
            print(bundle[i], new_item)
            return False

    T0 = bundle.copy()
    index = []
    for i in range(len(T0)):
        if T0[i] == og_item:
            index.append(i)
    if len(index) == 0:
        return False

    T0.pop(index[0])
    T0.append(new_item)

    new_val = valuation.value(T0)
    print("new_val", new_val)
    if og_item == new_item:
        return False
    if og_val == new_val:
        return True
    else:
        return False


def marginal_contribution(
    valuation: BaseValuation, bundle: List[BaseItem], item: BaseItem
):
    """
    Compute the marginal utility the agent gets form adding a particular item to a particular bundle of items
    """

    T = bundle.copy()
    current_val = valuation.value(T)
    T.append(item)
    new_val = valuation.value(T)

    return new_val - current_val


class BaseAgent:
    def __init__(self, valuation: BaseValuation):
        self.valuation

    def valuation(self, bundle: List[BaseItem]):
        return self.valuation.value(bundle)
