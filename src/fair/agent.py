from typing import List

from fair.constraint import PreferenceConstraint

from .item import BaseItem
from .valuation import RankValuation, UniqueItemsValuation


def exchange_contribution(
    valuation: RankValuation,
    bundle: List[BaseItem],
    og_item: BaseItem,
    new_item: BaseItem,
):
    """Check for improvement in utility

    Determine whether the agent can exchange original_item for new_item and keep the same utility


    Args:
        valuation (BaseValuation): Valuation object to be used for comparison
        bundle (List[BaseItem]): Original set of items
        og_item (BaseItem): Item to be removed
        new_item (BaseItem): Item to be added

    Returns:
        bool: True if utility can be improved; False otherwise
    """
    og_val = valuation.value(bundle)

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
    if og_item == new_item:
        return False
    if og_val == new_val:
        return True
    else:
        return False


def marginal_contribution(
    valuation: RankValuation, bundle: List[BaseItem], item: BaseItem
):
    """Marginal change in utility

    Compute the marginal utility the agent gets form adding a particular item to a particular bundle of items


    Args:
        valuation (BaseValuation): Valuation object to be used for computing utility
        bundle (List[BaseItem]): Initial set of items
        item (BaseItem): Item to be added

    Returns:
        Any: Change in value
    """
    T = bundle.copy()
    current_val = valuation.value(T)
    T.append(item)
    new_val = valuation.value(T)

    return new_val - current_val


class BaseAgent:
    """A wrapper class for apply a valuation to bundles of items"""

    def __init__(self, valuation: RankValuation):
        """
        Args:
            valuation (BaseValuation): Valuation object to apply to bundles
        """
        self.valuation = valuation

    def value(self, bundle: List[BaseItem]):
        """Apply valuation to bundle

        Args:
            bundle (List[BaseItem]): Items to evaluate

        Returns:
            Any: Value of bundle
        """
        return self.valuation.value(bundle)


class Student(BaseAgent):
    """A student agent"""

    def __init__(self, valuation: RankValuation):
        super().__init__(valuation)


class LegacyStudent:
    """A student compatible with https://github.com/cheerstopaula/Allocation"""

    def __init__(
        self, student: BaseAgent, all_courses_constraint: PreferenceConstraint
    ):
        """
        Raises:
            AttributeError: student must include valuation as member
            AttributeError: student.valuation.constraints must include all_courses_constraint

        Args:
            student (BaseAgent): Student to delegate value queries to
            all_courses_constraint (PreferenceConstraint): All desirable courses
        """
        if not hasattr(student, "valuation"):
            raise AttributeError("student delegate must have valuation member")

        if all_courses_constraint not in student.valuation.constraints:
            raise AttributeError("student delegate must include all_courses_constraint")

        student.valuation = UniqueItemsValuation(student.valuation)
        self.student = student
        self.all_courses_constraint = all_courses_constraint

    def valuation(self, bundle: List[BaseItem]):
        """Delegate to value function
        Args:
            bundle (List[BaseItem]): Items to evaluate
        """
        return self.student.value(bundle)

    def marginal_contribution(self, bundle: List[BaseItem], item: BaseItem):
        """Delegate to marginal_contribution function

        Args:
            bundle (List[BaseItem]): Initial set of items
            item (BaseItem): Item to be added
        """
        return marginal_contribution(self.student.valuation, bundle, item)

    def exchange_contribution(
        self, bundle: List[BaseItem], og_item: BaseItem, new_item: BaseItem
    ):
        """Delegate to exchange_contribution function

        Args:
            bundle (List[BaseItem]): Initial set of items
            og_item (BaseItem): Item to be removed
            new_item (BaseItem): Item to be added
        """
        exchange_contribution(self.student.valuation, bundle, og_item, new_item)

    def get_desired_items_indexes(self, items: List[BaseItem]):
        """Return subset of indices from items that are preferred by the student

        This method will currently only work if the delegate student object implements a
        ConstraintSatifactionValuation valuation, which itself includes a PreferrenceConstraint

        Args:
            items (List[BaseItem]): Candidate items list

        Returns:
            List[int]: Indices of desired items in list
        """
        desired_items = self.all_courses_constraint.constrained_items(items)

        return [i for i in range(len(items)) if items[i] in desired_items]
