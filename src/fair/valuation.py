from copy import deepcopy
from typing import List

from fair.item import BaseItem

from .constraint import BaseConstraint


class BaseValuation:
    """An agent's utility associated with bundles of items"""


class RankValuation(BaseValuation):
    """Matroid rank utility associated with bundles of items"""

    def independent(self, bundle: List[BaseItem]):
        """Does the bundle receive maximal value
        ]
                Args:
                    bundle (List[BaseItem]): Items in the bundle

                Raises:
                    NotImplemented: Must be implemented by child class
        """
        raise NotImplemented

    def value(self, bundle: List[BaseItem]):
        """Value of bundle

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Raises:
            NotImplemented: Must be implemented by child class
        """
        raise NotImplemented


class MemoableValuation:
    """A mixin that caches intermediate results"""

    def __init__(self, constraints: List[BaseConstraint]):
        """
        Args:
            constraints (List[BaseConstraint]): Constraints that limit independence
        """
        self.constraints = constraints
        self._independent_memo = {}
        self._value_memo = {}
        self._independent_ct = 0
        self._unique_independent_ct = 0
        self._value_ct = 0
        self._unique_value_ct = 0

    def _independent(self, bundle: List[BaseItem]):
        """Actual calculation of bundle independence

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Raises:
            NotImplementedError: Must be implemented by child class
        """
        raise NotImplementedError

    def independent(self, bundle: List[BaseItem]):
        """Does the bundle receive maximal value

        Retreives cached value if present, otherwise it calculates it

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Returns:
            bool: True if bundle receives maximal value; False otherwise
        """
        hashable_bundle = tuple(sorted(bundle))

        self._independent_ct += 1
        if hashable_bundle not in self._independent_memo:
            self._independent_memo[hashable_bundle] = self._independent(bundle)
            self._unique_independent_ct += 1

        return self._independent_memo[hashable_bundle]

    def _value(self, bundle: List[BaseItem], bottom_up: bool = True):
        """Actual implementation of value function

        Args:
            bundle (List[BaseItem]): Items in the bundle
            bottom_up (bool, optional): Method to test for indepenence

        Raises:
            NotImplementedError: Must be implemented by child class
        """
        raise NotImplementedError

    def value(self, bundle: List[BaseItem], bottom_up: bool = True):
        """Value of bundle

        Retreives cached value if present, otherwise it calculates it

        Args:
            bundle (List[BaseItem]): Items in the bundle
            bottom_up (bool, optional): Method to test for indepenence

        Returns:
            int: Bundle value
        """
        hashable_bundle = tuple(sorted(bundle))

        self._value_ct += 1
        if hashable_bundle not in self._value_memo:
            self._value_memo[hashable_bundle] = self._value(bundle, bottom_up)
            self._unique_value_ct += 1

        return self._value_memo[hashable_bundle]


class ConstraintSatifactionValuation(MemoableValuation):
    """Valuation that limits independence with constraints"""

    def __init__(self, constraints: List[BaseConstraint]):
        """
        Args:
            constraints (List[BaseConstraint]): Constraints that limit independence
        """
        super().__init__(constraints)

    def _independent(self, bundle: List[BaseItem]):
        """Does the bundle receive maximal value

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Returns:
            bool: True if bundle receives maximal value; False otherwise
        """
        satisfies = True
        for constraint in self.constraints:
            satisfies *= constraint.satisfies(bundle)

        return satisfies

    def _value(self, bundle: List[BaseItem], bottom_up: bool = True):
        """Value of bundle

        The value is the size of the largest independent set contained in the bundle

        Args:
            bundle (List[BaseItem]): Items in the bundle
            bottom_up (bool, optional): Method to test for indepenence

        Returns:
            int: Bundle value
        """
        if self.independent(bundle):
            return len(bundle)

        if bottom_up:
            return self._bottom_up_value(bundle)
        else:
            return self._top_down_value(bundle)

    def _top_down_value(self, bundle: List[BaseItem]):
        """Value of bundle

        The value is the size of the largest independent set contained in the bundle.
        Recursively analyze all possible subbundles beginning with the largest.

        Args:
            bundle (List[BaseItem]): Items in the bundle
            bottom_up (bool, optional): Method to test for indepenence

        Returns:
            int: Bundle value
        """
        value = 0
        for i in range(len(bundle)):
            subbundle = bundle[:i] + bundle[i + 1 :]
            value = max(value, self.value(subbundle, bottom_up=False))

        return value

    def _bottom_up_value(self, bundle: List[BaseItem]):
        """Value of bundle

        The value is the size of the largest independent set contained in the bundle.
        Grow the independent set from the bottom up, using the augmentation property.

        Args:
            bundle (List[BaseItem]): Items in the bundle
            bottom_up (bool, optional): Method to test for indepenence

        Returns:
            int: Bundle value
        """
        bundle = list(deepcopy(bundle))
        indep = []
        while len(bundle) > 0:
            cand = bundle.pop()
            if self.independent(indep + [cand]):
                indep.append(cand)

        return len(indep)

    def compile(self):
        """Compile constraints list into single constraint

        Returns:
            ConstraintSatifactionValuation: Valuation with constraints compiled
        """
        constraints = deepcopy(self.constraints)
        if len(constraints) == 0:
            return self

        constraint = constraints.pop()
        while len(constraints) > 0:
            constraint += constraints.pop()

        return ConstraintSatifactionValuation([constraint.prune()])


class UniqueItemsValuation:
    """An adapter that discards duplicate items before calculating independence and value"""

    def __init__(self, valuation: MemoableValuation):
        """
        Args:
            valuation (RankValuation): Underlying rank-based valuation
        """
        self.valuation = valuation

    def __getattr__(self, name):
        if name == "independent":
            return self.independent
        elif name == "value":
            return self.value
        else:
            return getattr(self.valuation, name)

    def independent(self, bundle: List[BaseItem]):
        """Do the unique items in this bundle receive maximal value

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Returns:
            bool: True if bundle receives maximal value; False otherwise
        """
        return self.valuation.independent(list(set(bundle)))

    def value(self, bundle: List[BaseItem]):
        """Value of unique items in bundle

        The value is the size of the largest independent set contained in the bundle

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Returns:
            int: Bundle value
        """
        return self.valuation.value(list(set(bundle)))


class StudentValuation(ConstraintSatifactionValuation):
    """A constraint-based student valuation"""

    def __init__(self, constraints: List[BaseConstraint]):
        """
        Args:
            constraints (List[BaseConstraint]): Constraints that limit independence
        """
        super().__init__(constraints)
