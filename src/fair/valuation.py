from copy import deepcopy
from itertools import chain, combinations
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

    def _value(self, bundle: List[BaseItem]):
        """Actual implementation of value function

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Raises:
            NotImplementedError: Must be implemented by child class
        """
        raise NotImplementedError

    def value(self, bundle: List[BaseItem]):
        """Value of bundle

        Retreives cached value if present, otherwise it calculates it

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Returns:
            int: Bundle value
        """
        hashable_bundle = tuple(sorted(bundle))

        self._value_ct += 1
        if hashable_bundle not in self._value_memo:
            self._value_memo[hashable_bundle] = self._value(bundle)
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

    def _value(self, bundle: List[BaseItem]):
        """Value of bundle

        The value is the size of the largest independent set contained in the bundle

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Returns:
            int: Bundle value
        """
        if self.independent(bundle):
            return len(bundle)

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


def is_mrf(ground, func):
    """checks 4 conditions of valid rank functions

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function

    Returns:
        bool: True if func is an MRF, given ground; False otherwise
    """
    check1 = nonnegative_rank_value(ground, func)
    check2 = rank_value_leq_cardinality(ground, func)
    check3 = is_submodular(ground, func)
    check4 = is_monotonic_non_decreasing(ground, func)

    return check1 and check2 and check3 and check4


def is_submodular(ground, func):
    """performs check for submodularity of func, given a ground set of items

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function

    Returns:
        bool: True if func is submodular, given ground; False otherwise
    """
    powerset_ground = list(powerset(ground))

    for i in range(len(powerset_ground)):
        for j in range(len(powerset_ground)):
            add_val = func(powerset_ground[i]) + func(powerset_ground[j])
            int_val = func(
                list(set(powerset_ground[i]).intersection(set(powerset_ground[j])))
            )
            union_val = func(
                list(set(powerset_ground[i]).union(set(powerset_ground[j])))
            )
            if add_val < int_val + union_val:
                return False

    return True


def powerset(iterable):
    """generates the power set of iterable

    Args:
        iterable (List or Set): ground set of items

    Returns:
        set of tuples: power set
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def is_monotonic_non_decreasing(ground, func):
    """checks that func is monotonic non-decreasing, given the ground set, ground

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function

    Returns:
        bool: True if func is monotonic non_decreasing, given ground; False otherwise
    """
    powerset_ground = list(powerset(ground))

    for i in range(len(powerset_ground)):
        for j in range(len(powerset_ground)):
            subset = False
            if set(powerset_ground[i]).issubset(set(powerset_ground[j])):
                subset = True
            if subset == True and func(powerset_ground[i]) > func(powerset_ground[j]):
                return False

    return True


def nonnegative_rank_value(ground, func):
    """checks that func always returns a non-negative value, given the ground set, ground

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function
    Returns:
        bool: True if func returns non-negative value, given ground; False otherwise
    """
    powerset_ground = list(powerset(ground))

    for i in range(len(powerset_ground)):
        val = func(powerset_ground[i])
        if val < 0:
            return False
    return True


def rank_value_leq_cardinality(ground, func):
    """checks that func always returns a value less than the cardinality

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function

    Returns:
        bool: True if func returns a value less than the cardinality, given ground; False otherwise
    """
    powerset_ground = list(powerset(ground))

    for i in range(len(powerset_ground)):
        val = func(powerset_ground[i])
        cardinality = len(powerset_ground[i])
        if val > cardinality:
            return False
    return True
