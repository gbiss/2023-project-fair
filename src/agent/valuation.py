from typing import List

from agent.item import BaseItem

from .constraint import BaseConstraint


class BaseValuation:
    pass


class MemoableValuation:
    def __init__(self, constraints: List[BaseConstraint]):
        self.constraints = constraints
        self._independent_memo = {}
        self._value_memo = {}

    def _independent(self, bundle: List[BaseItem]):
        raise NotImplementedError

    def independent(self, bundle: List[BaseItem]):
        hashable_bundle = tuple(sorted(bundle))

        if hashable_bundle not in self._independent_memo:
            self._independent_memo[hashable_bundle] = self._independent(bundle)

        return self._independent_memo[hashable_bundle]

    def _value(self, bundle: List[BaseItem]):
        raise NotImplementedError

    def value(self, bundle: List[BaseItem]):
        hashable_bundle = tuple(sorted(bundle))

        if hashable_bundle not in self._value_memo:
            self._value_memo[hashable_bundle] = self._value(bundle)

        return self._value_memo[hashable_bundle]


class ConstraintSatifactionValuation(MemoableValuation):
    def __init__(self, constraints: List[BaseConstraint]):
        super().__init__(constraints)

    def _independent(self, bundle: List[BaseItem]):
        satisfies = True
        for constraint in self.constraints:
            satisfies *= constraint.satisfies(bundle)

        return satisfies

    def _value(self, bundle: List[BaseItem]):
        if self.independent(bundle):
            return len(bundle)

        value = 0
        for i in range(len(bundle)):
            subbundle = bundle[:i] + bundle[i + 1 :]
            value = max(value, self.value(subbundle))

        return value
