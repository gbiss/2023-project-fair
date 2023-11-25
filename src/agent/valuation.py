from typing import List

from agent.item import BaseItem

from .constraint import BaseConstraint


class BaseValuation:
    pass


class ConstraintSatifactionValuation(BaseValuation):
    def __init__(self, constraints: List[BaseConstraint], memoize: bool = True):
        self.constraints = constraints
        self.memoize = memoize
        self.marginal_memo = {}
        self.value_memo = {}

    def marginal(self, bundle: List[BaseItem]):
        hashable_bundle = tuple(sorted(bundle))

        if hashable_bundle in self.marginal_memo:
            return self.marginal_memo[hashable_bundle]

        satisfies = True
        for constraint in self.constraints:
            satisfies *= constraint.satisfies(bundle)
        self.marginal_memo[hashable_bundle] = satisfies

        return satisfies

    def value(self, bundle: List[BaseItem]):
        hashable_bundle = tuple(sorted(bundle))

        if hashable_bundle in self.value_memo:
            return self.value_memo[hashable_bundle]

        if self.marginal(bundle) == 1:
            self.value_memo[hashable_bundle] = len(bundle)
            return len(bundle)

        submarginal = 0
        for i in range(len(bundle)):
            subbundle = bundle[:i] + bundle[i + 1 :]
            submarginal = max(submarginal, self.marginal(subbundle))
        self.value_memo[hashable_bundle] = submarginal

        return submarginal
