from typing import List

from agent.item import BaseItem

from .constraint import BaseConstraint


class BaseValuation:
    pass


class MemoableValuation:
    def __init__(self, constraints: List[BaseConstraint]):
        self.constraints = constraints
        self._marginal_memo = {}
        self._value_memo = {}

    def _marginal(self, bundle: List[BaseItem]):
        raise NotImplementedError

    def marginal(self, bundle: List[BaseItem]):
        hashable_bundle = tuple(sorted(bundle))

        if hashable_bundle not in self._marginal_memo:
            self._marginal_memo[hashable_bundle] = self._marginal(bundle)

        return self._marginal_memo[hashable_bundle]

    def _value(self, bundle: List[BaseItem]):
        raise NotImplementedError

    def value(self, bundle: List[BaseItem]):
        hashable_bundle = tuple(sorted(bundle))

        if hashable_bundle not in self._value_memo:
            self._value_memo[hashable_bundle] = self._value(bundle)

        self._value_memo[hashable_bundle]


class MatroidValuation(MemoableValuation):
    def __init__(self, constraints: List[BaseConstraint]):
        super().__init__(constraints)
        self._distortion = {}

    def marginal(self, bundle: List[BaseItem]):
        marginal = self._marginal(bundle)

        if marginal > 0:
            return marginal

        # set bundle + e to 0 for all e in domain - bundle
        hashable_bundle = tuple(sorted(bundle))
        base_distortion = self._distortion[hashable_bundle]
        bundle_values = {item.value for item in bundle}
        for constraint in self.constraints:
            domain_values = set(constraint.feature.domain)
            for value in domain_values - bundle_values:
                hashable_bundle = tuple(sorted(bundle + [value]))
                self._marginal_memo[hashable_bundle] = 0
                self._distortion[hashable_bundle] = base_distortion + 1

        return self._marginal(bundle)

    def distortion(self):
        return max(self._distortion.values())


class ConstraintSatifactionValuation(MatroidValuation):
    def __init__(self, constraints: List[BaseConstraint]):
        super().__init__(constraints)

    def _marginal(self, bundle: List[BaseItem]):
        satisfies = True
        for constraint in self.constraints:
            satisfies *= constraint.satisfies(bundle)

        return satisfies

    def _value(self, bundle: List[BaseItem]):
        submarginal = 0
        for i in range(len(bundle)):
            subbundle = bundle[:i] + bundle[i + 1 :]
            submarginal = max(submarginal, self.marginal(subbundle))

        return submarginal
