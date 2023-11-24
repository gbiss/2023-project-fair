from .feature import BaseFeature, DomainError, FeatureError
from typing import Any, List


class BaseItem:
    def __init__(self, name: str, features: List[BaseFeature]):
        self.name = name
        self.features = features


class Schedule(BaseItem):
    def __init__(self, features: List[BaseFeature], values: List[Any]):
        super().__init__("schedule", features)
        self.values = values

        # validate cardinality
        if len(self.values) != len(self.features):
            raise ValueError("values must correspond to features 1:1")

        # validate domain
        for feature, value in zip(self.features, self.values):
            if value not in feature.domain:
                raise DomainError(f"invalid value for feature {feature}")

    def value(self, feature):
        try:
            return self.values[self.features.index(feature)]
        except IndexError:
            raise FeatureError("feature unknown for this item")
