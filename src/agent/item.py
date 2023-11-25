from typing import Any, List

from .feature import BaseFeature, DomainError, FeatureError


class BaseItem:
    def __init__(self, name: str, features: List[BaseFeature]):
        self.name = name
        self.features = features

    def value(self, feature: BaseFeature):
        try:
            return self.values[self.features.index(feature)]
        except IndexError:
            raise FeatureError("feature unknown for this item")

    def index(self, feature: BaseFeature, value: Any):
        try:
            return feature.domain.index(value)
        except IndexError:
            raise DomainError(f"invalid value '{value}' for feature '{feature}'")

    def __repr__(self):
        return f"{self.name}: {[feature for feature in self.features]}"

    def __hash__(self):
        features_hash = sum([hash(feature) for feature in self.features])

        return hash(self.name) + features_hash


class ScheduleItem(BaseItem):
    def __init__(self, features: List[BaseFeature], values: List[Any]):
        super().__init__("schedule", features)
        self.values = values

        # validate cardinality
        if len(self.values) != len(self.features):
            raise FeatureError("values must correspond to features 1:1")

        # validate domain
        for feature, value in zip(self.features, self.values):
            if value not in feature.domain:
                raise DomainError(f"invalid value for feature '{feature}'")
