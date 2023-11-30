from typing import Any, List

from .feature import BaseFeature, DomainError, FeatureError


class BaseItem:
    """Item defined over multiple features"""

    def __init__(self, name: str, features: List[BaseFeature]):
        """
        Args:
            name (str): Item name
            features (List[BaseFeature]): Features revelvant for this item
        """
        self.name = name
        self.features = features

    def value(self, feature: BaseFeature):
        """Value associated with a given feature

        Args:
            feature (BaseFeature): Feature for which value is required

        Raises:
            FeatureError: Feature must have been registered during inititialization

        Returns:
            Any: Value for feature
        """
        try:
            return self.values[self.features.index(feature)]
        except IndexError:
            raise FeatureError("feature unknown for this item")

    def index(self, features: List[BaseFeature] = None):
        """Position of item in canonical order

        The domains of features provided as input are ordered according to their cartesian product.
        This method maps the feature values of the present item the associated point in that product.

        Args:
            features (List[BaseFeature], optional): Subset of features from initialization. Defaults to None.

        Raises:
            FeatureError: Features provided must be a subset of those provided during initialization

        Returns:
            Any: Point associated with item in the cartesian product of feature domains
        """
        features = self.features if features is None else features
        mult = 1
        idx = 0
        for feature in features:
            if feature not in self.features:
                raise FeatureError(f"feature {feature} not valid for {self}")
            idx += feature.domain.index(self.value(feature)) * mult
            mult *= len(feature.domain)

        return idx

    def __repr__(self):
        return f"{self.name}: {[self.value(feature) for feature in self.features]}"

    def __hash__(self):
        return hash(self.name) ^ hash(
            tuple([self.value(feature) for feature in self.features])
        )

    def __lt__(self, other):
        return self.__hash__() < hash(other)

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class ScheduleItem(BaseItem):
    """An item representing a class in a schedule"""

    def __init__(self, features: List[BaseFeature], values: List[Any]):
        """
        Args:
            features (List[BaseFeature]): Features revelvant for this item
            values (List[Any]): Value of each feature from its domain

        Raises:
            FeatureError: Values and features must correspond 1:1
            DomainError: Features can only take values from their domain
        """
        super().__init__("schedule", features)
        self.values = values

        # validate cardinality
        if len(self.values) != len(self.features):
            raise FeatureError("values must correspond to features 1:1")

        # validate domain
        for feature, value in zip(self.features, self.values):
            if value not in feature.domain:
                raise DomainError(f"invalid value for feature '{feature}'")
