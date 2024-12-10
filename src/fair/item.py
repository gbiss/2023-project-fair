from typing import Any, List

import pandas as pd

from .feature import (
    BaseFeature,
    Course,
    DomainError,
    FeatureError,
    Section,
    Slot,
    slot_list,
    slots_for_time_range,
)


class BaseItem:
    """Item defined over multiple features"""

    def __init__(
        self,
        name: str,
        features: List[BaseFeature],
        values: List[Any],
        index: int,
        capacity: int = 1,
    ):
        """
        Args:
            features (List[BaseFeature]): Features revelvant for this item
            values (List[Any]): Value of each feature from its domain
            index (int): Position relative to other items
            capacity (int, optional): Number of times item can be allocated. Defaults to 1.

        Raises:
            FeatureError: Values and features must correspond 1:1
            DomainError: Features can only take values from their domain
        """
        self.name = name
        self.features = features
        self.values = values
        self.index = index
        self.capacity = capacity

        # validate cardinality
        if len(self.values) != len(self.features):
            raise FeatureError("values must correspond to features 1:1")

        # validate domain
        for feature, value in zip(self.features, self.values):
            if value not in feature.domain:
                raise DomainError(f"invalid value for feature '{feature}'")

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

    @staticmethod
    def parse_excel(path: str, frequency: str = "15T"):
        """Read and parse schedule items from excel file

        Args:
            path (str): Full path to excel file

        Returns:
            List[ScheduleItem]: All items that could be extracted from excel file
        """
        with open(path, "rb") as fd:
            df = pd.read_excel(fd)
        df = df[
            df.columns.intersection(
                ["Catalog", "Section", "Mtg Time", "CICScapacity", "Categories"]
            )
        ].dropna()

        course = Course(df["Catalog"].unique())
        section = Section(df["Section"].unique())
        time_slots = slot_list(frequency)
        slot = Slot.from_time_ranges(df["Mtg Time"].unique(), "15T")
        features = [course, section, slot]
        items = []
        for idx, row in df.iterrows():
            values = [
                row["Catalog"],
                row["Section"],
                slots_for_time_range(row["Mtg Time"], time_slots),
            ]
            try:
                items.append(
                    ScheduleItem(
                        features,
                        values,
                        idx,
                        int(row["CICScapacity"]),
                        row["Categories"] if "Categories" in df else None,
                    )
                )
            except DomainError:
                pass

        return items

    def __init__(
        self,
        features: List[BaseFeature],
        values: List[Any],
        index: int,
        capacity: int = 1,
        category: str = None,
    ):
        """An Item appropriate for course scheduling

        Args:
            features (List[BaseFeature]): Features revelvant for this item
            values (List[Any]): Value of each feature from its domain
            index (int): Position relative to other items
            capacity (int): Number of times item can be allocated. Defaults to 1.
            category (str, optional): Topic for course. Defaults to None.
        """
        super().__init__("schedule", features, values, index, capacity)
        self.category = category


def sub_schedule(bundles: list[list[ScheduleItem]]):
    """Given a list of bundles, create a new sub schedule considering the items in the union of all bundles
    Capacities of the new schedule are determined by the sum of the capacities of the items in all bundles.

    Args:
        bundles (list[list[ScheduleItem]]): List of Items from class BaseItem

    Returns:
        new_schedule (list[ScheduleItem]): Items from class BaseItem, new reduced schedule
        course_strings (list[str]): List of course strings of the new schedule
        course (type[Course]): Course instance of the new schedule
    """
    sub_schedule = [item for bundle in bundles for item in bundle]
    set_sub_schedule = sorted(list(set(sub_schedule)), key=lambda item: item.values[0])

    features = sub_schedule[0].features
    course_strings = sorted([item.values[0] for item in set_sub_schedule])

    new_schedule = []
    for i, item in enumerate(set_sub_schedule):
        new_capacity = [
            new_item.capacity for new_item in sub_schedule if new_item == item
        ]
        new_schedule.append(
            ScheduleItem(features, item.values, index=i, capacity=sum(new_capacity))
        )
    return new_schedule, course_strings
