from .feature import BaseFeature
from typing import List


class BaseItem():

    def __init__(self, name: str, features: List[BaseFeature]):
        self.name = name
        self.features = features


class Schedule(BaseItem):

    def __init__(self, features: List[BaseFeature]):
        super().__init__("schedule", features)