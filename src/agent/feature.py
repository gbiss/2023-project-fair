class DomainError(Exception):
    pass


class FeatureError(Exception):
    pass


class BaseFeature:
    def __init__(self, name: str, domain: list):
        self.name = name
        self.domain = domain

    def __repr__(self):
        return f"{self.name}: [{self.domain[0]} ... {self.domain[-1]}]"

    def __hash__(self):
        return hash(self.name) ^ hash(tuple(self.domain))


class Course(BaseFeature):
    def __init__(self, domain):
        super().__init__("course", domain)


class Slot(BaseFeature):
    def __init__(self, domain):
        super().__init__("slot", domain)


class Section(BaseFeature):
    def __init__(self, domain):
        super().__init__("section", domain)
