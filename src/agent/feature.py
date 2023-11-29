class DomainError(Exception):
    pass


class FeatureError(Exception):
    pass


class BaseFeature:
    """A named, typed, and ordered space"""

    def __init__(self, name: str, domain: list):
        """
        Args:
            name (str): Feature name
            domain (list): Exhaustive, ordered list of possible feature values
        """
        self.name = name
        self.domain = domain

    def __repr__(self):
        return f"{self.name}: [{self.domain[0]} ... {self.domain[-1]}]"

    def __hash__(self):
        return hash(self.name) ^ hash(tuple(self.domain))


class Course(BaseFeature):
    """Ordered space of courses"""

    def __init__(self, domain):
        """
        Args:
            domain (_type_): Exhaustive, ordered list of possible feature values
        """
        super().__init__("course", domain)


class Slot(BaseFeature):
    """Ordered space of time slots"""

    def __init__(self, domain):
        """
        Args:
            domain (_type_): Exhaustive, ordered list of possible feature values
        """
        super().__init__("slot", domain)


class Section(BaseFeature):
    """Ordered space of course sections"""

    def __init__(self, domain):
        """
        Args:
            domain (_type_): Exhaustive, ordered list of possible feature values
        """
        super().__init__("section", domain)
