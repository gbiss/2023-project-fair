class BaseFeature:
    def __init__(self, name: str, domain: list):
        self.name = name
        self.domain = domain


class Course(BaseFeature):
    def __init__(self, domain):
        super().__init__("course", domain)
