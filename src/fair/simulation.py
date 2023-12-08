import numpy as np


class SimulatedAgent:
    """A randomly generated agent"""

    def __init__(self, seed: int | None = None):
        """
        Args:
            seed (int | None, optional): _description_. Defaults to None.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
