import numpy as np

class RNGMechanism:

    """
    Helper class for mechanisms using a pseudo random number generator
    """

    def __init__(self, word_size : int = 2**32) -> None:
        """
        Create a RNG mechanism

        Args:
            word_size: int
                Number of states in the simulated random number generator
        """
        self.word_size = word_size

    def uniform(self, n : int) -> np.ndarray:
        """
        Uniformly a float from {k / self.word_size | k \in {0, ..., self.word_size - 1}}

        Args:
            n: int
                Number of samples
        """
        return np.random.randint(0, self.word_size, n) / self.word_size
