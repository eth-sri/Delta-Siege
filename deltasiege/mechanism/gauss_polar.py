import numpy as np
from typing import Dict

from . import GaussMechanismClassic, RNGMechanism

class GaussPolarMechanism(GaussMechanismClassic, RNGMechanism):

    """
    Base class for different implementations of the Gauss mechansim as given in 
    https://www.nowpublishers.com/article/Details/TCS-042
    using the Polar method for generating Gaussian noise as given in
    https://www.jstor.org/stable/2027592
    """

    def _init_helper(
        self, 
        epsilon: float, delta: float, sensitivity: float, 
        word_size : int = 2**32, return_s2 : bool = False, 
        **kwargs
    ) -> None:
        """
        Initializer

        Args:
            epsilon: float
                Epsilon DP parameter
            delta: float
                Delta DP parameter
            sensitivity: float
                Sensitivity used to determine the size of the neighbourhood
            word_size: int
                Number of states in the simulated random number generator in the RNGMechanism
            return_s2: bool
                If True, then both s1 and s2 are returned which enables attacks as described in
                https://arxiv.org/abs/2112.05307 and https://arxiv.org/abs/2107.10138
            kwargs:
                Key word arguments to the Mechanism parent class
        """
        self.return_s2 : bool = return_s2
        RNGMechanism.__init__(self, word_size)
        GaussMechanismClassic._init_helper(self, epsilon, delta, sensitivity, **kwargs)

    def __call__(self, input : float, n: int) -> np.ndarray:
        """
        Get n samples from the Gaussian mechanism

        Args:
            input: float
                The input original data
            n: int
                Number of samples

        Returns:
            : np.ndarray
                Returns a Numpy array with the batches of samples from the Gaussian noise mechanism. 
                The shape of the returned arrays is (n, 1) if self.return_s2 is False or (n, 2) otherwise
        """

        # Get sufficent samples with norm less than one
        batches = []
        size = 0
        while size < n:
            b = self.get_batch(n)
            size += b.shape[0]
            batches.append(b)

        # Concatenate batches
        x = np.concatenate(batches, axis=0)[:n]

        # Compute samples using polar method
        r = np.linalg.norm(x, axis=1).reshape((-1, 1)) ** 2
        scale = np.sqrt(-2 * np.log(r) / r)

        # Get final samples
        s = input + self.std * x * scale

        # Return samples as 2d vector
        # Either just s1 or s1 & s2
        if self.return_s2:
            return s
        else:
            return s[:, 0].reshape((-1, 1))

    def get_batch(self, n : int) -> np.ndarray:
        """
        Get at most n samples sampled from the unit disc

        Args:
            n: int
                Number of samples

        Returns:
            : np.ndarray
                Returns a Numpy array with at most n samples with shape (-1, 1)
                All returned samples have norm in (0, 1)
        """

        x = 2 * self.uniform(2 * n).reshape((-1, 2)) - 1
        norm = np.linalg.norm(x, axis=1)
        return x[(0 < norm) & (norm < 1)]
