import numpy as np
from typing import Dict

from . import GaussMechanismClassic, RNGMechanism

class GaussBoxMullerMechanism(GaussMechanismClassic, RNGMechanism):

    """
    Base class for different implementations of the Gauss mechansim as given in 
    https://www.nowpublishers.com/article/Details/TCS-042
    using the Box Muller method for generating Gaussian noise as given in
    https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-29/issue-2/A-Note-on-the-Generation-of-Random-Normal-Deviates/10.1214/aoms/1177706645.full
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

        # Use Box-Muller method to create Gaussian Distributed random variable
        r = np.sqrt(-2 * np.log(self.uniform(n)))
        theta = 2 * np.pi * self.uniform(n)

        # Scale and translate samples
        loc = input
        s1 = loc + self.std * r * np.cos(theta)
        s2 = loc + self.std * r * np.sin(theta)

        # Return samples as 2d vector
        # Either just s1 or s1 & s2
        if self.return_s2:
            return np.stack([s1, s2], axis=1)
        else:
            return s1.reshape((-1, 1))
