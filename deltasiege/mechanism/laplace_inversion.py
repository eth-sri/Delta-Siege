import numpy as np
from typing import Dict

from . import LaplaceMechanism, RNGMechanism

class LaplaceInversionMechanism(LaplaceMechanism, RNGMechanism):

    """
    Base class for different implementations of the Laplacian mechansim as given in 
    https://link.springer.com/chapter/10.1007/11681878_14
    noise generation based on the inverse method
    """

    def _init_helper(self, epsilon: float, sensitivity: float, word_size : int = 2**32, **kwargs):
        """
        Create a Gaussian mechanism

        Args:
            epsilon: float
                Epsilon DP parameter
            sensitivity: float
                Sensitivity used to determine the size of the neighbourhood
            word_size: int
                Number of states in the simulated random number generator in the RNGMechanism
            kwargs:
                Key word arguments to the Mechanism parent class
        """
        LaplaceMechanism._init_helper(self, epsilon, sensitivity, **kwargs)
        RNGMechanism.__init__(self, word_size)

    def __call__(self, input : float, n: int) -> np.ndarray:
        """
        Get n samples from the Laplace mechanism

        Args:
            input: float
                The input original data
            n: int
                Number of samples

        Returns:
            : np.ndarray
                Returns a Numpy array with the batches of samples from the Laplace noise mechanism. 
                The shape of the returned arrays is (n, 1)
        """

        # Use inverse sampling
        x = self.cdf_inv_(self.uniform(n))

        # Add sign
        sgn = 2 * (self.uniform(n) < 0.5) - 1
        x = sgn * x

        # Add scaling and translation
        s = input + x * self.std
        return s
        
    def cdf_inv_(self, y) -> np.ndarray:
        """
        Compute inverse of the Laplace CDF

        Args:
            y: np.ndarray
                Input to inverse CDF

        Returns:
            : np.ndarray
                Inverse of CDF
        """
        return -np.log(1 - y) / np.sqrt(2)
