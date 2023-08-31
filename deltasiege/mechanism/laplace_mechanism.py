import numpy as np
from typing import Dict

from . import Mechanism

class LaplaceMechanism(Mechanism):

    """
    Base class for different implementations of the Laplacian mechansim as given in 
    https://link.springer.com/chapter/10.1007/11681878_14
    """

    def _init_helper(self, epsilon: float, sensitivity : float, **kwargs) -> None:
        """
        Create a Laplace mechanism

        Args:
            epsilon: float
                Epsilon DP parameter
            sensitivity: float
                Sensitivity used to determine the size of the neighbourhood
            kwargs:
                Key word arguments to the Mechanism parent class
        """
        
        # Set most important parameters
        self.sensitivity : float = sensitivity
        self.std : float = self.sensitivity * np.sqrt(2) / epsilon
        kwargs["delta"] = 0.0

        Mechanism._init_helper(self, epsilon=epsilon, **kwargs)

    def constraint(self, epsilon: float, delta: float) -> bool:
        """
        Returns if epsilon and delta are valid DP parameters for the mechanism

        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        
        Returns:
            : bool
                Indicates if epsilon and delta are valid DP parameters for the mechanism
                Only needs that epsilon is positive
        """
        return epsilon > 0
    
    def guarantee_(self, epsilon : float, delta : float) -> float:
        """
        A wrapper for the mapping of (epsilon, delta) to a parameter rho, which uniquely specifies the privacy level.
        Should be non-increasing in both epsilon and delta for any valid constraints.
        Defined as 1 / epsilon as epsilon is the unique DP parameter

        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        
        Returns:
            : float
                1 / epsilon defines the privacy level uniquely
        """

        # Compute the privacy parameter
        if epsilon > 0:
            return 1 / epsilon
        else:
            return float("inf")

    def perturb_delta(self, new_delta: float) -> float:
        """
        Returns a epsilon value such that self.guarantee(epsilon, new_delta) == self.g0

        Args:
            new_delta: float
                New delta DP parameter
        
        Returns:
            : float
                As epsilon is the only DP parameter, the only valid delta value for permutation is 0
        """
        
        if new_delta > 0:
            return None
        else:
            return self.epsilon
