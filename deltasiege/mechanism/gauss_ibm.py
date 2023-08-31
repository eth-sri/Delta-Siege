# This code contains elements from https://github.com/IBM/differential-privacy-library/tree/main
# under the following lisence:
# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from diffprivlib.mechanisms import Gaussian, GaussianAnalytic, GaussianDiscrete
import numpy as np
from typing import Callable

from . import GaussMechanismBlackBox, GaussMechanismClassic


class GaussIBMMechanism(GaussMechanismClassic):

    """
    Gaussian mechanism implemented in the OpenDP library as described in
    https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/gaussian.py#L32-L117
    """

    def _init_helper(self, epsilon: float, delta: float, sensitivity: float, seed : int = None, **kwargs) -> None:
        """        
        Create a Gaussian mechanism

        Args:
            epsilon: float
                Epsilon DP parameter
            delta: float
                Delta DP parameter
            sensitivity: float
                Sensitivity used to determine the size of the neighbourhood
            seed: int
                Seed of the mechanism
            kwargs:
                Key word arguments to the Mechanism parent class
        """
        self._gauss = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=seed)
        super()._init_helper(epsilon, delta, sensitivity, **kwargs)

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
                The shape of the returned arrays is (n, 1)
        """

        # Return array
        return np.array([
            self._gauss.randomise(input) for _ in range(n)
            ], 
            dtype=np.float64
        )


class GaussIBMAnalyticalMechanism(GaussMechanismBlackBox):

    """
    Gaussian mechanism implemented in the OpenDP library as described in
    https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/gaussian.py#L120-L206
    """

    def _init_helper(self, epsilon: float, delta: float, sensitivity: float, seed : int = None, **kwargs) -> None:
        """        
        Create a Gaussian mechanism

        Args:
            epsilon: float
                Epsilon DP parameter
            delta: float
                Delta DP parameter
            sensitivity: float
                Sensitivity used to determine the size of the neighbourhood
            seed: int
                Seed of the mechanism
            kwargs:
                Key word arguments to the Mechanism parent class
        """
        self.seed : int = seed
        super()._init_helper(epsilon, delta, sensitivity, **kwargs)

    def _helper_class(self, epsilon: float, delta: float) -> Callable:
        """
        Initialize the black box gaussian mechanism

        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        
        Returns:
            : Callable
                The underlying Gaussian noise mechanism
        """
        return GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=self.sensitivity, random_state=self.seed)
    
    @property
    def _upper_eps(self) -> float:
        """
        Upper bound on the allowed epsilon
        """
        return float("inf")
    
    @property
    def _scaling_name(self) -> str:
        """
        Name of the standard deviation in the underlying mechanism
        """
        return "_scale"
    
    @property
    def _randomize_name(self) -> str:
        """
        Name of the method for randomizing in the underlying mechanism
        """
        return "randomise"


class GaussIBMDiscreteMechanism(GaussMechanismBlackBox):

    """
    Gaussian mechanism implemented in the OpenDP library as described in
    https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/gaussian.py#L209-L362
    """

    def _init_helper(self, epsilon: float, delta: float, sensitivity: float, seed : int = None, **kwargs) -> None:
        """        
        Create a Gaussian mechanism

        Args:
            epsilon: float
                Epsilon DP parameter
            delta: float
                Delta DP parameter
            sensitivity: float
                Sensitivity used to determine the size of the neighbourhood
            seed: int
                Seed of the mechanism
            kwargs:
                Key word arguments to the Mechanism parent class
        """
        self.seed : int = seed
        super()._init_helper(epsilon, delta, sensitivity, **kwargs)

    def _helper_class(self, epsilon: float, delta: float) -> Callable:
        """
        Initialize the black box gaussian mechanism

        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        
        Returns:
            : Callable
                The underlying Gaussian noise mechanism
        """
        return GaussianDiscrete(epsilon=epsilon, delta=delta, sensitivity=self.sensitivity, random_state=self.seed)
    
    @property
    def _upper_eps(self) -> float:
        """
        Upper bound on the allowed epsilon
        """
        return float("inf")
    
    @property
    def _scaling_name(self) -> str:
        """
        Name of the standard deviation in the underlying mechanism
        """
        return "_scale"
    
    @property
    def _randomize_name(self) -> str:
        """
        Name of the method for randomizing in the underlying mechanism
        """
        return "randomise"
    
    @property
    def _dtype(self) -> np.number:
        """
        Dtype used by the mechanism - is integer as it is a descrete mechanism
        """
        return np.int64
