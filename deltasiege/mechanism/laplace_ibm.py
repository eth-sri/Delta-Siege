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

from diffprivlib.mechanisms import Laplace
import numpy as np

from . import LaplaceMechanism


class LaplaceIBMMechanism(LaplaceMechanism):

    """
    Gaussian mechanism implemented in the OpenDP library as described in
    https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/laplace.py#L29-L149
    """


    def _init_helper(self, epsilon: float, sensitivity : float, seed : int = None, **kwargs) -> None:
        """
        Create a Gaussian mechanism

        Args:
            epsilon: float
                Epsilon DP parameter
            sensitivity: float
                Sensitivity used to determine the size of the neighbourhood
            kwargs:
                Key word arguments to the Mechanism parent class
        """

        # Helper class for adding Laplace noise
        self.laplace_ibm = Laplace(epsilon=epsilon, sensitivity=sensitivity, random_state=seed)

        # Initialize super class
        super()._init_helper(epsilon=epsilon, sensitivity=sensitivity, **kwargs)

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
        return np.array([self.laplace_ibm.randomise(input) for _ in range(n)], dtype=np.float64)
