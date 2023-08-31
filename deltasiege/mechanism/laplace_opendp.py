# This code contains elements from https://github.com/opendp/opendp/tree/main
# under the following lisence:
# MIT License
#
# Copyright (c) 2021 President and Fellows of Harvard College
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from opendp.measurements import make_base_laplace
from opendp.mod import enable_features, binary_search_param

from . import LaplaceMechanism

enable_features("contrib")


class LaplaceOpenDPMechanism(LaplaceMechanism):

    """
    Laplace mechanism implemented in the OpenDP library as described in
    https://docs.opendp.org/en/v0.2.0/api/python/opendp.meas.html?highlight=laplace#opendp.meas.make_base_laplace
    """

    def _init_helper(self, epsilon: float, sensitivity: float, **kwargs) -> None:
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

        # Compute the scale parameter by a binary search
        odp_sigma = binary_search_param(
            make_base_laplace, 
            d_in=sensitivity, d_out=epsilon
        )

        # Sample from laplacian noise mechanism
        self.base_gauss = make_base_laplace(scale=odp_sigma, D="VectorDomain<AllDomain<float>>")

        super()._init_helper(epsilon, sensitivity, **kwargs)


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

        # Add Laplacian noise
        return np.array(self.base_gauss([input] * n), dtype=np.float64)
