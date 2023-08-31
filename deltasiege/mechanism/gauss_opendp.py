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

from cmath import isfinite
import numpy as np
from opendp.combinators import make_zCDP_to_approxDP, make_fix_delta
from opendp.measurements import make_base_gaussian
from opendp.mod import enable_features, binary_search_param

from . import GaussMechanism

enable_features("contrib")

class GaussOpenDPMechanism(GaussMechanism):

    """
    Gaussian mechanism implemented in the OpenDP library as described in
    https://docs.opendp.org/en/v0.2.0/api/python/opendp.meas.html?highlight=gaussia#opendp.meas.make_base_gaussian 
    """

    def _init_helper(self, epsilon: float, delta: float, sensitivity: float, **kwargs) -> None:
        """
        Create a Gaussian mechanism

        Args:
            epsilon: float
                Epsilon DP parameter
            delta: float
                Delta DP parameter
            sensitivity: float
                Sensitivity used to determine the size of the neighbourhood
            kwargs:
                Key word arguments to the Mechanism parent class
        """

        # Create helper
        def make_εδ_gaussian(scale):
            return make_fix_delta(make_zCDP_to_approxDP(make_base_gaussian(scale)), delta=delta)

        scale = binary_search_param(make_εδ_gaussian, d_in=sensitivity, d_out=(epsilon, delta))
        self._gauss = make_base_gaussian(scale=scale)

        # Make curve for changing delta
        meas_approxDP = make_zCDP_to_approxDP(self._gauss)
        self.curve = meas_approxDP.map(d_in=sensitivity)

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
        return np.array([ self._gauss(input) for _ in range(n) ], dtype=np.float64)
    
    def guarantee_(self, epsilon : float, delta : float) -> float:
        """
        A mapping of (epsilon, delta) to a parameter rho, which uniquely specifies the privacy level
        Is non-increasing in both epsilon and delta. 
        Defined as the scale parameter of the underlying mechanism


        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        
        Returns:
            : float
                The scale parameter of the underlying mechanism
        """

        # Ensure that guarantee is only given for valid (\epsilon, \delta)-pairs
        if epsilon <= 0 or delta <= 0:
            return float("inf")

        if delta >= 1.0 or not isfinite(epsilon):
            return -float("inf")

        def make_εδ_gaussian(scale):
            return make_fix_delta(make_zCDP_to_approxDP(make_base_gaussian(scale)), delta=delta)
        
        # Use a binary search to find the scale correponding to the epsilon, delta pair
        scale = binary_search_param(make_εδ_gaussian, d_in=self.sensitivity, d_out=(epsilon, delta))

        return scale

    def perturb_delta(self, new_delta: float) -> float:
        """
        Returns a epsilon value such that self.guarantee(epsilon, new_delta) == self.g0

        Args:
            new_delta: float
                New delta DP parameter
        
        Returns:
            : float
                Value epsilon such that self.guarantee(epsilon, new_delta) == self.g0
                If invalid new_delta return None
        """
        
        # Translate along the fixed cure
        def f():
            if 0 < new_delta < 1:
                return self.curve.epsilon(delta=new_delta)
            return None
        
        # Use the caching structure
        return self.get_value(f"delta/{new_delta}", f)

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
        """
        return 0 < delta < 1 and 0 < epsilon
