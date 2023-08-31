# This code contains elements from https://github.com/OpenMined/PyDP/tree/dev
# under the following lisence:
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pydp.algorithms.numerical_mechanisms as num_mech

from . import LaplaceMechanism


class LaplacePyDPDPMechanism(LaplaceMechanism):

    """
    Gaussian mechanism implemented in the OpenDP library as described in
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

        # sample laplacian noise
        self._laplace = num_mech.LaplaceMechanism(epsilon, sensitivity)

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
        return np.array([self._laplace.add_noise(input) for _ in range(n)], dtype=np.float64)
