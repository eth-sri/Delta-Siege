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

import pydp.algorithms.numerical_mechanisms as num_mech
from typing import Callable

from . import GaussMechanismBlackBox

class GaussPyDPMechanism(GaussMechanismBlackBox):

    """
    Gaussian mechanism implemented in the PyDP library as described in
    https://github.com/OpenMined/PyDP/blob/1a5be58860f4b806a9e861d99f7976cea4106100/src/bindings/PyDP/mechanisms/mechanism.cpp#L109-L146
    """

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
        return num_mech.GaussianMechanism(epsilon, delta, self.sensitivity)
    
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
        return "std"
    
    @property
    def _randomize_name(self) -> str:
        return "add_noise"
