# This code contains parts of code found from https://github.com/eth-sri/dp-sniper
# under the following lisence:
#
# MIT License
#
# Copyright (c) 2021 SRI Lab, ETH Zurich
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
from typing import Any, Dict, Tuple

from . import WitnessStatic
from .. import Estimator
from ... import Config, Mechanism, StableClassifier

class WitnessDPSniper(WitnessStatic):

    """
    Witness which implements the attack methodology followed in DP-Sniper
    https://www.sri.inf.ethz.ch/publications/bichsel2021dpsniper
    """

    def _init_helper(
        self, 
        a1 : Any, a2 : Any, 
        classifier : StableClassifier, config : Config, 
        mechanism : Mechanism, estimator : Estimator,
        fixed_delta : float = 0, a1_portion : float = 1.0, c: float = 0.01
    ) -> None:
        """
        Args:
            a1: Any
                The input corresponding to the first input
                Is neighbouring to a2 under the studied mechanism
            a2: Any
                The input corresponding to the second input
                Is neighbouring to a1 under the studied mechanism 
            classifier: Module
                A classifier trained to distinguish outputs from $M(a1)$ and $M(a2)$
                Is a pytorch module
            config: Config
                The global configuration
            mechanism: Mechanism
                The mechanism which is being investigated
        """

        # Compute theshold and tie break mass
        self.fixed_delta : float = fixed_delta
        self.a1_portion : float  = a1_portion
        self.c : float  = c

        # Initialize super class
        super()._init_helper(a1, a2, classifier, config, mechanism, estimator)

    def _find_threshold(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select threshold as described in the paper on DP-Sniper
        https://www.sri.inf.ethz.ch/publications/bichsel2021dpsniper

        Returns:
            : np.ndarray
                Threshold parameters (t). Is an Numpy array of shape (-1,)
            : np.ndarray
                Threshold proabnability parameters (q). Is an Numpy array of shape (-1,)
        """

        # ToDo - avoid taking all samples from both a1 and a2
        
        # Get postive samples
        n1 = round(self.a1_portion * self._get_num_samples())
        p1 = np.concatenate([
          p_sample for (p_sample,) in self._get_samples(self.a1)
        ])[:n1]

        # Get negative samples
        n2 = self._get_num_samples() - n1
        p2 = np.concatenate([
            p_sample for (p_sample,) in self._get_samples(self.a2)
        ])[:n2]

        # Combine to one vector
        p = np.concatenate([p1, p2])
        
        # Uniform samples from a
        quantiles = np.array([1 - self.c - self.fixed_delta])
        thresh = np.quantile(p, quantiles)

        # Find number of samples strictly above thresh
        target = (1 - quantiles) * self._get_num_samples()
        n_above = np.sum(p.reshape(-1, 1) > thresh, axis=0)

        # Find number of samples equal to thresh
        # Equality by being epsilon close
        n_equal = np.sum(np.isclose(p.reshape(-1, 1), thresh), axis=0)
        
        # Split remaining weight
        q = np.divide(
            target - n_above, n_equal, 
            out=np.zeros_like(target), where = n_equal != 0
        )
        q = np.clip(q, a_min = 0, a_max = 1)

        return thresh, q
