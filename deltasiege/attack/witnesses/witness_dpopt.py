# This code contains parts of code found from https://github.com/barryZZJ/dp-opt
# under the following lisence:
#
# MIT License
#
# Copyright (c) 2022 Zejun Zhou
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
import warnings
from typing import Any, Tuple

from . import WitnessStatic
from .. import Estimator
from ... import Config, Mechanism, StableClassifier

# Imports from the DP-Opt library
from dpopt.probability.estimators import PrEstimator, EpsEstimator, ucb, lcb
from dpopt.optimizer.optimizer_generator import OptimizerGenerator
from dpopt.optimizer.optimizers import *

class HelperEpsEstimator(EpsEstimator):
    """
    Helper class to add the delta parameter to DP-Opt
    """

    # Modification of original code - added delta parameter
    def compute_lcb_by_pp(self, t, q, post_probs1, post_probs2, delta = 0) -> float:
        """
        Estimates lower bound on eps(a1, a2, attack) with provided post probability samples.

        Returns:
            lcb: a lower confidence bound for eps
        """
        t = np.around(t, 3)
        p1 = self.pr_estimator.compute_pr_estimate_by_pp(t, q, post_probs1)
        p2 = self.pr_estimator.compute_pr_estimate_by_pp(t, q, post_probs2)
        lcb, p1_lcb, p2_ucb = self._compute_lcb(p1, p2, delta=delta)
        return lcb

    # Modification of original code - added delta parameter
    def _compute_lcb(self, p1, p2, return_probs=True, delta = 0):
        n_samples = self.pr_estimator.n_samples
        # confidence accounts for the fact that two bounds could be incorrect (union bound)
        confidence = 1 - (1 - self.pr_estimator.config.confidence) / 2
        p1_lcb = lcb(n_samples, int(p1 * n_samples), 1 - confidence)
        p2_ucb = ucb(n_samples, int(p2 * n_samples), 1 - confidence)
        if return_probs:
            return self._compute_eps(p1_lcb - delta, p2_ucb), p1_lcb, p2_ucb
        else:
            return self._compute_eps(p1_lcb - delta, p2_ucb)

class WitnessDPOpt(WitnessStatic):

    """
    Witness which implements the attack methodology followed in DP-Opt
    https://link.springer.com/chapter/10.1007/978-3-031-19214-2_34
    """

    def _init_helper(
        self, 
        a1 : Any, a2 : Any, 
        classifier : StableClassifier, config : Config, 
        mechanism : Mechanism, estimator : Estimator,
        fixed_delta : float = 0
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

        # Fix delta level
        self.fixed_delta = fixed_delta

        # Initialize super class
        super()._init_helper(a1, a2, classifier, config, mechanism, estimator)

    def _find_threshold(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select threshold as described in the paper on DP-Opt
        https://link.springer.com/chapter/10.1007/978-3-031-19214-2_34

        Returns:
            : np.ndarray
                Threshold parameters (t). Is an Numpy array of shape (-1,)
            : np.ndarray
                Threshold proabnability parameters (q). Is an Numpy array of shape (-1,)
        """

        # Optimizers from DP-Opt
        t0 = 0.5
        q0 = 0.5

        univariate_optimizers = [
            NelderMead(1, t0),
            COBYLA(1, t0),
            BruteForce(1), DifferentialEvolution(1),
            Bounded(),
            Powell(1, t0),
        ]
        bivariate_optimizers = [
            NelderMead(2, t0, q0),
            DifferentialEvolution(2),
            COBYLA(2, t0, q0),
            Powell(2, t0, q0),
            BruteForce(2)
        ]

        # Compute theshold and tie break mass
        optimizer_generator = \
            OptimizerGenerator(univariate_optimizers + bivariate_optimizers)

        pr_estimator = PrEstimator(self.mechanism, self.config.n_init, self.config)
        eps_estimator = HelperEpsEstimator(pr_estimator)
        
        # To Do: Avoid taking all samples from a1 and a2
        sorted_post_probs1 = np.concatenate([
            p_sample for (p_sample,) in self._get_samples(self.a1)
        ])
        sorted_post_probs2 = np.concatenate([
            p_sample for (p_sample,) in self._get_samples(self.a2)
        ])
        
        # Sort samples
        sorted_post_probs1 = np.sort(sorted_post_probs1)
        sorted_post_probs2 = np.sort(sorted_post_probs2)

        tmin = min(sorted_post_probs1[0], sorted_post_probs2[0])  # minimum postprob possible
        tmax = sorted_post_probs1[-1]  # maximum postprob possible
        if tmin >= tmax:
            tmin = 0
            tmax = 1
        
        best_t = tmin
        best_q = 0
        best_lcb = -np.inf

        for optimizer in optimizer_generator.get_optimizers():
                warnings.filterwarnings("ignore")
                t, q, lcb = optimizer.maximize(
                    lambda t, q, post_probs1, post_probs2: eps_estimator.compute_lcb_by_pp(
                        t, q, post_probs1, post_probs2, self.fixed_delta
                    ),
                    (sorted_post_probs1, sorted_post_probs2), 
                    tmin, tmax
                )
                warnings.filterwarnings("default")
                if lcb > best_lcb:
                    best_lcb = lcb
                    best_t = np.round(t, 3)
                    best_q = q
        
        return np.array([best_t], dtype=np.float64), np.array([best_q], dtype=np.float64)
