import numpy as np
from typing import Any, Dict, Tuple

from .binomial_bound import binomial_lcb, binomial_ucb
from . import WitnessStatic
from .. import Estimator
from ... import Config, Mechanism, StableClassifier


class WitnessOptimization(WitnessStatic):

    """
    An implementation of the Witness which optimize the threshold parameters (t, q)
    """
    
    def _init_helper(
        self, 
        a1 : Any, a2 : Any, 
        classifier : StableClassifier, config : Config, 
        mechanism : Mechanism,
        estimator : Estimator, thresh_step  : float = 0.01, thresh_confidence : float = 0.9
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

        # Set parameters
        self.thresh_step : float = thresh_step
        self.thresh_confidence : float = thresh_confidence

        # Initialize super class
        super()._init_helper(a1, a2, classifier, config, mechanism, estimator)

    def _find_threshold(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select threshold which maximizes the performance of the witness according 
        to a confident estimate

        Returns:
            : np.ndarray
                Threshold parameters (t). Is an Numpy array of shape (-1,)
            : np.ndarray
                Threshold proabnability parameters (q). Is an Numpy array of shape (-1,)
        """
        
        # To Do: Make more reproducable
        sorted_probs_1 = np.concatenate([p for (p,) in self._get_samples(self.a1)], axis=0)
        sorted_probs_2 = np.concatenate([p for (p,) in self._get_samples(self.a2)], axis=0)
        sorted_probs_1[::-1].sort()
        sorted_probs_2[::-1].sort()
        
        # Compute thesholds
        if self.thresh_step is not None:
            step = max(int(sorted_probs_1.size * self.thresh_step), 1)
        else:
            step = 1
        t = np.concatenate([sorted_probs_1[::step], sorted_probs_2[::step]])

        # Compute confident thesholds using union bound
        confidence = (1 - self.thresh_confidence) / 2

        # Get samples size
        num_samples = self._get_num_samples()

        # Lower bound on p1
        count1 = np.count_nonzero(sorted_probs_1.reshape((-1, 1)) > t, axis=0)
        p1_c = np.array([binomial_lcb(num_samples, c1, confidence) for c1 in count1])

        # Upper bound on p2
        count2 = np.count_nonzero(sorted_probs_2.reshape((-1, 1)) > t, axis=0)
        p2_c = np.array([binomial_ucb(num_samples, c2, confidence) for c2 in count2])

        with self.logger(f"estimation"):
            _, idx = self.estimator(p1_c, p2_c, self.mechanism, self.logger)

        return np.array([t[idx]]), np.zeros(1)
