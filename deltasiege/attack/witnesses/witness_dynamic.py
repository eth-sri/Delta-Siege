import numpy as np
from typing import Any, Dict
import warnings

from . import Witness
from .. import Estimator
from ... import Config, Mechanism, StableClassifier

class WitnessDynamic(Witness):
  
  
    def _init_helper(
        self, 
        a1 : Any, a2 : Any, 
        classifier : StableClassifier, config : Config, 
        mechanism : Mechanism, estimator : Estimator,
        confidence_dwk : float = 0.5, quantiles : np.array = np.linspace(0.75, 1, 25)
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

        # Parameter to select quantiles
        self.quantiles = quantiles
        self.confidence_dwk = confidence_dwk

        # Initialize super class
        super()._init_helper(a1, a2, classifier, config, mechanism, estimator)

    def compute_estimate(self) -> None:
        """
        Computes an estimate of the performance of the current witness
        using the given estimator.
        """

        # Get samples from classifier
        sample_1 = np.concatenate([p for (p,) in self._get_samples(self.a1)])
        sample_2 = np.concatenate([p for (p,) in self._get_samples(self.a2)])

        # Consider uniformly spaced 
        p1, p2 = self._find_mass(sample_1, sample_2, self.quantiles)
        p1_all, p2_all = self._find_mass(sample_1, sample_2, np.linspace(0.5, 1, 500))

        # Add confident estimate
        p1_lcb, p2_ucb = self._add_confidence(p1, p2, p1_all, p2_all, self._get_num_samples())

        # Unconfident estimate
        with self.logger(f"estimation"):
            loss, _ = self.estimator(p1_all, p2_all, self.mechanism, self.logger)
            self.estimate = loss.min().item()
        
        # Confident estimate
        with self.logger(f"estimation_c"):
            loss_c, _ = self.estimator(p1_lcb, p2_ucb, self.mechanism, self.logger)
            self.estimate_c = loss_c.min().item()
            
        if self.estimate >= self.estimate_c and np.isfinite(self.estimate_c):
            warnings.warn(
                f"There might be a soundness issure as " \
                f"estimate={self.estimate} and estimate_c={self.estimate_c}"
            )

        return self.estimate, self.estimate_c

    def _find_mass(self, sample_a, sample_b, quantiles):
        """
        Args: 1d arrays of predicted probabilities 
        Returns: Probability mass thresholds from 
                 sample_a with corresponding mass from sample_b with same threshold
        """

        # Compute corresponding thresholds
        thresh = np.quantile(sample_a, quantiles)

        # Comute corresponding mass in samples from a with thresholds
        sample_a_in_set = sample_a.reshape((1, -1)) >= thresh.reshape((-1, 1))
        p_a = np.count_nonzero(sample_a_in_set, axis=1) / sample_a.size

        # Comute corresponding mass in samples from b with thresholds
        sample_b_in_set = sample_b.reshape((1, -1)) >= thresh.reshape((-1, 1))
        p_b = np.count_nonzero(sample_b_in_set, axis=1) / sample_b.size

        return p_a, p_b

    def _add_confidence(self, p1, p2, p1_all, p2_all, n):

        # Use multiple sound confidence bounds
        # Use the thightest confidence bound
        bounds = []

        if self.confidence_dwk > 0:
            bounds.append(lambda: self.dkw_bound(p1, p2, n, self.confidence_dwk * (1 - self.config.confidence)))
        
        if self.confidence_dwk < 1:
            bounds.append(lambda: self.binomial_bound(p1_all, p2_all, n, (1 - self.confidence_dwk) * (1 - self.config.confidence)))

        # Compute bounds
        all_p = [b() for b in bounds]
        
        # Use all valid confident bounds
        p1_lcb = np.concatenate([p for (p, _) in all_p])
        p2_ucb = np.concatenate([p for (_, p) in all_p])

        return p1_lcb, p2_ucb
