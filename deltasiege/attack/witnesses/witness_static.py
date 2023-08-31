from abc import abstractmethod
import numpy as np
from typing import Any, Dict, Tuple
import warnings

from . import Witness
from .. import Estimator
from ... import Config, Mechanism, StableClassifier


class WitnessStatic(Witness):

    """
    An abstract base class for witnesses where the thresholding parameters (t, q)
    are fixed and selected beforehand
    """

    def _init_helper(
        self, 
        a1 : Any, a2 : Any, 
        classifier : StableClassifier, config : Config, 
        mechanism : Mechanism, estimator : Estimator,
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

        # Initialize super class
        super()._init_helper(a1, a2, classifier, config, mechanism, estimator)

        # Compute theshold and tie break mass
        t, q = self._find_threshold()
        self.t : np.ndarray = t
        self.q : np.ndarray = q

    def compute_estimate(self) -> None:
        """
        Computes an estimate of the performance of the current witness
        using the given estimator.
        """

        # Get samples from classifier
        sample_1 = np.concatenate([p for (p,) in self._get_samples(self.a1)])
        sample_2 = np.concatenate([p for (p,) in self._get_samples(self.a2)])

        # Count number of samples generated from a1 and a2
        # are considered members of the attack set
        c1 = np.count_nonzero(self.is_member(sample_1, is_raw = False), axis=0)
        c2 = np.count_nonzero(self.is_member(sample_2, is_raw = False), axis=0)

        # Conpute maximum likelihood estimates of p1, p2
        n_samples = self._get_num_samples()
        p1 = c1 / n_samples
        p2 = c2 / n_samples

        # Compute confident estimates of p1, p2.
        p1_lcb, p2_ucb = self.binomial_bound(p1, p2, n_samples)
        assert((p1_lcb <= p1).all() and (p2_ucb >= p2).all())
        
        # Unconfident estimate
        with self.logger(f"estimation"):
            loss, _ = self.estimator(p1, p2, self.mechanism, self.logger)
            self.estimate = loss
        
        # Confident estimate
        with self.logger(f"estimation_c"):
            loss_c, _ = self.estimator(p1_lcb, p2_ucb, self.mechanism, self.logger)
            self.estimate_c = loss_c
            
        if self.estimate >= self.estimate_c and np.isfinite(self.estimate_c):
            warnings.warn(
                f"There might be a soundness issure as " \
                f"estimate={self.estimate} and estimate_c={self.estimate_c}"
            )

    def is_member(self, x : Any, is_raw : bool = True) -> bool:
        """
        Returns if x is considered a member of the attack set described by the witness

        Args:
            x: Any
                Input to the mechanism
            is_raw: bool
                Describes if x is the raw input to the mechanism (True)
                or if x is the probability predicted by self.classifier
        Returns:
            : bool
                Whether x is a member of the witness
        """

        # For raw inputs get probability from the classifier
        if is_raw:
            p = self.classifier(x)

            # If the classifier is inverted - use the inverse probability
            if self.inverted_classifier:
                p = 1 -p

        # Else it holds that x is the output from the classifier
        else:
            p = x

        # Allow to compare multiple values of (t, q) with p
        p = p.reshape((-1, 1))
        t = self.t.reshape(1, -1)
        q = self.q.reshape(1, -1)

        # x is always a member if p > t or is a member w.p q if p == t
        over_t = p > t
        equal_t = (p == t) & np.random.binomial(1, q, (1, q.size))

        return over_t | equal_t

    @abstractmethod
    def _find_threshold(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract method for finding the threshold parameters during initialization

        Returns:
            : np.ndarray
                Threshold parameters (t). Is an Numpy array of shape (-1,)
            : np.ndarray
                Threshold proabnability parameters (q). Is an Numpy array of shape (-1,)
        """
        pass