from __future__ import annotations
from abc import abstractmethod
from enum import Enum
import numpy as np
from typing import Any, Dict, Generator, Optional, Tuple

from .binomial_bound import binomial_bound
from .dwk_bound import dkw_bound
from .. import Estimator
from ... import Config, Mechanism, StableClassifier, Entity


class Witness(Entity):

    """
    Abstract base class for witnesses demonstrating differential disthinguishability (DD)
    """

    class State(Enum):

        """
        Helper class for indicating the current state of a witness

        The states are:
        * Init: The initialization state where static parameters are selected.
        * Check: The checking state where the performance of different witnesses 
                 are estimated and the best selected.
        * Final: The final state where the final estimation of the performance
                 of the best witness from the checking state is assesed.
        """
        Init = 0
        Check = 1
        Final = 2

        @property
        def names(self) -> Dict:
            """
            Provides acces to a dictionary containg all states
            and their corresponding plain names.

            Returns:
                names: dict
                    Dictionary containg all states.
                    The key is the state and the values is the corresponding plain name
            """
            names = {
                Witness.State.Init: "init",
                Witness.State.Check: "check",
                Witness.State.Final: "final"
            }

            return names

        def __str__(self) -> str:
            """
            A state is stringified by its plain name
            """
            return self.names[self]

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

        # Add logger
        self.logger.open(str(Witness.State.Init))

        # Set basic parameters
        self.state : Witness.State = Witness.State.Init
        self.classifier : StableClassifier = classifier
        self.config : Config = config
        self.mechanism : Config = mechanism
        self.estimator : Estimator = estimator
        self.a1, self.a2 = a1, a2
        self.estimate, self.estimate_c = None, None

    def _get_samples(self, *inputs) -> Generator:
        """
        Generates output of size n from the mechanism (n samples for each input given).
        Returns the probabiliy of the observation originating from a1 according 
        to the classifier, i.e. an approximation of $P[a == a1 | b]$ when $b ~ M(a)$.
        The output is returned batchwise with batch size 
        given by self.config.prediction_batch_size

        Args:
            inputs: *Any
                Input to the mechanism

        Returns:
            prob_batch: Generator
                A generator of vectors containing probabilities that samples originate from a1.
                Shape is (v_1, v_2, ..., v_k) where each v_i is a np.ndarray 
                of shape (batch_size, ). v_i corresponds to samples from the i-th input
        """

        # Defualt inputs is (a1, a2)
        if not inputs:
            inputs = (self.a1, self.a2)

        for a in inputs:
            if not self.classifier.stringify_input(a) in list(self.classifier.train_inputs):
                self.logger.raise_exception(ValueError(
                    f"Input ({self.classifier.stringify_input(a)}) is not in the inputs to the classifier which was trained on {self.classifier.train_inputs}"
                ))
        
        # Use the current state
        n = self._get_num_samples()
        idx = self._get_offset()

        # Get the predictions
        itrs = tuple(
            self.classifier.get_batches(a, n, self.config.batch_size, idx) 
            for a in inputs
        )

        # Handle the case if the classifier is inverted
        for batch in zip(*itrs):
            
            # If the classifier is inverted, 
            # the probability is 1 - p where the classifier returns p
            if self.inverted_classifier:
                yield (1 - b.flatten() for b in batch)
            else:
                yield (b.flatten() for b in batch)

    def _get_num_samples(self) -> int:
        """
        Get the number of samples used by 
        the iterator for the current state of the witness

        Returns:
            : int
                Get the number of samples to be used in the current state
        """

        # Perform a switch case
        if self.state == Witness.State.Init:
            return self.config.n_init
        
        elif self.state == Witness.State.Check:
            return self.config.n_check
        
        elif self.state == Witness.State.Final:
            return self.config.n_final
        
        # If state is not among the know states, raise an exception
        else:
            raise Exception("Unknown state")

    def _get_offset(self) -> int:
        """
        Get the offset for the current state

        Returns:
            : int
                Get the offset in the classifier used in the current state
        """

        # Perform a switch case
        if self.state == Witness.State.Init:
            return 0
        
        elif self.state == Witness.State.Check:
            return self.config.n_init
        
        elif self.state == Witness.State.Final:
            return self.config.n_init + self.config.n_check
        
        # If state is not among the know states, raise an exception
        else:
            raise Exception("Unknown state")

    @property
    def inverted_classifier(self) -> bool:
        """
        Returns if self.classifier was train on the inputs (a2, a1) instead of (a1, a2)
        """
        a1_, a2_ = self.classifier.train_inputs
        a1_key = self.mechanism.stringify_input(self.a1)
        a2_key = self.mechanism.stringify_input(self.a2)
        return (a1_ != a1_key) or (a2_ != a2_key)

    @property
    def state(self) -> State:
        """
        The current state of the witness
        """
        return self.state_

    @state.setter
    def state(self, state : State) -> None:
        """
        Set the current state of the witness

        Args:
            state: State
                The new state of the witness
        """
        
        if not isinstance(state, Witness.State):
            raise ValueError("State must be of type Witness.State")
        
        # Set state which requires to update the logger
        self.logger.close()
        self.state_ = state
        self.logger.open(str(self.state))

    @abstractmethod
    def compute_estimate(self) -> None:
        """
        Computes an estimate of the performance of the current witness
        using the given estimator.
        """
        pass

    @abstractmethod
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
        pass

    def __lt__(self, other :  Witness) -> bool:
        """
        Returns if self < other

        Args:
            other: Witness
                Other witess to compare agains
        
        Returns
            : bool
                Returns self.estimate_c < other.estimate_c if confident_comparison.
                Otherwise, use self.estimate < other.estimate
                If the estimates have not been computed, then None is returned
        """

        # Compare based on estimate; either confident or non-confident
        a, b = (self.estimate_c, other.estimate_c) if self.config.confident_comparison \
               else (self.estimate, other.estimate)
        
        # Test is estimates have been computed
        if (a is None) or (b is None):
            return None
        else:       
            return a < b

    # Import static bound helpers
    def binomial_bound(
        self, 
        p1 : np.ndarray, p2 : np.ndarray, 
        n : int, alpha : Optional[float] = None
    )-> Tuple[np.ndarray, np.ndarray]:
        """
        Computes an lower confidence bound on p1 and upper confidence on p2
        by using the binomial proportion condfidence interval

        Args:
            p1: np.ndarray
                A vector of empirical probability mass for $P[M(a1) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            p2: np.ndarray
                A vector of empirical probability mass for $P[M(a2) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            n: int
                Number of samples used for the estimation of p1 and p2, assumed to be equal
            alpha: Optional[float]
                Confidence level. Defaults to 1 - self.config.confidence

        Returns:
            p1_lcb: np.ndarray
                A vector of confident lower bounds of the elements of p1
                The lower bounds all hold uniformly w.p. at least 1 - alpha / 2
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            p2_ucb: np.ndarray
                A vector of confident upper bounds of the elements of p2
                The lower bounds all hold uniformly w.p. at least 1 - alpha / 2
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
        """

        # Handle default value
        if alpha is None:
            alpha = 1 - self.config.confidence

        return binomial_bound(p1, p2, n, alpha)
 
    def dkw_bound(
        self, 
        p1 : np.ndarray, p2 : np.ndarray, 
        n : int, alpha : Optional[float] = None
    )-> Tuple[np.ndarray, np.ndarray]:
        """
        Computes an lower confidence bound on p1 and upper confidence on p2
        by using the Dvoretzky–Kiefer–Wolfowitz inequality.

        Args:
            p1: np.ndarray
                A vector of empirical probability mass for $P[M(a1) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            p2: np.ndarray
                A vector of empirical probability mass for $P[M(a2) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            n: int
                Number of samples used for the estimation of p1 and p2, assumed to be equal
            alpha: Optional[float]
                Confidence level. Defaults to 1 - self.config.confidence

        Returns:
            p1_lcb: np.ndarray
                A vector of confident lower bounds of the elements of p1
                The lower bounds all hold uniformly w.p. at least 1 - alpha / 2
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            p2_ucb: np.ndarray
                A vector of confident upper bounds of the elements of p2
                The lower bounds all hold uniformly w.p. at least 1 - alpha / 2
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
        """

        # Handle default value
        if alpha is None:
            alpha = 1 - self.config.confidence

        return dkw_bound(p1, p2, n, alpha)
