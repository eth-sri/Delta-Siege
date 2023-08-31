from abc import abstractmethod
import numpy as np
from typing import Dict, Optional, Tuple

from ... import Logger, Mechanism, Entity

class Estimator(Entity):

    def get_theoretical_curve(
        self,
        mechanism: Mechanism, logger : Logger, 
        delta : Optional[np.ndarray] = None, epsilon : Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the (epsilon, delta)-pairs such that mechanism.guarantee(epsilon, delta) == mechanism.g0

        Args:
            mechansim: Mechanism
                The mechanism which is being audited
            logger: Logger
                Logger for storing results
            delta : Optional[np.ndarray]
                Possible delta values. Either this or epsilon must be provided, but not both
            epsilon : Optional[np.ndarray]
                Possible delta values. Either this or delta must be provided, but not both
        
        Returns:
            : np.ndarray
                Theoretical epsilon values
            : np.ndarray
                Theoretical delta values
        """

        # Check precondition
        if (delta is None) == (epsilon is None):
            logger.raise_exception(ValueError("Either epsilon or delta must be None and the other a Numpy array"))
        
        # Make switch
        perturb_delta = epsilon is None
        size = delta.size if perturb_delta else epsilon.size

        # Perturb the DP parameters so that the mechanism remains equivalent
        theoretical = []

        # Print the computation with a progression bar
        with logger(
            "perturb", timing=True, timing_string="perturbing privacy parameters",
            pbar=True, pbar_total=size, pbar_desc="Pertubing privacy parameters"
        ):
            for x in (delta if perturb_delta else epsilon):
                theoretical.append(mechanism.perturb_delta(x) if perturb_delta else mechanism.perturb_epsilon(x))
                logger.step()
        
        # Make into np.ndarray
        theoretical = np.array(theoretical)

        # Return according to which was perturbed
        return (theoretical, delta) if perturb_delta else (epsilon, theoretical)

    def get_empirical_curve(
        self,
        logger : Logger, 
        p1 : np.ndarray, p2 : np.ndarray,
        delta : Optional[np.ndarray] = None, epsilon : Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the (epsilon, delta)-pairs corresponding to the observed values of p1 and p2

        Args:
            mechansim: Mechanism
                The mechanism which is being audited
            logger: Logger
                Logger for storing results
            p1: np.ndarray
                A vector of empirical probability mass for $P[M(a1) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            p2: np.ndarray
                A vector of empirical probability mass for $P[M(a2) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            delta : Optional[np.ndarray]
                Possible delta values. Either this or epsilon must be provided, but not both
            epsilon : Optional[np.ndarray]
                Possible delta values. Either this or delta must be provided, but not both
        
        Returns:
            : np.ndarray
                Empirical epsilon values
            : np.ndarray
                Empirical delta values
        """

        # Check precondition
        if (delta is None) == (epsilon is None):
            logger.raise_exception(ValueError("Either epsilon or delta must be None and the other a Numpy array"))

        # Check inputs
        if not (isinstance(p1, np.ndarray) and p1.ndim == 1):
            logger.raise_exception(ValueError("p1 should be a 1d Numpy array"))
        
        if not (isinstance(p2, np.ndarray) and p2.ndim == 1):
            logger.raise_exception(ValueError("p2 should be a 1d Numpy array"))

        # Exand dims of arrays
        p1 = np.expand_dims(p1, axis=1)
        p2 = np.expand_dims(p2, axis=1)

        # Compute epsilon values
        if epsilon is None:

            # Subtract delta
            p1_sub_d = p1 - np.expand_dims(delta, axis=0)

            # Epsilon is found as a parameter of delta
            # Default value is negative inf if invalid computation
            epsilon = np.empty_like(p1_sub_d)
            epsilon.fill(-np.inf)

            # Where valid division
            mask_valid = (p1_sub_d > 0) & (p2 > 0)
            np.divide(p1_sub_d, p2, out=epsilon, where=mask_valid)
            epsilon[mask_valid] = np.log(epsilon[mask_valid])

            # Where mass of p1 - delta > 0 and p2 == 0, eps = inf
            mask_eps_inf = (p1_sub_d > 0) & ~(p2 > 0)
            epsilon[mask_eps_inf] = np.inf

            # Make delta have same shape as epsilon
            delta = np.repeat(np.expand_dims(delta, axis=0), epsilon.shape[0], axis=0)

        # Compute delta values
        else:
            delta = p1 - p2 * np.expand_dims(np.exp(epsilon), axis=0)
            epsilon = np.repeat(np.expand_dims(epsilon, axis=0), delta.shape[0], axis=0)
        
        return epsilon, delta

    @abstractmethod
    def __call__(
        self, 
        p1 : np.ndarray, p2 : np.ndarray, 
        mechanism : Mechanism, logger : Logger
    ) -> Tuple[float, int]:
        """
        Abstract method for computing $\min_{\delta, \epsilon} \\rho(\epsilon, \delta)$ 
        subject to $p1 - \delta \ge \exp{\epsilon} p2$

        Args:
            p1: np.ndarray
                A vector of empirical probability mass for $P[M(a1) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            p2: np.ndarray
                A vector of empirical probability mass for $P[M(a2) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            mechansim: Mechanism
                The mechanism which is being audited
            logger: Logger
                Logger for storing results
        
        Returns:
            : float
                Returns the minimal relative decrease in $\\rho$ over all simultaneously 
                tested attacks. If rel_rho < 1, then a violation is found
            : int
                Indicates which $\delta$-value was used to produce the minimal relative decrease
        """
        pass


class LinesearchEstimator(Estimator):

    """
    An estimator for detecting the largest violation given a mechanism
    and mass of an attack set. 
    Optimization is done over the Pareto front of the $\epsilon$, $\delta$ values
    """

    def _init_helper(self, epsilon_steps : Optional[np.ndarray] = None, delta_steps : Optional[np.ndarray] = None) -> None:
        """
        Args:
            epsilon_steps: Optional[np.ndarray]
                If given it sets the specific steps of $\epsilon$-values which 
                are used during optimization. 
            delta_steps: Optional[np.ndarray]
                If given it sets the specific steps of $\delta$-values which 
                are used during optimization. 
                Defaults to 900 log-uniformly distributed points over [1e-9, 1]
                if both epsilon_steps and delta_steps is not provided
        """
        super()._init_helper()

        # Compute the delta steps if not given
        # They are log uniformly distributed over the range [1e-9, 1]
        # Also include the boundary value 0
        if delta_steps is None and epsilon_steps is None:
            delta_steps = np.concatenate([
                np.zeros(1),
                np.exp(np.linspace(np.log(1e-9), 0, num = 900))
            ], axis=0)
        
        # Set steps
        self.delta_steps = delta_steps
        self.epsilon_steps = epsilon_steps

    def __call__(
        self, 
        p1 : np.ndarray, p2 : np.ndarray, 
        mechanism : Mechanism, logger : Logger
    ) -> Tuple[float, int]:
        """
        Computes $\min_{\delta, \epsilon} \\rho(\epsilon, \delta)$ 
        subject to $p1 - \delta \ge \exp{\epsilon} p2$

        Args:
            p1: np.ndarray
                A vector of empirical probability mass for $P[M(a1) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            p2: np.ndarray
                A vector of empirical probability mass for $P[M(a2) \in S]$
                where S is the attack set.
                Dimension is (n, ) where n gives the number of attacks tested simultaneous
            mechansim: Mechanism
                The mechanism which is being audited
            logger: Logger
                Logger for storing results
        
        Returns:
            : float
                Returns the minimal relative decrease in $\\rho$ over all simultaneously 
                tested attacks. If rel_rho < 1, then a violation is found
            : int
                Indicates which $\delta$-value was used to produce the minimal relative decrease
        """

        # Helpers to get curves
        empirical_curve, theoretical_curve, argmax_idx = [] , [], []

        # Fixed delta values
        if not self.delta_steps is None:

            # Compute the maximal empirical epsilon for different values of delta
            epsilon, _ = self.get_empirical_curve(logger, p1, p2, delta=self.delta_steps)

            # Only use the maximal possible epsilon value for each delta value
            empirical_curve.append((epsilon.max(axis=0), self.delta_steps))
            argmax_idx.append(epsilon.argmax(axis=0))

            # Compute the maximal theorethical epsilon for different values of delta
            theoretical_curve.append(self.get_theoretical_curve(mechanism, logger, delta=self.delta_steps))

        # Fixed epsilon values
        if not self.epsilon_steps is None:

            # Compute the maximal empirical epsilon for different values of delta
            _, delta = self.get_empirical_curve(logger, p1, p2, epsilon=self.epsilon_steps)

            # Only use the maximal possible epsilon value for each delta value
            empirical_curve.append((self.epsilon_steps, delta.max(axis=0)))
            argmax_idx.append(delta.argmax(axis=0))

            # Compute the maximal theorethical epsilon for different values of delta
            theoretical_curve.append(self.get_theoretical_curve(mechanism, logger, epsilon=self.epsilon_steps))

        # Concatenate results
        empirical_curve = np.concatenate(empirical_curve, axis=1)
        theoretical_curve = np.concatenate(theoretical_curve, axis=1)
        argmax_idx = np.concatenate(argmax_idx)

        # Only consider delta values
        guarantee = []
        with logger(
            "guarantee", timing=True, timing_string="computing the guarantees", 
            pbar=True, pbar_total=empirical_curve.shape[-1], pbar_desc="Computing guarantee"
        ):
            for (e_emp, d_emp), (e_theo, d_theo) in zip(empirical_curve.T, theoretical_curve.T):

                # If the theoretical guarantee is None - this is not a valid comparison
                # Therefore no need to comppute it
                if e_theo is None or d_theo is None:
                    guarantee.append(None)
                else:
                    guarantee.append(mechanism.guarantee(e_emp, d_emp, ignore_constraints=True))
                
                logger.step()
        guarantee = np.array(guarantee, dtype=np.float32)
        
        # Find the most position resulting in the most violation
        if (guarantee != None).any():

            # Fill any None values in with np.inf 
            helper = guarantee
            helper[helper == None] = np.inf
            helper[np.isnan(helper)] = np.inf
            
            # Find the most violated guarantee
            idx = np.argmin(helper)
            g = helper[idx]
        
        else:
            idx, g = 0, float("inf")

        # Add values to the logger
        logger.add_datapoint(**{
            "initial_epsilon": mechanism.epsilon,
            "initial_delta": mechanism.delta,
            "initial_guarantee": mechanism.g0,
            "p1": p1.flatten().tolist(),
            "p2": p2.flatten().tolist(),
            "argmax_idx": argmax_idx.tolist(),
            "empirical_curve": empirical_curve.tolist(),
            "theoretical_curve": theoretical_curve.tolist(),
            "empirical_guarantee": guarantee.tolist(),
            "idx": idx,
        })

        return g / mechanism.g0, argmax_idx[idx]
