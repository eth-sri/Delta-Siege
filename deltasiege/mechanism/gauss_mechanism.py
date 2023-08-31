from abc import abstractmethod
import numpy as np
from typing import Callable, Dict

from . import Mechanism

class GaussMechanism(Mechanism):

    """
    Base class for different implementations of the Gauss mechansim as given in 
    https://www.nowpublishers.com/article/Details/TCS-042
    """

    def _init_helper(self, epsilon: float, delta : float, sensitivity : float, **kwargs) -> None:
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

        self.sensitivity = sensitivity
        Mechanism._init_helper(self, epsilon=epsilon, delta=delta, **kwargs)
        
    @property
    def std(self) -> float:
        """
        Returns the standard deviation used by the mechanism
        """
        return self.g0


class GaussMechanismClassic(GaussMechanism):
    """
    Base class for different implementations of the Gauss mechansim as given in 
    https://www.nowpublishers.com/article/Details/TCS-042
    based on the analysis provided in the paper
    """

    def constraint(self, epsilon : float, delta : float) -> bool:
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
                Defaults to epsilon >= 0 and 0 <= delta <= 1
        """
        return 0 <= epsilon <= 1 and 0 <= delta <= 1
    
    def guarantee_(self, epsilon : float, delta : float) -> float:
        """
        A mapping of (epsilon, delta) to a parameter rho, which uniquely specifies the privacy level
        Is non-increasing in both epsilon and delta.
        For the classical Gaussian mechanism, one possible parameter is the standard deviation as given in
        https://www.nowpublishers.com/article/Details/TCS-042


        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        
        Returns:
            : float
                The standard deviation if delta in (0, 1] and epsilon in (0, 1]
                If epsilon or delta <= 0, inf is returned
        """

        if epsilon <= 0 or delta <= 0:
            return float("inf")

        return np.sqrt(2 * np.log(1.25 / delta) * np.square(self.sensitivity / epsilon))

    def perturb_delta(self, new_delta : float) -> float:
        """
        Returns a epsilon value such that self.guarantee(epsilon, new_delta) == self.g0
        Uses the formula for the standard deviation in 
        https://www.nowpublishers.com/article/Details/TCS-042

        Args:
            new_delta: float
                New delta DP parameter
        
        Returns:
            : float
                Value epsilon such that self.guarantee(epsilon, new_delta) == self.g0
                If no x is found or an error occurs, None is returned
        """

        # Can't have a new delta
        if new_delta <= 0:
            return None

        # Use the explicit formula
        new_epsilon = self.sensitivity / self.std * np.sqrt(2 * np.log(1.25 / new_delta))

        # Epsilon values are only valid up to 1
        if new_epsilon > 1.0:
            return None
        else:
            return new_epsilon

    def perturb_epsilon(self, new_epsilon : float) -> float:
        """
        Returns a epsilon value such that self.guarantee(epsilon, new_epsilon) == self.g0
        Uses the formula for the standard deviation in 
        https://www.nowpublishers.com/article/Details/TCS-042

        Args:
            new_delta: float
                New delta DP parameter
        
        Returns:
            : float
                Value delta such that self.guarantee(new_epsilon, delta) == self.g0
                If no x is found or an error occurs, None is returned
        """

        # Can't have a new delta
        if new_epsilon <= 0 or new_epsilon > 1:
            return None

        # Use the explicit formula
        new_delta = np.exp(-0.5 * np.square(new_epsilon * self.std / self.sensitivity)) * 1.25

        return new_delta


class GaussMechanismBlackBox(GaussMechanism):
    """
    Base class for different implementations of the Gauss mechansim as given in 
    https://www.nowpublishers.com/article/Details/TCS-042
    with a potentially more general analysis
    """

    def _init_helper(self, epsilon: float, delta : float, sensitivity : float, **kwargs) -> None:
        """
        Initializer

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
        # Initialize helper
        self.sensitivity = self._dtype(sensitivity)
        self._gauss = self._helper_class(epsilon, delta)
        
        super()._init_helper(epsilon=epsilon, delta=delta, sensitivity=self._dtype(sensitivity).item(), **kwargs)


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

        # Get location and noise mechanism
        loc = self._dtype(input).item()
        randomizer = getattr(self._gauss, self._randomize_name)

        # Return an array of the original data with noise added
        out = np.array([
            randomizer(loc) for _ in range(n)
            ], 
            dtype=self._dtype
        )

        return out

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
                Equals self.upper_eps > epsilon > 0 and 0 < delta < 1
        """
        return 0.0 < epsilon < self._upper_eps and 0.0 < delta < 1.0

    def guarantee_(self, epsilon: float, delta: float) -> float:
        """
        A wrapper for the mapping of (epsilon, delta) to a parameter rho, which uniquely specifies the privacy level.
        Should be non-increasing in both epsilon and delta for any valid constraints.
        Defined as the standard deviation of the underlying mechanism

        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        
        Returns:
            : float
                The standard deviation of the underlying mechanism
        """

        # Ensure that guarantee is only given for valid (\epsilon, \delta)-pairs
        if epsilon <= 0 or delta <= 0:
            return float("inf")
        
        if delta >= 1 or epsilon == float("inf"):
            return 0

        # Initialize new mechanism
        gauss_helper = self._helper_class(epsilon, delta)

        # Return the scaling parameter
        return getattr(gauss_helper, self._scaling_name)

    @abstractmethod
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
        pass
    
    @property
    @abstractmethod
    def _upper_eps(self) -> float:
        """
        Upper bound on the allowed epsilon
        """
        pass
    
    @property
    @abstractmethod
    def _scaling_name(self) -> str:
        """
        Name of the standard deviation in the underlying mechanism
        """
        pass
    
    @property
    @abstractmethod
    def _randomize_name(self) -> str:
        """
        Name of the method for randomizing in the underlying mechanism
        """
        pass
    
    @property
    def _dtype(self) -> np.number:
        """
        Dtype used by the mechanism
        """
        return np.float64
