from abc import abstractmethod
import json
from pathlib import Path
from scipy.optimize import bisect
from typing import Any, Callable, Dict, Optional

from .. import DataSource

class Mechanism(DataSource):
    """
    Base class for differentially private mechanisms
    """

    ALLOW_LOAD = False

    def __init__(self, *args, **kwargs) -> None:
        
        # Save the initialization arguments
        self._args,self._kwargs = args, kwargs

        super().__init__(*args, **kwargs)

    def _init_helper(
        self, 
        epsilon : float, delta : float
    ) -> None:
        """
          Initializes the mechanism

          Args:
              epsilon: float
                  Epsilon DP parameter
              delta: float
                  Delta DP parameter
              base_folder: Path
                  Path to the base folder for storing the data. Allows for exact reproducability and reusage.
                  If not specified, no data is stored
              logger : Optional[Logger]
                  Logger to log results. Defualts to a logger which is down, i.e. nothing is loged or printed
        """
        super()._init_helper()

        # Save general parameters
        self.epsilon : float = epsilon
        self.delta : float = delta
        self.g0 : float = self.guarantee(self.epsilon, self.delta)

        # Warns if invalid parameter
        if not self.constraint(epsilon, delta):
            self.logger.print(f"Warning: Invalid initialization of {self.name} with {str(self)}")

        # Set up caching structure
        if self.store_data:
            self.guarantee_cache_path : Path = self.base_folder / "guarantee_cache.json"

            # Load cache from file
            if self.guarantee_cache_path.exists():
                with open(self.guarantee_cache_path, "r") as f:
                    self.guarantee_cache = json.load(f)
            else:
                self.guarantee_cache = {}

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
        return 0 <= epsilon and 0 <= delta <= 1
    
    def guarantee(self, epsilon : float, delta : float, ignore_constraints : bool = False) -> float:
        """
        A wrapper for the mapping of (epsilon, delta) to a parameter rho, which uniquely specifies the privacy level.
        Should be non-increasing in both epsilon and delta for any valid constraints

        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
            ignore_constraints : bool
                If to compute the parameter, ignoring the soundness
        
        Returns:
            : float
                The rho-parameter which describes the differential privacy level
                If ignore_constraints is False and (epsilon, delta) is not valid according to self.constraint
                None is returned
        """
        if not ignore_constraints and not self.constraint(epsilon, delta):
            return None
        else:
            return self.guarantee_(epsilon, delta)
    
    @abstractmethod
    def guarantee_(self, epsilon : float, delta : float) -> float:
        """
        A mapping of (epsilon, delta) to a parameter rho, which uniquely specifies the privacy level
        Should be non-increasing in both epsilon and delta. Does not consider any constraints

        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        
        Returns:
            : float
                The rho-parameter which describes the differential privacy level
                Should be non-increasing in both epsilon and delta
        """
        pass
      
    def perturb_epsilon(self, new_epsilon : float, a : float = 0.0, b : float = 1.0) -> float:
        """
        Returns a delta value such that self.guarantee(new_epsilon, delta) == self.g0

        Args:
            new_epsilon: float
                New epsilon DP parameter
            a: float
                Lower bound on the returned delta
            b: float
                Upper bound on the returned delta
        
        Returns:
            : float
                Value delta such that self.guarantee(new_epsilon, delta) == self.g0
                If no x is found or an error occurs, None is returned
        """

        # Finde delta in [low, high] such that
        # self.guarantee(new_epsilon, delta) == self.g0
        def f(delta : float):
            g = self.guarantee(new_epsilon, delta)

            # Use switch to handle None values
            if g is None:
                return -float("inf") if delta > self.delta else float("inf")
            else:
                return g - self.g0

        return self.get_value(
            f"epsilon/{new_epsilon}", 
            lambda: self._root_finder(f, a=a, b=b)
        )

    def perturb_delta(self, new_delta : float, a : float = 0.0, b : float = 10.0) -> float:
        """
        Returns a epsilon value such that self.guarantee(epsilon, new_delta) == self.g0

        Args:
            new_delta: float
                New delta DP parameter
            a: float
                Lower bound on the returned epsilon
            b: float
                Upper bound on the returned epsilon
        
        Returns:
            : float
                Value epsilon such that self.guarantee(epsilon, new_delta) == self.g0
                If no x is found or an error occurs, None is returned
        """

        # Finde epsilon in [low, high] such that
        # self.guarantee(epsilon, new_delta) == self.g0
        def f(epsilon : float):
            g = self.guarantee(epsilon, new_delta)

            # Use switch to handle None values
            if g is None:
                return -float("inf") if epsilon > self.epsilon else float("inf")
            else:
                return g - self.g0

        return self.get_value(
            f"delta/{new_delta}", 
            lambda: self._root_finder(f, a=a, b=b)
        )

    def get_value(self, key : str, f : Callable) -> Any:
        """
        Helper function which allows for the use of the caching structure.
        Returns the value of f() which is stored in the caching structure with key as its identifier

        Args:
            key: str
                Key uniquely associated with f()
            f: Callable
                Functor which produce the desired value
        
        Returns:
            : Any
                Value of f() which is associated with key in the caching structure
        """

        # If data is stored - test if value is in cache
        if self.store_data and key in self.guarantee_cache:
            value = self.guarantee_cache[key]
        
        else:

            # Compute value
            value = f()

            # If data is stored - save to cache
            if self.store_data:

                # Save the value in cache
                self.guarantee_cache[key] = value                
                with open(self.guarantee_cache_path, "w+") as f:
                    json.dump(self.guarantee_cache, f)

        return value

    # Use a helper function for computing the new epsilon value
    def _root_finder(self, f : Callable, *args, **kwags) -> float:
        """
        Root finder method. Returns the root of f

        Args:
            f: Callable
                Function to find the root from, i.e. an x s.t. f(x) == 0
            *args:
                Arguments to the solver
            **kwargs:
                Key word arguments to the solver

        Returns:
            : float
                Value x such that f(x) == 0
                If no x is found or an error occurs, None is returned            
        """

        try:
            # Solve by bisection such that f(x) is zero
            x, hist = bisect(
                f, *args, **kwags, full_output=True
            )

            # If not converged raise error
            if not hist.converged:
                raise RuntimeError("Not converged")

        # If an error return None as no solution discoverd
        except Exception as e:
            x = None
        
        return x

    def __getstate__(self, params: Optional[Dict] = None):

        # Only save the initialization parameters
        if params is None:
            params = {"_args": self._args, "_kwargs": self._kwargs}

        return super().__getstate__(params)
    
    def load(self, folder: Optional[Path] = None):

        # Use a normal loading of the stored parameters
        super().load(folder)

        copy_self = type(self)(*self._args, **self._kwargs)

        for attr, value in vars(copy_self).items():
            setattr(self, attr, value)
