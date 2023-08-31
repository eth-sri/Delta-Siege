import numpy as np
from typing import Dict, Generator

from . import GaussMechanismClassic, RNGMechanism

class GaussZigguratMechanism(GaussMechanismClassic, RNGMechanism):

    def _init_helper(
        self, 
        epsilon: float, delta: float, sensitivity: float, 
        word_size : int = 2**32,
        **kwargs
    ) -> None:
        """
        Initializer

        Args:
            epsilon: float
                Epsilon DP parameter
            delta: float
                Delta DP parameter
            sensitivity: float
                Sensitivity used to determine the size of the neighbourhood
            word_size: int
                Number of states in the simulated random number generator in the RNGMechanism
            kwargs:
                Key word arguments to the Mechanism parent class
        """
        RNGMechanism.__init__(self, word_size)
        GaussMechanismClassic._init_helper(self, epsilon, delta, sensitivity, **kwargs)
        self.make_tables()

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
        
        # Get sufficent samples with norm less than one
        batches = []
        size = 0
        while size < n:
            for b in self.get_batch(n):
                size += b.shape[0]
                batches.append(b)

        # Concatenate batches
        x = np.concatenate(batches, axis=0)[:n]

        # Add sign
        sgn = 2 * (self.uniform(n) < 0.5) - 1
        x = sgn * x

        # Add scaling and translation
        s = input + self.std * x
        return s

    def get_batch(self, n : int) -> Generator:
        """
        Get at most n samples sampled according to the Ziggurat method

        Args:
            n: int
                Number of samples

        Returns:
            : Generator
                Returns a generator of Numpy array with in total 
                at most n samples each with shape (-1, 1)
        """
        
        # Generate index
        idx = np.random.randint(0, self.N, n)

        # Get x
        u_x = self.uniform(n)
        x = u_x * self.w[idx]
        mask_x = u_x < self.k[idx]

        yield x[mask_x]

        mask_0 = ~mask_x & (idx == 0)
        yield self.tail(np.count_nonzero(mask_0))
        
        # Get remaining samples
        x = x[~mask_x & ~mask_0]
        idx = idx[~mask_x & ~mask_0]

        # Get y
        u_y = self.uniform(x.size)
        mask_y = u_y * (self.f[idx - 1] - self.f[idx]) < self.f_(x) - self.f[idx]
        yield x[mask_y]

    def tail(self, n: int) -> np.ndarray:
        """
        Get exactly n samples sampled according to the tail
        distribution of the Ziggurat method

        Args:
            n: int
                Number of samples

        Returns:
            : np.ndarray
                Returns a Numpy array with at most n samples with shape (n, 1)
        """
        
        # Get sufficent samples with norm less than one
        batches = []
        size = 0
        while size < n:
            b = self.get_batch_tail(n)
            size += b.shape[0]
            batches.append(b)

        # Concatenate batches
        if len(batches):
            x = np.concatenate(batches, axis=0)[:n]
        else:
            x = np.array([], dtype=np.float64)

        return x

    def get_batch_tail(self, n : int) -> np.ndarray:
        """
        Get at most n samples sampled according to the tail
        distribution of the Ziggurat method

        Args:
            n: int
                Number of samples

        Returns:
            : np.ndarray
                Returns a Numpy array with at most n samples with shape (-1, 1)
        """
        x = -np.log(self.uniform(n)) / self.r
        y = -np.log(self.uniform(n))
        mask = x ** 2 > y ** 2
        return self.r + x[mask]

    def make_tables(self) -> None:
        """
        Initialize constants in accordance with the paper
        """

        # Magic constants from paper
        self.N = 256
        self.v = 0.00492867323399
        self.r = 3.6541528853610088

        # Set up x and f
        self.x = np.empty(self.N)
        self.f = np.empty(self.N)

        # Set top of recursion
        self.x[self.N - 1] = self.r
        self.f[self.N - 1] = self.f_(self.x[-1])

        # Fill midle
        for i in range(self.N - 2, 0, -1):
            self.x[i] = self.f_inv_(self.f[i + 1] + self.v / self.x[i + 1])
            self.f[i] = self.f_(self.x[i])

        # Set bottom of recursion
        self.x[0] = 0
        self.f[0] = self.f_(self.x[0])

        # Set up k and w
        self.k = np.empty(self.N)
        self.k[1:] = self.x[:-1] /self.x[1:]
        self.k[0] = self.r * self.f_(self.r) / self.v

        self.w = np.empty(self.N)
        self.w = self.x
        self.w[0] = self.v / self.f_(self.r)

    def f_(self, x : np.ndarray) -> np.ndarray:
        """
        Compute unnormalized density function

        Args:
            x: np.ndarray
                Input to density function

        Returns:
            : np.ndarray
                Unnormalized density function
        """
        return np.exp(-0.5 * (x ** 2))
    
    def f_inv_(self, y) -> np.ndarray:
        """
        Compute inverse of unnormalized density function (self.f_)

        Args:
            y: np.ndarray
                Input to inverse density function

        Returns:
            : np.ndarray
                Inverse of unnormalized density function
        """
        return np.sqrt(-2 * np.log(y))
