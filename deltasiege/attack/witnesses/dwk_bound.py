import numpy as np
from typing import Tuple

def dkw_bound(
    p1 : np.ndarray, p2 : np.ndarray, 
    n : int, alpha : float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a confidence bound based on the Dvoretzky–Kiefer–Wolfowitz inequality.
    Returns an upper bound on all elements in p2 and a lower bound in all elements in p1.
    The confidence bound holds simoultaneous on all elements in p1 and p2, 
    as the Dvoretzky–Kiefer–Wolfowitz provides a uniform bound.

    For further details see: 
    https://en.wikipedia.org/wiki/Dvoretzky%E2%80%93Kiefer%E2%80%93Wolfowitz_inequality

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
        alpha: float
            Confidence level

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

    # Check inputs
    if not (isinstance(p1, np.ndarray) and p1.ndim == 1):
        raise ValueError("p1 should be a 1d Numpy array")
    
    if not (isinstance(p2, np.ndarray) and p2.ndim == 1):
        raise ValueError("p2 should be a 1d Numpy array")

    # Use union bound on confidence
    alpha = alpha / 2
    
    # Create lower bound confidence estimate on p1 
    # with probability less than alpha to not hold
    margin_1 = np.sqrt(np.log(min(2, 1 / alpha)) / 2 / n)
    p1_lcb = p1 - margin_1

    # Create upper bound confidence estimate on p2
    # with probability less than alpha to not hold
    margin_2 = np.sqrt(np.log(min(2, 1 / alpha)) / 2 / n)
    p2_ucb = p2 + margin_2

    return p1_lcb, p2_ucb
