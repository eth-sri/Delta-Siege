# This code is in large parts reused from https://github.com/eth-sri/dp-sniper/blob/master/dpsniper/probability/binomial_cdf.py
# under the following lisence:
#
# MIT License

# Copyright (c) 2021 SRI Lab, ETH Zurich

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import floor, ceil
import numpy as np
from scipy.stats import beta
from typing import Tuple

def binomial_lcb(n: int, k: int, alpha: float) -> float:
    """
    Computes a lower confidence bound on the probability parameter p of a binomial CDF.

    Args:
        n: int
            Number of samples  used to estimate p empirically
        k: int
            Number of positive samples
        alpha: float
            Confidence level

    Returns:
        p_lb: float
            The largest p such that Pr[Binom[n,p] >= k] <= alpha
    """
    if k == 0:
        p_lb = 0.0
    else:
        # Inspired by https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
        p_lb = beta.ppf(alpha, k, n - k + 1)
    
    return p_lb


def binomial_ucb(n: int, k: int, alpha: float) -> float:
    """
    Computes an upper confidence bound on the probability parameter p of a binomial CDF.

    Args:
        n: int
            Number of samples used to estimate p empirically
        k: int
            Number of positive samples
        alpha: float
            Confidence level

    Returns:
        p_ub: float
            The smallest p such that Pr[Binom[n,p] <= k] <= alpha
    """
    if k == n:
        p_ub = 1
    else:
        # Inspired by https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
        p_ub = beta.ppf(1 - alpha, k + 1, n - k)
    
    return p_ub


def binomial_bound(
    p1 : np.ndarray, p2 : np.ndarray, 
    n : int, alpha : float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a confidence bound based on the binomial proportion condfidence interval.
    Returns an upper bound on all elements in p2 and a lower bound in all elements in p1.
    The confidence bound holds simoultaneous on all elements in p1 and p2, 
    by means of the union bound.

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

    # Use lower confidence estimate according to binomial estimate
    # Use union bound with p1.size parts
    p1_lcb = np.array([
        binomial_lcb(n, floor(n * x), alpha / p1.size) for x in p1
    ], dtype=p1.dtype)

    # Use upper confidence estimate according to binomial estimate
    # Use union bound with p2.size parts
    p2_ucb = np.array([
        binomial_ucb(n, ceil(n * x), alpha / p2.size) for x in p2
    ], dtype=p2.dtype)

    return p1_lcb, p2_ucb
