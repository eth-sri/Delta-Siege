# This code contains elements from https://github.com/ryan112358/private-pgm/tree/master
# under the following lisence:
#
# Copyright 2019, PyTorch team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
import itertools
import numpy as np
from pathlib import Path
from typing import List, Tuple

# DeltaSiege imports
from . import Mechanism

# PGM imports
from mbi import Dataset
from mst import MST, measure
from aim import AIM, compile_workload, Identity
from cdp2adp import cdp_rho

class PGMMechanism(Mechanism):

    """
    Helper class to audit methods from McKenna's library: Graphical-model based estimation and inference for differential privacy
    """
    
    def guarantee_(self, epsilon: float, delta: float) -> float:
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
                The rho-parameter which is found by using cdp_rho as is done by the authors.
                The proof method only uses zero-concentraded DP to provide DP guarantees and the rho parameter 
                which is found using cdp_rho.
        """

        if epsilon < 0 or delta <= 0:
            return float("inf")

        r = cdp_rho(epsilon, delta)

        if r <= 0:
            return float("inf")

        return 1 / r

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
                Defaults to epsilon >= 0 and 0 < delta <= 1
        """
        return epsilon >= 0 and 1 >= delta > 0

    def stringify_input(self, input : Tuple[Path, Path]) -> str:
        """
        Injectively maps input to a string.

        Args:
            input: Tuple[Path, Path]
                A tuple describing the path to the dataset and the domain
        
        Returns:
            : str
                Use filename to uniquely map to string.
        """
        data_file, _ = input
        return ".".join(data_file.name.split(".")[:-1])

    def __call__(self, input : Tuple[str, str], n_samples : int = 1) -> np.ndarray:
        """
        Get n samples from the data source

        Args:
            input: Tuple[Path, Path]
                A tuple describing the path to the dataset and the domain
            n: int
                Number of samples

        Returns:
            : np.ndarray
                Returns a Numpy array with the batches of samples. The shape of the returned arrays is (n, m) 
                m is the number of measurements we perform
        """

        # Load the dataset
        data_file, domain_file = input
        data = Dataset.load(data_file, domain_file)

        # Make results into (n, m) Numpy array
        result = np.array([
            self._helper(data, self.epsilon, self.delta) for _ in range(n_samples)
        ], dtype=np.float64)

        return result
    
    @abstractmethod
    def _helper(self, data : Dataset, epsilon: float, delta: float) -> List:
        """
        Get m measurements from the internal mechanism

        Args:
            data: Dataset
                The dataset which is measured
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter

        Returns:
            : List
                A list of the measurements
        """
        pass


class MSTInternalMechanism(PGMMechanism):

    """
    Helper class to audit the inner parts of McKenna's MST mechanism as found:
    https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
    """

    def _helper(self, data : Dataset, epsilon: float, delta: float) -> List:
        """
        Get m measurements from the internal mechanism

        Args:
            data: Dataset
                The dataset which is measured
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter

        Returns:
            : List
                A list of the measurements
        """

        # First part of MST mechanism
        # Can be found at 
        # https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py#L20-L23
        rho = cdp_rho(epsilon, delta)
        sigma = np.sqrt(3/(2*rho))
        cliques = [(col,) for col in data.domain]
        log1 = measure(data, cliques, sigma)

        return [y[1] for _, y, _, _ in log1]

    @property
    def name(self) -> str:
        """
        Provides a plain name for the data source
        """
        return "MST Internal"


class MSTMechanism(PGMMechanism):

    """
    Helper class to audit the full part of McKenna's MST mechanism as found:
    https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
    """

    def _helper(self, data, epsilon: float, delta: float) -> List:
        """
        Get m measurements from the MST mechanism.

        Args:
            data: Dataset
                The dataset which is measured
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter

        Returns:
            : List
                A list of the measurements
        """
        return MST(data, epsilon, delta).datavector()

    @property
    def name(self) -> str:
        """
        Provides a plain name for the data source
        """
        return "MST"


class AIMInternalMechanism(PGMMechanism):

    """
    Helper class to audit the inner parts of McKenna's AIM mechanism as found:
    https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py
    """

    class AIMHelper(AIM):

        """
        Derives directly from McKenna's implementation of AIM, 
        but returns after computing the first measurements. This is given by
        https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py#L65-L82
        """

        def run(self, data, W):
            rounds = self.rounds or 16*len(data.domain)
            workload = [cl for cl, _ in W]
            candidates = compile_workload(workload)
            answers = { cl : data.project(cl).datavector() for cl in candidates }

            oneway = [cl for cl in candidates if len(cl) == 1]

            sigma = np.sqrt(rounds / (2*0.9*self.rho))
            epsilon = np.sqrt(8*0.1*self.rho/rounds)
          
            measurements = []
            print('Initial Sigma', sigma)
            rho_used = len(oneway)*0.5/sigma**2
            for cl in oneway:
                x = data.project(cl).datavector()
                y = x + self.gaussian_noise(sigma,x.size)
                I = Identity(y.size) 
                measurements.append((I, y, sigma, cl))
            
            return [y[1] for _, y, _, _ in measurements]

    def _init_helper(self, epsilon: float, delta: float, **kwargs) -> None:
        """
        Initialize the method and the helper class for the internal AIM procedure

        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        """
        self._aim = AIMInternalMechanism.AIMHelper(epsilon, delta)
        super()._init_helper(epsilon, delta, **kwargs)

    def _helper(self, data, epsilon: float, delta: float):
        """
        Get m measurements from the internal mechanism.
        Use the same workload as in the original implementation
        https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py#L166

        Args:
            data: Dataset
                The dataset which is measured
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter

        Returns:
            : List
                A list of the measurements
        """
        workload = list(itertools.combinations(data.domain, 1))
        return self._aim.run(data, [(cl, 1.0) for cl in workload])

    @property
    def name(self) -> str:
        """
        Provides a plain name for the data source
        """
        return "AIM Internal"


class AIMMechanism(PGMMechanism):

    """
    Helper class to audit the full part of McKenna's AIM mechanism as found:
    https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py
    """

    def _init_helper(self, epsilon: float, delta: float, **kwargs) -> None:
        """
        Initialize the method and the helper class for the internal AIM procedure

        Args:
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter
        """

        # Use the full mechanism as given in 
        # https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py
        self._aim = AIM(epsilon, delta)
        super()._init_helper(epsilon, delta, **kwargs)

    def _helper(self, data, epsilon: float, delta: float) -> List:
        """
        Get m measurements from the AIM mechanism.

        Args:
            data: Dataset
                The dataset which is measured
            epsilon: float
                The epsilon DP-parameter
            delta: float
                The delta DP-parameter

        Returns:
            : List
                A list of the measurements
        """
        workload = list(itertools.combinations(data.domain, 1))
        return self._aim.run(data, [(cl, 1.0) for cl in workload]).datavector()

    @property
    def name(self) -> str:
        """
        Provides a plain name for the data source
        """
        return "AIM"
