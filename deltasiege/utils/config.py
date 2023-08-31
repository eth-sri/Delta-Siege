import multiprocessing
from typing import Dict

from . import Entity

class Config(Entity):
    
    """
    An object holding the global configuration
    """

    def _init_helper(
        self,
        n_train : int = 1_000_000,
        n_init  : int = 1_000_000,
        n_check : int = 1_000_000,
        n_final : int = 5_000_000,
        batch_size : int = 100_000,
        confidence : float = 0.9,
        confident_comparison = True,
        n_jobs : int = multiprocessing.cpu_count(),
        n_experiments : int = multiprocessing.cpu_count(),
        n_trials : int = 1,
        n_trial_jobs : int = 1,
        full_run_all : bool = False,
        seed : int = 0
    ):
        """
        Initialize the global configuration

        Args:
            n_train : int
                Number of input samples used to train the machine learning attack model
            n_init :  int
                Number of samples used to select the hyperparameters of the witness
            n_check : int
                Number of samples used to estimate probabilities P[M(a) in S] approximately
            n_final : int
                Number of samples used to estimate probabilities P[M(a) in S] with high precision
            batch_size : 
                Number of samples in a single batch
            confidence : 
                Requested confidence for the computed lower bound on epsilon
            confident_comparison : 
                Whether to use a confident estimate when compring witnesses
            n_jobs : int
                Number of jobs to run in paralell. If less or equal to 1, no paralellisim is used.
                Defaults to multiprocessing.cpu_count().
            n_experiments : int                 
                Number of independent experiments to run, i.e. each experiment is reseeded.
                Defaults to multiprocessing.cpu_count().
            n_trials : int
                Number of dependent trials to run within an experiments using the optuna module.
                Defaults to 1
            n_trial_jobs : int
                Number of dependent trials to run in parallel. Defaults to 1
            full_run_all : bool
                Whether to run the final estimation for all witnesses

        Note:
            Small batch sizes lead to higher runtime, while large batch sizes require more memory
        """
        self.n_train = n_train
        self.n_init  = n_init
        self.n_check = n_check
        self.n_final = n_final
        self.batch_size = batch_size
        self.confidence = confidence
        self.confident_comparison = confident_comparison
        self.n_jobs = n_jobs
        self.n_experiments = n_experiments
        self.n_trial_jobs = n_trial_jobs
        self.n_trials = n_trials
        self.full_run_all = full_run_all
        self.seed = seed
