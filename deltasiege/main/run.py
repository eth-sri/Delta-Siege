from __future__ import annotations
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import Experiment
from .. import Entity, Config

class Run(Entity):

    """
    Holds the information about a single run of DeltaSiege
    """

    ALLOW_LOAD = False

    def _init_helper(self, 
        experiment_factory : Experiment.Factory, config : Config, run_name : Optional[str] = None, pool : Optional[Pool] = None,
        mechanism_folder : Optional[Path] = None, classifier_folder : Optional[Path] = None, witness_folder : Optional[Path] = None
    ) -> None:
        """
        Initialization helper which is called by __init__

        Args:
            experiment_factory : Experiment.Factory
                Factory to create experiments
            config : Optional[Config]
                Configuration of the search
            run_name : str
                Name of the run
            pool : Optional[Pool]
                A pool which can be used to distribute the jobs.
                Defaults to a pool of the size equal to self.factory.config.n_jobs
                If self.factory.config.n_jobs is less than or equal to 1, no paralellism is used
            mechanism_folder : Optional[Path]
                The directory where mechanism data is stored.
                Defaults to self.base_folder
            classifier_folder : Optional[Path]
                The directory where classifier data is stored.
                self.base_folder
            witness_folder : Optional[Path]
                The directory where witness data is stored.
                self.base_folder
        """
        
        # Handle defaults
        if mechanism_folder is None:
            mechanism_folder = self.base_folder
        if classifier_folder is None:
            classifier_folder = self.base_folder
        if witness_folder is None:
            witness_folder = self.base_folder

        # Initialize parameters
        self.experiment_factory : Experiment.Factory = experiment_factory
        self.config : Config = config
        self.run_name : Optional[str] = run_name
        self.pool : Optional[Pool] = pool
        self.mechanism_folder : Path = mechanism_folder
        self.classifier_folder : Path = classifier_folder
        self.witness_folder : Path = witness_folder
        self.n_jobs : int = config.n_jobs
        self.complete : bool = False

        # Append experiments to run
        self.experiments : List[Experiment] = []
        for idx in range(self.config.n_experiments):

            experiment_name = f"experiment_{idx}"
            subfolder = None if self.base_folder is None else self.base_folder / experiment_name
            
            self.experiments.append(self.experiment_factory.create(
                experiment_id=idx, 
                mechanism_folder = None if self.mechanism_folder is None else self.mechanism_folder / experiment_name,
                classifier_folder = None if self.classifier_folder is None else self.classifier_folder / experiment_name,
                witness_folder = None if self.witness_folder is None else self.witness_folder / experiment_name,
                base_folder=subfolder, logger=self.logger.subopen()
            ))

    @property
    def data(self):
        data = {}

        # Iterate over all experiments
        for experiment in self.experiments:

            # Make data per experiment
            exp_data = {}

            # Iterate over all trials
            for trial_id, trial in experiment.trials.items():
                exp_data[trial_id] = trial.logger.data
        
            data[experiment.experiment_id] = data
        
        return data

    def run(self) -> Run:
        """
        Runs the single run

        Returns:
            : Run
                Returns itself
        """

        # If completed - do not rerun
        if self.complete:
            return self

        # Open logger with name corresponding to the run name
        with self.logger(self.run_name, timing=True, timing_string=f"run experiment{'' if self.run_name is None else ' ' + self.run_name}"):

            # Open new pool if none is explicitly provided
            if self.pool is None:

                # Only use a multiprocessing if necessary
                if self.n_jobs > 1:
                    pool_ = Pool(self.n_jobs)
                else:
                    pool_ = None
            else:
                pool_ = self.pool

            

            # Run manually
            if pool_ is None:
                results = [self._single_experiment(x) for x in self.experiments]

            # Run in paralell
            else:

                # Run experiments
                results = pool_.map(self._single_experiment, self.experiments)

                # Close pool if created
                if self.pool is None:
                    pool_.close()

            # Load all run experiments if Needed
            for idx, exp in results:

                # If None is returned then the experiment was saved
                if exp is None:
                    self.experiments[idx].load()
                else:
                    self.experiments[idx] = exp

        # Run is completed
        self.complete = True

        # Save results
        if self.store_data:
            self.save()

        return self

    @staticmethod
    def _single_experiment(experiment : Experiment) -> Tuple[int, Optional[Experiment]]:
        """
        Runs an experiment, which is independent from other experiments.
        Trials within the experiment are dependent

        Args:
            experiment_id : int
                Integer id for the experiment
            trial : Trial
                Trial from the optuna optimizer
        
        Returns:
            : 
        """
        experiment.run()

        # Return dynamically
        if experiment.store_data:
            return (experiment.experiment_id, None)
        else:
            return (experiment.experiment_id, None)