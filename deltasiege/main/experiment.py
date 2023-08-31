from __future__ import annotations
import numpy as np
import optuna
from optuna.trial import Trial as OptunaTrial
from pathlib import Path
import random
import torch
from typing import Dict, Optional

from . import Trial
from .. import Entity, SearchSpace, Config

class Experiment(Entity):

    """
    Holds the information about a single experiment of DeltaSiege
    """

    def _init_helper(self, 
        tiral_factory : Trial.Factory, config : Config, experiment_id : int, search_space : SearchSpace, 
        mechanism_folder : Optional[Path] = None, classifier_folder : Optional[Path] = None, witness_folder : Optional[Path] = None
    ):
        """
        Initialization helper which is called by __init__

        Args:
            tiral_factory : Trial.Factory
                Factory to create trials
            config : Optional[Config]
                Configuration of the search
            experiment_id : int
                Id of the experiment - used to seed
            search_space : SearchSpace
                A search space for optimizing the mechanism which is tested
                Allows to search for suitable (epsilon, delta) values which breaks the mechanism
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
        self.tiral_factory : Trial.Factory = tiral_factory
        self.config : Config = config
        self.experiment_id : int = experiment_id
        self.search_space : SearchSpace = search_space
        self.mechanism_folder : Path = mechanism_folder
        self.classifier_folder : Path = classifier_folder
        self.witness_folder : Path = witness_folder
        self.trials : Dict = [None for _ in range(self.config.n_trials)]
        self.complete : bool = False

    def run(self) -> Experiment:
        """
        Runs a single experiment

        Returns:
            : Run
                Returns itself
        """

        # If completed - do not rerun
        if self.complete:
            return self

        # Seed everything for the experiment
        self._seed_all(self.experiment_id + self.config.seed)   

        # The objective to optimize in the study
        def objective(optuna_trial : OptunaTrial):

            # Identifier
            idx = optuna_trial._trial_id
            trial_name = f"trial_{idx}"
            subfolder = None if self.base_folder is None else self.base_folder / trial_name

            # Get mechanism
            input_pair, mechanism = self.search_space(
                optuna_trial,
                logger=self.logger.subopen(),
                base_folder=None if self.mechanism_folder is None else self.mechanism_folder / trial_name / "mechanism"
            )

            # Handle the folder structure
            trial = self.tiral_factory.create(
                input_pair = input_pair, mechanism = mechanism, 
                classifier_folder = None if self.classifier_folder is None else self.classifier_folder / trial_name,
                witness_folder = None if self.witness_folder is None else self.witness_folder / trial_name,
                base_folder = subfolder, logger=self.logger.subopen()
            )
                
            # Run trial
            trial.run()

            # Add result to final
            self.trials[idx] = trial

            # Return the confident estimate
            return trial.witnesses[trial.best_witness_idx].estimate_c
        
        # Perform the study - minimizing the confident estimate for the witness
        study = optuna.create_study()
        study.optimize(objective, n_trials=self.config.n_trials, n_jobs=self.config.n_trial_jobs)

        # Experiment is completed
        self.complete = True

        # Save results
        if self.store_data:
            self.save()

        return self

    def _seed_all(self, seed : int) -> None:
        """
        Helper to seed all relevant libaries. Is done once for each independent experiment

        Args:
            seed : int
                Seed which is used
        """

        # Seed all relevant libraries
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
