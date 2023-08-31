from __future__ import annotations
from pathlib import Path
from typing import Dict, Union

from ..attack import Witness
from ..classifiers import StableClassifier
from ..logging.logger import *
from ..mechanism import Mechanism
from ..utils import Config

from .. import Entity

class Trial(Entity):

    """
    Holds the information about a single run of DeltaSiege
    """

    def _init_helper(
        self, 
        config : Config, input_pair : Tuple[Any, Any], mechanism : Mechanism, 
        classifier_factories : Union[StableClassifier.Factory, List[StableClassifier.Factory]], witnesses_factories : Union[Witness.Factory, List[Witness.Factory]],
        classifier_folder : Optional[Path] = None, witness_folder : Optional[Path] = None
    ) -> None:
        """
        Initialization helper which is called by __init__

        Args:
            config : Optional[Config]
                Configuration of the search
            input_pair : Tuple[Any, Any]
                Input to mechanism
            mechanism : Mechanism
                Mechanism to be audited
            classifier_factories : Union[StableClassifier.Factory, List[StableClassifier.Factory]]
                Factories to create classifiers
            witnesses_factories : Union[Witness.Factory, List[Witness.Factory]]
                Factories to create witnesses
            classifier_folder : Optional[Path]
                The directory where classifier data is stored.
                self.base_folder
            witness_folder : Optional[Path]
                The directory where witness data is stored.
                self.base_folder
        """

        # Handle to be lists
        if not isinstance(classifier_factories, list):
            classifier_factories = [classifier_factories]            
        if not isinstance(witnesses_factories, list):
            witnesses_factories = [witnesses_factories]

        # Handle defaults
        if classifier_folder is None:
            classifier_folder = self.base_folder
        if witness_folder is None:
            witness_folder = self.base_folder

        # Initialize parameters
        self.config : Config = config
        self.input_pair : Tuple[Any, Any] = input_pair
        self.mechanism : Mechanism = mechanism
        self.classifier_factories : List[StableClassifier.Factory] = classifier_factories
        self.witnesses_factories : List[Witness.Factory] = witnesses_factories
        self.classifier_folder : Path = classifier_folder
        self.witness_folder : Path = witness_folder
        self.classifiers : List[StableClassifier] = None
        self.witnesses : List[Witness] = None
        self.best_witness_idx : int = None
        self.complete : bool = False

    def run(self) -> Trial:
        """
        Runs a single trial
        """

        # If completed - do not rerun
        if self.complete:
            return self
        
        # Train all classifiers
        with self.logger(timing=True, timing_string="to initialize the classifiers"):
            self._init_classifiers()
        
        # Train all classifiers
        with self.logger(timing=True, timing_string="to initialize the witnesses"):
            self._init_witnesess()

        # Create all possible witnesses
        with self.logger(timing=True, timing_string="to evaluate the witnesses"):
            
            # Compare the different witnesses
            self._run_estimation(Witness.State.Check)
            self.best_witness_idx = min(range(len(self.witnesses)), key=lambda x : self.witnesses[x])
            best_witness = self.witnesses[self.best_witness_idx]

            # Compute either all
            if self.config.full_run_all:
                self._run_estimation(Witness.State.Final)
            else:
                best_witness.state = Witness.State.Final
                best_witness.compute_estimate()

        # Trial is completed
        self.complete = True

        # Save if required
        if self.store_data:
            self.save()
        
        # If either of the subclasses require saving, do this
        else:
            
            # Handle mechansim
            if self.mechanism.store_data():
                self.mechanism.save()

            # Handle classifiers
            if not self.witness_folder is None:
                for w in self.witnesses:
                    w.save()
            
            # Handle witnesses
            if not self.witness_folder is None:
                for w in self.witnesses:
                    w.save()

        return self

    def _run_estimation(self, state : Witness.State) -> None:
        """
        Helper compute the estimates for all witnesses

        Args:
            state : Witness.State
                State which all witnesses are assigned to before running the estimation
        """
        for w in self.witnesses:
            w.state = state
            w.compute_estimate()

    def _init_classifiers(self) -> None:
        """
        Tries to load all classifiers from the given factories. If not possivle, retrain
        """

        # Rest classifiers
        self.classifiers = []

        for i, factory in enumerate(self.classifier_factories):

            # Create the classifier
            classifier = factory.create(
                data_source=self.mechanism, 
                config=self.config, 
                base_folder=None if self.classifier_folder is None else self.classifier_folder / f"classifier_{i}",
                logger=self.logger.subopen(),
            )

            # Try to load model directly
            if {classifier.stringify_input(a) for a in self.input_pair} != {a for a in list(classifier.train_inputs)}:

                # Get iterators from mechanism
                classifier.train(self.input_pair)

            # Add classifier
            self.classifiers.append(classifier)

    def _init_witnesess(self) -> None:
        """
        Initialize all the witnesses
        """

        # Rest the classifier
        self.witnesses = []

        # Get witnesses
        a1, a2 = self.input_pair

        # Add all witnesses - add both (a1, a2) and (a2, a1) as input pairs
        for classifier in self.classifiers:
            for factory in self.witnesses_factories:
                for a1_, a2_ in [(a1, a2), (a2, a1)]:
                    w = factory.create(
                        a1=a1_, a2=a2_, 
                        classifier=classifier, 
                        config=self.config, mechanism=self.mechanism, 
                        base_folder=None if self.witness_folder is None else self.witness_folder / f"witness_{len(self.witnesses)}",
                        logger=self.logger.subopen()
                    )

                    # Add the witness
                    self.witnesses.append(w)
