from pathlib import Path
from multiprocessing import Pool
from typing import Optional, List, Union

from . import Run, Trial, Experiment
from ..utils import Config
from ..attack import Witness
from ..classifiers import StableClassifier
from ..logging import Logger

from ..ddsampler import SearchSpace

class DeltaSiege:

    """
    Delta Siege: A frame work for auditing differential privacy claims
    """

    def __init__(
        self,
        search_space : Optional[SearchSpace] = None,
        classifier_factories : Optional[Union[StableClassifier.Factory, List[StableClassifier.Factory]]] = None,
        witnesses_factories : Optional[Union[Witness.Factory, List[Witness.Factory]]] = None,
        config : Optional[Config] = None,
        logger : Optional[Logger] = None,
        base_folder : Optional[Path] = None,
        mechanism_folder : Optional[Path] = None, 
        classifier_folder : Optional[Path] = None, 
        witness_folder : Optional[Path] = None
    ) -> None:
        """
        Initalize the DeltaSiege procedure with parameters which will be used upon running

        Args:
            search_space : Optional[SearchSpace]
                A search space for optimizing the mechanism which is tested
                Allows to search for suitable (epsilon, delta) values which breaks the mechanism
            classifier_factories : Optional[Union[StableClassifier.Factory, List[StableClassifier.Factory]]]
                A factory of the classifier or a list of such factories
                Allows to create indepentendt classifiers
            witnesses_factories : Optional[Union[Witness.Factory, List[Witness.Factory]]]
                A factory of the witness or a list of such factories
                Allows to create indepentendt witnesses
            config : Optional[Config]
                Configuration of the search
            logger : Optional[Logger]
                The logger which is used to log all results
                If non is provided, a logger which is DOWN is used, i.e. nothing is logged
            base_folder : Optional[Path]
                The directory where data is stored during the auditing
                If none is provided, nothing is stored
            mechanism_folder : Optional[Path]
                The directory where mechanism data is stored
            classifier_folder : Optional[Path]
                The directory where classifier data is stored
            witness_folder : Optional[Path]
                The directory where witness data is stored
        """

        # Save all parameters in a factories allowing to easily share them
        # between different subprocesses
        self.trial_factory : Trial.Factory = Trial.get_factory(
            config=config,
            classifier_factories=classifier_factories,
            witnesses_factories=witnesses_factories,
        )
        self.experiment_factory = Experiment.get_factory(config=config, search_space=search_space)
        self.run_factory = Run.get_factory(config=config, mechanism_folder=mechanism_folder, classifier_folder=classifier_folder, witness_folder=witness_folder)

        self.logger=logger
        self.base_folder=base_folder

    def run(self, run_name : Optional[str] = None, pool : Optional[Pool] = None, **kwargs) -> Logger:
        """
        Run the auditing procedure

        Args: 
            run_name : Optional[str]
                Name of the experiment. If already used experiment, the results are overwritten
            pool : Optional[Pool]
                A pool which can be used to distribute the jobs.
                Defaults to a pool of the size equal to self.factory.config.n_jobs
                If self.factory.config.n_jobs is less than or equal to 1, no paralellism is used
            kwargs : **Any
                Keyword arguments used to overwrite any of the arguments provided upon initialization.
                Only overwrite for this experiment
        
        Returns:
            : Logger
                The logger with all experiment results for the run
        """

        # Assign the arguments to the right factory
        trial_args = {key: value for key, value in kwargs.items() if key in list(Trial._init_helper.__code__.co_varnames)}
        experiment_args = {key: value for key, value in kwargs.items() if key in list(Experiment._init_helper.__code__.co_varnames)}
        run_args = {key: value for key, value in kwargs.items() if key in list(Run._init_helper.__code__.co_varnames)}

        # Update the trial factory
        trial_factory = self.trial_factory.copy(**trial_args)

        # Update the trial factory
        experiment_factory = self.experiment_factory.copy(tiral_factory=trial_factory).copy(**experiment_args)

        # Update the run factory
        run_factory = self.run_factory.copy(
            experiment_factory=experiment_factory, 
            run_name=run_name, pool=pool, 
        ).copy(**run_args)

        # Get run
        run_folder = self.base_folder if run_name is None or self.base_folder is None else self.base_folder / run_name
        run = run_factory.create(base_folder=run_folder, logger=self.logger.subopen())
        run.run()

        return run
