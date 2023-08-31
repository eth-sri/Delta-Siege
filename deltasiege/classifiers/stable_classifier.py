from abc import abstractmethod
import numpy as np
from typing import Any, Dict, Generator, Optional, Tuple

from deltasiege.dataset.datahandler import DataSource
from deltasiege.logging.logger import Logger


class StableClassifier(DataSource):
    """
    A classifier for two classes 0 and 1. The classifier is stable w.r.t. numerical rounding noise
    and provides functionality for feature transformation and normalization.
    """

    def _init_helper(
        self, 
        data_source: DataSource,
        config
    ):
        """
        Creates an abstract stable classifier.

        Args:
            feature_transform: an optional feature transformer to be applied to the input features
            normalize_input: whether to perform normalization of input features (after feature transformation)
        """
        super()._init_helper()

        # General parameters
        self.config = config
        self.train_inputs = (None, None)
        
        # Parameters related to the datasource
        self.data_source : DataSource = data_source
        self._used_samples : Dict = {}

    def stringify_input(self, input : Any) -> str:
        """
        Injectively maps input to a string. Use the method from the original data source

        Args:
            input: Any
                Input to the data source
        
        Returns:
            : str
                String which uniquely identifies the input
        """
        return self.data_source.stringify_input(input)

    def get_data(self, input : Any, n : int, batch_size : Optional[int] = None, idx : Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """
        Get data from the data source. Same interface as get_batches

        Args:
            input: Any
                The input to the data source
            n: int
                Number of samples
            batch_size: Optional[int]
                Batch size of the returned samples. Defaults to being equal to n, if not specified
            idx: Optional[int]
                Index to use in data source. 
                Defaults to the number of samples drawn from self.data_source, i.e. self._used_samples[self.stringify_input(input)]

        Returns:
            : Generator[np.ndarray, None, None]
                Returns a generator where each element is a Numpy array with the batches of samples. 
                The shape of the returned arrays is (B, x1, ..., xk) 
                where each sample from the data source is a Numpy array with shape (x1, ..., xk), and
                B is the size of the batch, which is equal to batch_size for all except potentially the last batch.
        """

        # Get key
        key = self.stringify_input(input)

        # Handle default values for idx
        if idx is None:

            if not key in self._used_samples:
                self._used_samples[key] = 0

            idx = self._used_samples[key]
        
        # Update number of samples used
        self._used_samples[key] = max(self._used_samples[key], idx + n)

        return self.data_source.get_batches(input, n, batch_size, idx)

    def train(self, input_pair : Tuple[Any]):
        """
        Trains the classifier.

        Args:
            training_batch_generator: generator for batches containing training data.

        Note:
            Each batch returned by the generator must be a tuple (x, y) where
            x: nd array of shape (n_samples, feature_dimensions) containing features;
            y: 1d array of shape (n_samples, ) containing labels in {0, 1}
        """
        
        # Get the training data for each input
        a1, a2 = input_pair
        a1_iterator = self.get_data(a1, self.config.n_train, self.config.batch_size)
        a2_iterator = self.get_data(a2, self.config.n_train, self.config.batch_size)

        # Combine the iterators
        train_gen = (
            (
                np.concatenate([x1, x2], axis=0), 
                np.concatenate([np.zeros(x1.shape[0]), np.ones(x2.shape[0])])
            ) for x1, x2 in zip(a1_iterator, a2_iterator)
        )

        # Train the model
        with self.logger("train", timing=True, timing_string="training classifier"):
            self._train(train_gen)

        # Set train pair
        self.train_inputs = tuple(self.stringify_input(a) for a in input_pair)

    def __call__(self, input : Any, n : int):
        """
        Computes the probabilities p(y = 0 | x) for a vector x based on the trained classifier.

        Args:
            x: nd array of shape (n_samples, feature_dimensions)

        Returns:
            1d array of shape (n_samples, )
        """

        # Get data from data source - make not print in sub loop
        if self.data_source.logger.get_state() is Logger.State.DOWN:
            sub_logger_state = Logger.State.DOWN
        else:
            sub_logger_state = Logger.State.SILENT
        
        with self.data_source.logger(state = sub_logger_state):
            x = next(self.get_data(input, n))

        # Predict the probabilities
        p = self._predict_probabilities(x)

        return p

    @abstractmethod
    def _train(self, training_batch_generator):
        """
        Trains the classifier on feature-transformed and normalized data.

        Args:
            training_batch_generator: generator for batches containing training data.

        Note:
            Each batch returned by the generator must be a tuple (x, y) where
            x: nd array of shape (n_samples, feature_dimensions) containing features;
            y: 1d array of shape (n_samples, ) containing labels in {0, 1}
        """
        pass

    @abstractmethod
    def _predict_probabilities(self, n):
        """
        Computes the probabilities p(y = 0 | x) for a (feature-transformed and normalized)
        vector x based on the trained classifier.

        Args:
            x: nd array of shape (n_samples, feature_dimensions)

        Returns:
            1d array of shape (n_samples, )
        """
        pass
