import numpy as np
from pathlib import Path
from typing import Optional

from .stable_classifier import StableClassifier
from deltasiege.dataset.datahandler import DataSource
from deltasiege.logging.logger import Logger


class SklearnClassifier(StableClassifier):
    """
    A classifier for two classes 0 and 1. The classifier is stable w.r.t. numerical rounding noise
    and provides functionality for feature transformation and normalization.
    """

    def _init_helper(
        self, 
        classifier,
        data_source: DataSource,
        config,
    ):
        """
        Creates an abstract stable classifier.

        Args:
            feature_transform: an optional feature transformer to be applied to the input features
            normalize_input: whether to perform normalization of input features (after feature transformation)
        """
        super()._init_helper(
            data_source=data_source, 
            config=config,
        )

        # Save classifier
        self.classifier = classifier

    def _train(self, training_batch_generator):

        X, y = [], []
        for batch in training_batch_generator:
            X.append(batch[0])
            y.append(batch[1])
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        
        if len(X.shape) == 1:
            X = np.atleast_2d(X).T

        self.classifier.fit(X, y)

    def _predict_probabilities(self, batch):
        """
        Compute the probability of class 0 by performing inference on the trained model.
        """

        if len(batch.shape) == 1:
            batch = np.atleast_2d(batch).T
        
        return self.classifier.predict_proba(batch)[:, 0]
