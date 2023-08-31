from sklearn import preprocessing
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import TransformerMixin

from deltasiege.classifiers.stable_classifier import StableClassifier
from .torch_optimizer_factory import TorchOptimizerFactory
from deltasiege.dataset.datahandler import DataSource
from deltasiege.utils.config import Config


class MultiLayerPerceptron(StableClassifier):
    """
    A feedforward neural network classifier.
    """

    def _init_helper(
      self,
      data_source: DataSource,
      config : Config,
      in_dimensions: int,
      optimizer_factory: TorchOptimizerFactory,
      feature_transform: Optional[TransformerMixin] = None,
      normalize_input: bool = True,
      n_test_batches: int = 0,
      hidden_sizes: Tuple = (10, 5),
      epochs: int = 10,
      regularization_weight: float = 0.001,
      device : str = "cpu", # ToDo: Add CUDA torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    ):
        """
        Creates a feedforward neural network classifier.

        Args:
            in_dimensions: number of input dimensions for the classifier (dimensionality of features)
            optimizer_factory: a factory constructing the optimizer to be used for training
            feature_transform: an optional feature transformer to be applied to the input features
            normalize_input: whether to perform normalization of input features (after feature transformation)
            n_test_batches: number of batches reserved for the test set (non-zero allows to track test accuracy during training)
            hidden_sizes: a tuple (x_1, ..., x_n) of integers defining the number x_i of hidden neurons in the i-th hidden layer
            epochs: number of epochs for training
            regularization_weight: regularization coefficient in [0, 1]
        """
        super()._init_helper(
            data_source=data_source, 
            config=config,
        )

        # General params
        self.feature_transform = feature_transform
        self.normalize_input = normalize_input
        self.normalizer = None  # remember data normalizer, needs to be re-used at prediction time

        # Model parameters
        self.in_dimensions = in_dimensions
        self.optimizer_factory = optimizer_factory
        self.n_test_batches = n_test_batches
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.regularization_weight = regularization_weight

        # initialize torch-specific fields
        self.device = device
        self.criterion = nn.BCELoss()
        self.model = self._get_perceptron_model()
        self.model.to(device=self.device)
        self.dtype = self.model[0].weight.dtype

    def _regularized(self, unregularized):
        """
        Computes the regularized loss.
        """
        l2_reg = torch.tensor(0., requires_grad=True)
        for p in self.model.parameters():
            # l2_reg = l2_reg + (p*p).sum()   # NOTE: computes squared L2 norm (which is differentiable at 0)
            l2_reg = l2_reg + p.abs().sum()
        return unregularized + self.regularization_weight*l2_reg

    def _get_perceptron_model(self):
        """
        Creates the pytorch model.
        """
        # list of layers
        model = []

        # add layers
        previous_size = self.in_dimensions
        for size in self.hidden_sizes:
            model.append(nn.Linear(previous_size, size))
            model.append(nn.ReLU())
            previous_size = size

        # output layer (size 1)
        model.append(nn.Linear(previous_size, 1))
        model.append(nn.Sigmoid())

        # create end-to-end model
        model = nn.Sequential(*model)

        return model

    def _transform(self, x):
        """
        Perform feature transformation.
        """

        if len(x.shape) == 1:
            x = np.atleast_2d(x).T

        # transform input
        if self.feature_transform is not None:
            x = self.feature_transform.transform(x)

        # normalize input
        if self.normalize_input:

            # fit normalizer on first batch
            if self.normalizer is None:
                self.normalizer = preprocessing.StandardScaler().fit(x)
            x = self.normalizer.transform(x)
        
        return x

    @staticmethod
    def _get_test_set(batch_generator, n_test_batches):
        if n_test_batches > 0:
            test_x_list = []
            test_y_list = []
            for _ in range(n_test_batches):
                x, y = next(batch_generator)
                test_x_list.append(x)
                test_y_list.append(y)
            x_test = np.vstack(test_x_list)
            y_test = np.vstack(test_y_list)
            return torch.Tensor(x_test), torch.Tensor(y_test)
        else:
            return None, None

    def _train(self, training_batch_generator):
        
        # Reset normalizer
        self.normalizer = None
        
        # get test set from first n_test_batches training batches
        x_test, y_test = MultiLayerPerceptron._get_test_set(training_batch_generator.__iter__(), self.n_test_batches)

        # get optimizer
        optimizer, scheduler = self.optimizer_factory.create_optimizer_with_scheduler(self.model.parameters())

        # training
        self.model.train()

        # Update with test batches
        with self.logger("train_loop", pbar=True):

            test_x_list = []
            test_y_list = []

            for batch_idx, (x_train, y_train) in enumerate(training_batch_generator):

                # transform input
                x_train = self._transform(x_train)

                # add to test set
                if batch_idx < self.n_test_batches:
                    test_x_list.append(x_train)
                    test_y_list.append(y_train)

                # train dataset
                else:

                    if batch_idx == self.n_test_batches:
                        x_test = torch.Tensor(np.vstack(test_x_list))
                        y_test = torch.Tensor(np.vstack(test_y_list))

                    # convert to tensors
                    x_train = torch.Tensor(x_train).to(device=self.device, dtype=self.dtype)
                    y_train = torch.Tensor(y_train).to(device=self.device, dtype=self.dtype)

                    # run epochs on this batch
                    self._train_one_batch(batch_idx, x_train, y_train, x_test, y_test, optimizer, scheduler)

    def _train_one_batch(
        self, 
        batch_idx: int, 
        x_train, y_train, 
        x_test, y_test, 
        optimizer, scheduler
    ):
        for epoch in range(self.epochs):
            # not really an "epoch" as it does not loop over the whole data set, but only over one batch

            # closure function required for optimizers such as LBFGS that need to compute
            # the gradient of the loss themselves
            def closure():
                # initialize gradients
                optimizer.zero_grad()

                # compute loss (forward pass)
                y_pred = self.model(x_train).squeeze()
                loss = self._regularized(self.criterion(y_pred, y_train))

                # backward pass
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            
            self.logger.append_datapoint(loss_step=loss.item())


            if scheduler is not None:
                scheduler.step()

        desc_values = {"loss": loss.item()}

        # Logg validation set if given
        if self.n_test_batches:
            y_pred_test = self.model(x_test)
            loss_test = self._regularized(self.criterion(y_pred_test.flatten(), y_test.flatten()))
            desc_values["loss_test"] = loss_test.item()

        # Step in the logger
        self.logger.step(**desc_values)

    def _run(self, features):
        """
        Run inference on the trained model.
        """
        
        features = self._transform(features)

        features = torch.Tensor(features).to(device=self.device, dtype=self.dtype)
        self.model.eval()
        y_pred = self.model(features)
        return y_pred

    def _predict_probabilities(self, features):
        """
        Compute the probability of class 0 by performing inference on the trained model.
        """
        y_pred = self._run(features)
        y_pred = y_pred.data.cpu().numpy()
        # want to return probability of class 0 -> must compute the opposite probability
        probs = 1-y_pred.T[0]
        return np.around(probs, decimals=3).flatten()     # round to 3 decimals for numerical stability
