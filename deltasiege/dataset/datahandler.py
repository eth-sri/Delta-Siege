from abc import abstractmethod
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Generator

from .. import Entity

class DataSource(Entity):
    """
    Base class for maniging data sources
    """

    @property
    def base_folder(self) -> Path:
        """
        Path to the base folder for storing the data
        """
        return self.base_folder_

    @base_folder.setter
    def base_folder(self, base_folder : Path) -> None:
        """
        Setter for the base folder

        Args:
            base_folder: Path
                Path to base folder
        """
        
        # Save header if exists
        if hasattr(self, "header"):
            self.header.save()

        # Set up base directory
        self.base_folder_ : Path = base_folder

        # Set up base folder for storage if specified
        if self.store_data:
            self.header : DataSource.Header = DataSource.Header(base_folder=self.base_folder / "header")

    def stringify_input(self, input : Any) -> str:
        """
        Injectively maps input to a string. Default is to use the str() method

        Args:
            input: Any
                Input to the data source
        
        Returns:
            : str
                String which uniquely identifies the input
        """
        return str(input)

    def add_samples(self, input : Any, n : int, batch_size : Optional[int] = None) -> None:
        """
        Add size number samples to the the dataset associated with the specified input

        Args:
            input: Any
                Input to the data source
            n: int
                Number of samples to add
            batch_size:
                Batch size used to create and store the samples.
                Allows to handle if n is too large to fit in memory
        """

        # Handle the case where n <= 0
        if n <= 0:
            return

        # Handle default values for batch_size
        if batch_size is None:
            batch_size = n

        # Iterate untill all samples are added
        remaining_samples = n

        # Initialize sublogger
        n_batches = np.ceil(remaining_samples / batch_size)
        key = self.stringify_input(input)
        desc = f"Generate data for {self.name}({key})"
        with self.logger(f"generate_{key}", pbar=True, pbar_total=n_batches, pbar_desc=desc):

            # Run batchwise until all samples are added
            while remaining_samples > 0:

                # Get number of samples to add
                size = min(remaining_samples, batch_size)

                # Get key from input - unique for each input
                key = self.stringify_input(input)

                # Get new file from header
                filename = self.header.add_file(key, size)

                # Get data
                data = self(input, size)

                # Save data and persist to header
                with open(filename, "wb+") as out_file:
                    np.save(out_file, data)
                
                # Account for added samples
                remaining_samples -= size

                # Step in the logger
                self.logger.step(size=size)

    def get_size(self, input : Any) -> int:
        """
        Get the number of elements stored for a certain input
        """
        return self.header.get_size(self.stringify_input(input))

    
    def get_batches(self, input : Any, n : int, batch_size : Optional[int] = None, idx : Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """
        Get n samples from the data source

        Args:
            input: Any
                The input to the data source
            n: int
                Number of samples
            batch_size: Optional[int]
                Batch size of the returned samples. Defaults to being equal to n, if not specified
            idx: Optional[int]
                Index to use in dataset. Defaults to the last element in the sequence

        Returns:
            : Generator[np.ndarray, None, None]
                Returns a generator where each element is a Numpy array with the batches of samples. 
                The shape of the returned arrays is (B, x1, ..., xk) 
                where each sample from the data source is a Numpy array with shape (x1, ..., xk), and
                B is the size of the batch, which is equal to batch_size for all except potentially the last batch.
        """

        # Handle default values for batch_size
        if batch_size is None:
            batch_size = n
        if idx is None:
            idx = self.header.get_size(self.stringify_input(input))

        # Use storage if activated
        if self.store_data:

            # Handle if more samples need to be added
            key = self.stringify_input(input)
            n_existing = self.header.get_size(key)
            n_overflow = max(0, n + idx - n_existing)
            self.add_samples(input, n_overflow, batch_size)

            # Use Iterator as helper to create Generator
            for x in DataSource.Iterator(idx, n, batch_size, self.header[key], self.header.base_folder):
                yield x
        
        # Otherwise, generate data on the fly
        else:

            # Iterate untill all samples are produced
            remaining_samples = n
            while remaining_samples > 0:

                # Get number of samples to add
                size = min(remaining_samples, batch_size)

                # Yield the data
                yield self(input, size)
            
                # Account for produced samples
                remaining_samples -= size

    @abstractmethod
    def __call__(self, input : Any, n : int) -> np.ndarray:
        """
        Get n samples from the data source

        Args:
            input: Any
                The input to the data source
            n: int
                Number of samples

        Returns:
            : np.ndarray
                Returns a Numpy array with the batches of samples. The shape of the returned arrays is (n, x1, ..., xk) 
                where each sample from the data source is a Numpy array with shape (x1, ..., xk).
        """
        pass

    # Helper classes
    from .header import Header
    from .iterator import Iterator
