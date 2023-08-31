import numpy as np
from pathlib import Path
from typing import Dict, Generator


class Iterator:

    """
    An iterator helper class for iterating over files in the structure storing the data for a data handler
    """

    def __init__(self, offset : int, size : int, batch_size : int, header_data : Dict, base_path : Path) -> None:
        """
        Initialize iterator and set up structure

        Args:
            offset: int
                Index to use in dataset.
            n: int
                Number of samples
            batch_size: int
                Batch size of the returned samples.
            header_data : Dict
                The data in the header used to navigate the structure
            base_path : Path
                The path to the folder where the data is present
        """
        self.offset = offset
        self.size = size
        self.batch_size = batch_size
        self.header_data = header_data
        self.base_path = base_path

        # Test for inconsitency
        if self.offset + self.size > self.header_data["size"]:
            raise RuntimeError(f"Iterator out of bounds. Requesting range [{self.offset}, {self.offset + self.size - 1}]. "
                               f"Dataset has size {self.header_data['size']}")

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Iterate over the data in the rage [self.offset, self.offset + self.size - 1].
        The data is returned as batches of size self.batch_size

        Returns:
            : Generator[np.ndarray, None, None]
                Returns a generator where each element is a Numpy array with the self.batch_size number of samples
                from the data handler which is stored at self.base_path with the header given by self.header_data.
        """
        
        # Set up structure
        current_offset = self.offset
        sizes = [file["size"] for file in self.header_data["files"]]
        filenames = [self.base_path.joinpath(file["filename"]) for file in self.header_data["files"]]

        # Find start position
        current_idx = 0
        while sizes[current_idx] <= current_offset:
            current_idx += 1
            current_offset -= sizes[current_idx]

        # Iterate through the files batchwise
        remaining_total = self.size

        # Load the current file
        current_file = np.load(filenames[current_idx])

        while remaining_total > 0:

            # Get size of current batch
            current_size = min(remaining_total, self.batch_size)

            # Get data from files
            remaining_batch = current_size
            batch_data = []
            while remaining_batch > 0:
                
                # Test if the remainer of the batch can be loaded from a single file
                if remaining_batch <= sizes[current_idx] - current_offset:

                    # Add relevant portion of file
                    batch_data.append(current_file[current_offset : current_offset + remaining_batch])
                    current_offset += remaining_batch
                    remaining_batch -= remaining_batch

                else:
                    
                    # Add reminder of file
                    batch_data.append(current_file[current_offset:])
                    remaining_batch -= sizes[current_idx] - current_offset

                    # Load new file
                    current_idx += 1
                    current_file = np.load(filenames[current_idx])
                    current_offset = 0

            # Get data for batch
            batch = np.concatenate(batch_data, axis=0)

            yield batch

            # Update how many samples need yet to be returned
            remaining_total -= current_size

    def __len__(self) -> int:
        """
        Returns the number of samples which the iterator iterates over in total
        """
        return self.size
