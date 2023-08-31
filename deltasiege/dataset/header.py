from pathlib import Path
from typing import Dict, List

from .. import Entity

class Header(Entity):

    """
    Header class which handels the file structure for a Data handler
    """

    def _init_helper(self) -> None:
        """
        Initialize the header
        """
        self.data : Dict = {}
        self.description : Dict = {}

    def add_description(self, description : Dict) -> None:
        """
        Add or overwrite the description of the data handler

        Args:
            description : Dict
                A dictionary which discribes the data handler
        """
        self.description = description

        # Store changes
        self.save()

    def add_file(self, key : str, size : int) -> Path:
        """
        Add new file to hold size datapoints. Updates header structure.
        Returns new filename

        Args:
            key : str
                A key which describes the type of file which is being added.
                For a data handler, this corresponds to the stringified version of the input.
            size : int
                Number of samples which are being added

        Returns:
            : Path
                Path to the new file being added
        """

        # Count number of previous files and add new file
        count = len(self[key]["files"])
        filename = f"{key}_{count}.npy"

        # Add new file
        self[key]["files"].append({
            "filename": filename,
            "size": size
        })
        self[key]["size"] += size

        # Store changes
        self.save()

        return self.base_folder.joinpath(filename)

    def get_size(self, key : str) -> int:
        """
        Get the number of elements stored for a certain key

        Args:
            key : str
                A key which describes the type of file which is being added.
                For a data handler, this corresponds to the stringified version of the input.

        Returns:
            : int
                The number of samples in the structure associated with the key
        """

        # Test if data is present
        if key not in self:
            size = 0
        else:
            size = self.data[key]["size"]
        
        return size

    def __getitem__(self, key : str) -> Dict:
        """
        Returnes a the header information assosiated with a key.
        If the key is not already added to the header, an empty list is returned

        Args:
            key : str
                A key which describes the type of file which is being added.
                For a data handler, this corresponds to the stringified version of the input.

        Returns:
            : Dict
                A dictionary containing the header information assosiated the key
        """

        # Test if present - if not initialize            
        if not key in self:
            self.data[key] = {"size": 0, "files": []}
        
        return self.data[key]

    def __contains__(self, key : str) -> bool:
        """
        Test if key is contained in the header

        Args:
            key : str
                A key which describes the type of file which is being added.
                For a data handler, this corresponds to the stringified version of the input.

        Returns:
            : bool
                A boolean value indicating wether the key is already present in the structure
        """
        return key in self.data
