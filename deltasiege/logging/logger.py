from __future__ import annotations
from enum import Enum
from pathlib import Path
import pickle
import time
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Tuple

class Logger:

    """
    Logger which keeps track of information flow during execution
    """

    class State(Enum):

        """
        Helper class for indicating the current state of the logger

        The states are:
        * VERBOSE: All functions are activated
        * SILENT: No printing to the terminal
        * DOWN: No function except handling exception or setting/getting the state
        """
        VERBOSE = 0
        SILENT = 1
        DOWN = 2

    def __init__(self, state : State = State.VERBOSE) -> None:
        """
        Initializer logger

        Args:
            state : State
                The verbosity level used by the logger.
                Defaults to State.VERBOSE, which is the most verbose level
        """

        # Data stored
        self.data = {}
        self.curr, self.prev = self.data, []

        # Helper arguments
        self.pbar = None
        self.timing_start : List[float] = []
        self.timing_string : List[str] = []
        self.states : List[Logger.State] = []

        # Helper arguments
        self._args : Tuple = ()
        self._kwargs : Dict = {}

        # Set state
        self.set_state(state)
        
    def load(self, folder : Path):
        """
        Load Logger

        Args:
          folder: Optional[Path]
              Directory where the model is saved.
              Defaults to self.base_folder
        """

        # To Do
        # Load and save full as an entity

        # Load all data
        filename = folder / "logger.pkl"

        if filename.exists():
            with open(filename, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = {}
        
        # Reset all arguments
        self.curr, self.prev = self.data, []

        # Helper arguments
        self.pbar = None
        self.timing_start : List[float] = []
        self.timing_string : List[str] = []
        self.states : List[Logger.State] = []

        # Helper arguments
        self._args : Tuple = ()
        self._kwargs : Dict = {}

        # Set state
        if not hasattr(self, "state"):
            setattr(self, "state", Logger.State.VERBOSE)

        return folder

    def save(self, folder : Path):
        """
        Save the Logger

        Args:
            folder : Path
                Folder where to save the entity
                Defaults to self.base_folder
        """

        # To Do
        # Load and save full as an entity

        # Make sure directory exists
        folder.mkdir(exist_ok=True, parents=True)

        # Save all data
        with open(folder / "logger.pkl", "wb+") as f:
            pickle.dump(self.data, f)

    def mergo_into(self, logger : Logger) -> None:
        """
        Update current structure to reflect logger

        Returns:
            : Logger
                The logger to merge into self
        """

        def helper(d1, d2):
            for key, value in d1.items():
                if key in d2 and isinstance(value, dict):
                    helper(d1[key], d2[key])
                else:
                    d2[key] = value

        helper(logger.data, self.data)

    def get_state(self) -> State:
        """
        Returns the current state of the logger

        Returns:
            : State
                The current state of the logger
        """
        return self.state_

    def set_state(self, state : State) -> None:
        """
        Set the current state of the logger

        Args:
            state: State
                The new state of the logger
        """        
        if not isinstance(state, Logger.State):
            raise ValueError("State must be of type Logger.State")

        
        # Close pbar if activated and the logger is turned of to be silent
        if (state is Logger.State.SILENT or state is Logger.State.DOWN) and self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        
        # Set state
        self.state_ = state

    def open(self, key : Optional[str] = None, timing : bool = False, timing_string : Optional[str] = None, state : Optional[State] = None) -> Logger:
        """
        Open substructure with the possibility to revert by closing

        Args:
            key: Optional[str]
                Key to the substructure. If none provided, no substructure is opened
            timing: bool
                If true, the time until closing is logged
            timing_string: Optional[str]
                If given provies a description to the timing step which is printed
                Upon entering the substructure the logger prints: "Started <timing_string>..."
                Upon closing the substructure the logger prints: "Finished <timing_string> after <elapsed> seconds..."
            state: Optional[State]
                If given then the state is set to that state when opening the sub structure.
                When closing the substructure, the state is reset to the original level
        """

        # No action if the logger is down
        if self.get_state() is Logger.State.DOWN:
            return self
        
        # Create stucture
        self.prev.append(self.curr)

        # Handle if a new subfolder is opened
        if isinstance(key, str):
            if not key in self.curr:
                self.curr[key] = {}
            self.curr = self.curr[key]
        
        # If key is None - nothing is done
        elif key is None:
            pass
        
        # If neither string nor None, an exception is raised
        else:
            self.raise_exception(ValueError(f"key({key}) should be a string or None"))
        
        # Handle states
        self.states.append(self.get_state())

        if state is not None:
            self.set_state(state)

        # Handle timing
        if timing:
            self.timing_start.append(time.time())

            # Print timing string
            if not timing_string is None:
                self.print(f"Started {timing_string}...")

        else:
            self.timing_start.append(None)
        
        # String associated with the timing
        self.timing_string.append(timing_string)

        return self

    def close(self) -> Logger:        
        """
        Close the opened substructure

        Returns:
            : Logger
                Returns itself
        """

        # No action if the logger is down
        if self.get_state() is Logger.State.DOWN:
            return

        # To Do: Fix this quick fix
        try:
            # Handling timing
            start = self.timing_start.pop()
            end = time.time()
            timing_string = self.timing_string.pop()

            # Append timing data point and print string
            # if given when opening the substructure
            if not start is None:
                elapsed = end - start
                self.append_datapoint(timing=elapsed)

                if timing_string is not None:                
                    self.print(f"Finised {timing_string} after {elapsed} seconds...")

            # Handling data structure
            self.curr = self.prev.pop()
        except:
            pass

        return self

    def subopen(self, key : Optional[str] = None) -> Logger:
        """
        Open and return a new substructure which is a shallow copy of the original logger

        Args:
            key: Optional[str]
                Key to the substructure. If none provided, no substructure is opened
        """

        # No action if the logger is down
        if self.get_state() is Logger.State.DOWN:
            return self
  
        # Create new logger in the substructure
        # new_logger = deepcopy(self)
        # return new_logger.open(key)

        return Logger(self.get_state()).open(key)

    def step(self, **desc_kwargs) -> None:
        """
        Perform a step when the logger is in a substructure with a loop

        Args:
            desc_kwargs: **Any
                Key-value pairs to print as post fixes and also append as datapoints
        """

        # No action if the logger is down
        if self.get_state() is Logger.State.DOWN:
            return

        # If pbar is given - set the description to the postfix
        if not self.pbar is None:
            self.pbar.set_postfix(desc_kwargs)
            self.pbar.update(1)
        
        # Add all datapoints
        self.append_datapoint(**desc_kwargs)

    def add_datapoint(self, **kwargs) -> None:
        """
        Add data points to the structur

        Args:
            kwargs: **Any
                Key-value pairs to add as datapoints
        """

        # No action if the logger is down
        if self.get_state() is Logger.State.DOWN:
            return

        # Add value directly
        for key, value in kwargs.items():
            self.curr[key] = value

    def append_datapoint(self, **kwargs) -> None:
        """
        Add data points to the structur

        Args:
            kwargs: **Any
                Key-value pairs to append as datapoints
        """

        # No action if the logger is down
        if self.get_state() is Logger.State.DOWN:
            return
        
        for key, value in kwargs.items():

            # Ensure that is list
            if not key in self.curr:
                self.curr[key] = []

            # Append value
            self.curr[key].append(value)

    def raise_exception(self, e : Exception) -> None:
        """
        Raise an exception to the logger

        Args:
            e: Exception
                Exception to raise
        """        
        raise e

    def print(self, string : str) -> None:
        """
        Print the string to logger

        Args:
            string: str
                String to print 
        """

        # No action if the logger is down or silent
        if self.get_state() is Logger.State.SILENT or self.get_state() is Logger.State.DOWN:
            return

        print(string, flush=True)
    
    def _enter_helper(
        self, 
        key : Optional[str] = None, 
        timing : bool = False,
        timing_string: Optional[str] = None,
        state : Optional[State] = None,
        pbar : bool = False, 
        pbar_total : Optional[int] = None, 
        pbar_desc : Optional[str] = None,
        *args: Any, **kwargs: Any
    ) -> None:
        """
        Open substructure with the possibility to revert by closing

        Args:
            key: Optional[str]
                Key to the substructure. If none provided, no substructure is opened
            timing: bool
                If true, the time until closing is logged
            timing_string: Optional[str]
                If given provies a description to the timing step which is printed
                Upon entering the substructure the logger prints: "Started <timing_string>..."
                Upon closing the substructure the logger prints: "Finished <timing_string> after <elapsed> seconds..."
            state: Optional[State]
                If given then the state is set to that state when opening the sub structure.
                When closing the substructure, the state is reset to the original level
            pbar: bool
                Indicates wether to initialize a tqdm bar upon entering the structure
            pbar_total : Optional[int]
                Total number of iterations for the tqdm pbar
            pbar_desc : Optional[str]
                Description for the tqdm pbar            
        """

        # Open the substructure
        self.open(key, timing, timing_string, state)

        # Allow for a progressbar in the current section
        if pbar and self.get_state() is Logger.State.VERBOSE:
            self.pbar = tqdm(total=pbar_total, leave=True)
            self.pbar.set_description(pbar_desc)

    def __call__(self, *args: Any, **kwargs: Any) -> Logger:
        """
        Helper for saving arguments before entering

        Args:
            args: *Any
                Positional arguments passed on to self._enter_helper
            kwargs: **Any
                Key word arguments passed on to self._enter_helper
        """
        self._args = args
        self._kwargs = kwargs
        return self

    def __enter__(self) -> None:
        """
        Enter a substructure by using argments which have been saved by __call__
        """
        self._enter_helper(*self._args, **self._kwargs)

    def __exit__(self, *exc_args) -> None:
        """
        Close a substructure upon exiting
        """

        # Close pbar if activated
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

        # Close the substructure
        self.close()
