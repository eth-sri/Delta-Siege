from __future__ import annotations
from copy import deepcopy
from ctypes import Union
import inspect
import pickle
from pathlib import Path
from typing import Dict, Optional

from ..logging import Logger



class Entity:
    """
    Base class for describing named, parameterized object

    To Do:
        * Make able to save and load
    """

    ALLOW_LOAD = True

    def __init__(self, *args, base_folder : Optional[Path] = None, logger : Optional[Logger] = None, **kwargs) -> None:

        # Set the basic structure
        self.base_folder : Optional[Path] = base_folder
        
        # Use an empty logger if none is given
        if logger is None:
            logger = Logger(Logger.State.DOWN)
        self.logger : Logger = logger

        # Initialize using the object
        if self.ALLOW_LOAD:
            try:
                self.load()
            except Exception as e:
                self._init_helper(*args, **kwargs)
        else:
            self._init_helper(*args, **kwargs)

        # Log the object itself
        # self.log_self()

    @property
    def name(self) -> str:
        """
        The name of the object. Defaults to the programatic class name
        """
        return self.__class__.__name__

    @property
    def store_data(self) -> bool:
        """
        Returns if data is being stored for future usage
        """
        return self.base_folder is not None
    
    @classmethod
    def get_factory(cls, *args, **kwargs) -> Factory:
        """
        Construct a factory for the given type of entity

        Args:
            cls: 
                The type of the entity
            *args:
                Positional arguments
            **kwargs: 
                The key word arguments to be passed when constructing the entity
        """
        return cls.Factory(cls, *args, **kwargs)

    def log_self(self) -> None:
        """
        Log the object itself
        """
        self.logger.add_datapoint(name=self.name, params=vars(self))

    def save(self, folder : Optional[Path] = None, params : Optional[Dict] = None):

        # Handle default arguments
        if folder is None:
            if self.store_data:
                folder = self.base_folder
            else:
                self.logger.raise_exception(FileNotFoundError())

        # Handle the default case where params are implicily given
        if params is None:
            folder = self.__getstate__()
            if not isinstance(folder, Path):
                self.logger.raise_exception(ValueError("__getstate__ must return a Path if storing data"))
        
        else:
            # Make sure directory exists
            self.base_folder.mkdir(exist_ok=True, parents=True)

            # Save self_save_dict and entity_dict
            with open(folder / "params.pkl", "wb+") as f:
                pickle.dump(params, f)

        return folder

    def load(self, folder : Optional[Path] = None):

        # Handle default arguments
        if folder is None:
            if self.store_data:
                folder = self.base_folder
            else:
                self.logger.raise_exception(FileNotFoundError())

        # Save self_save_dict and entity_dict
        with open(folder / "params.pkl", "rb") as f:
            for attr, value in pickle.load(f).items():
                setattr(self, attr, value)
    
    def _init_helper(self, *args, **kwargs):
        """
        Initialization helper which is called by __init__
        """
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        """
        Stringify the 
        """
        return self.name + "(" + ", ".join(["{}={}".format(key, str(value)) for key, value in vars(self).items()]) + ")"

    def __hash__(self) -> int:
        """
        Hashes the parameters defining the the mechanism uniqely
        """
        return hash(tuple(vars(self).values()))

    def __getstate__(self, params : Optional[Dict] = None):

        # If params is not set, make state based on all members
        if params is None:
            params = vars(self)
        
        # If the folder is given, save and represent using the path
        if self.base_folder is not None:

            # Save the state
            path = self.save(params=params)

            return path
        
        else:
            return params

    def __setstate__(self, state : Union[Path, dict]):

        if isinstance(state, Path):
            self.load(state)
            
        elif isinstance(state, dict):
            for attr, value in state.items():
                setattr(self, attr, value)
        
        else:
            self.logger.raise_exception(ValueError("state must be path or dict"))


class Factory(Entity):
    """
    A factory class for passing around entities which can be initialized individually
    """

    def _init_helper(self, cls, *args, **kwargs) -> None:
        """
        Construct a factory which creates instances of a fixed type using fixed arguments.

        Args:
            cls: 
                The type of the entity
            *args: Any
                Positional arguments
            **kwargs: Any
                The key word arguments to be passed when constructing the entity
        """
        self.cls = str(cls).split("'")[1]
        self.args = args
        self.kwargs = kwargs

    def copy(self, **kwargs) -> Factory:
        """
        Helper to make copy of the factory

        Args:
          kwargs : **Any
              Keyword arguments used to overwrite any of the arguments provided upon initialization.
              Must correspond to the arguments in the initialization
        """

        # Make copy
        args_ = deepcopy(self.args)
        kwargs_ = deepcopy(self.kwargs)

        # Update based on kwargs
        kwargs_.update(kwargs)

        return self.get_cls().get_factory(*args_, **kwargs_)

    def create(self, **kwargs) -> Entity:
        """
        Get an entity with the type cls and the saved arguments, 
        as well as the provided key word arguments

        Args:
            kwargs:
                Key word arguments
        
        Returns:
            : Entity
                A witness with the given parameters
        """

        # Find signature and name of self variable in the initializer
        signature = inspect.signature(self.get_cls()._init_helper)
        kwarg_names = list(signature._parameters.keys())
        self_name = kwarg_names[0]
        
        # Update based on kwargs
        kwargs_ = deepcopy(self.kwargs)
        kwargs_.update({key: value for key, value in kwargs.items() if key in kwarg_names[1:]})

        # Bind arguments 
        bound_args = signature.bind(*self.args, **kwargs_, **{self_name: None})
        bound_args.apply_defaults()
        bound_arguments_dict = bound_args.arguments

        # Pop self argument
        bound_arguments_dict.pop(self_name)

        # Update based on kwargs
        bound_arguments_dict.update(kwargs)

        return self.get_cls()(**bound_arguments_dict)

    def get_cls(self):
        """
        Helper to convert the string of self.cls to the actual class

        Returns:
            : type
                The class corresponding to self.cls
        """
        components = self.cls.split('.')
        cls = __import__(components[0])
        for comp in components[1:]:
            cls = getattr(cls, comp)
        
        return cls

# Regester the Factory
Entity.Factory = Factory
