from typing import Dict, Sequence, Tuple

from deltasiege.mechanism import Mechanism

class SearchSpace:
    pass

class SimpleSampler(SearchSpace):

    def __init__(
        self,
        mechanism : Mechanism,
        input_pairs : Sequence,
        fixed_kwargs : Dict = {},
        **hyper_params_range : Tuple[float]
    ) -> None:
        self.mechanism = mechanism
        self.input_pairs = input_pairs
        self.fixed_kwargs = fixed_kwargs
        self.hyper_params_range = hyper_params_range

    def __call__(self, trial, **fixed_kwargs):
        
        # Select input pair statically, independent from mechanism
        input_pair = self.input_pairs[
            trial.suggest_int("input_pair", 0, len(self.input_pairs) - 1)
        ]

        # Select hyperparameters for mechanism
        kwargs = {
            key: trial.suggest_float(key, *value) 
            for key, value in self.hyper_params_range.items()
            if key not in fixed_kwargs
        }
        kwargs.update(fixed_kwargs)
        
        return input_pair, self.mechanism(**kwargs, **self.fixed_kwargs)
