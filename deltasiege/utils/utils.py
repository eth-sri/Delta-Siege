import numpy as np
from typing import Iterator

from . import Witness

def compute_overlap(w1 : Witness , w2 : Witness):

    result = {}

    for key in [w1.a1, w1.a2]:

        n1, n2, n, n_both, n_either = 0, 0, 0, 0, 0
        iterator : Iterator = w1._get_iterator(key)
        for x in iterator:
            x_w1 = w1.is_member(x)
            x_w2 = w2.is_member(x)
            n1 += np.count_nonzero(x_w1)
            n2 += np.count_nonzero(x_w2)
            n_both += np.count_nonzero(x_w1 & x_w2)
            n_either += np.count_nonzero(x_w1 | x_w2)
            n  += x.shape[0]

        result[w1.dataset.mechanism.stringify_input(key)] = {"n1": n1, "n2": n2, "n_both": n_both, "n_either": n_either}
    
    return result
