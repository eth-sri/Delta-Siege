import numpy as np
from sklearn.base import TransformerMixin


class BitPatternFeatureTransformer(TransformerMixin):
    """
    Use the bit-representation of a 1d 64-bit floating point number as feature.
    """
    def __init__(self, merge_into_last = True) -> None:
        super().__init__()
        self.merge_into_last = merge_into_last

    def fit(self, X : np.ndarray, y : np.ndarray):
        return self

    def transform(self, X  : np.ndarray):
        assert(X.dtype == np.float64 or X.dtype == np.int64)   # must be 64 bit floats (double precision)

        # Use helper function
        int_to_bin = lambda y: [c == "1" for c in "{:0>64b}".format(y)]

        # Preserve shape with bit-dimension appended at end
        if self.merge_into_last:
            shape = list(X.shape)
            shape[-1] *= 64
            shape = tuple(shape)
        else:
            shape = X.shape + (64,)

        out = np.array(
            list(
                map(int_to_bin, X.view(dtype=np.uint64).flatten())
            )
        ).reshape(shape)

        return out
