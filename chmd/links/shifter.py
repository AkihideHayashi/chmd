"""Energy Shifter."""
import numpy as np
import chainer
from chainer import Link
import chainer.functions as F
from sklearn.linear_model import LinearRegression


class EnergyShifter(Link):
    """Energy Shifter."""

    def __init__(self, coef):
        """Initializer."""
        super().__init__()
        dtype = chainer.config.dtype
        self.add_persistent('coef', np.array(coef, dtype))

    def forward(self, ei):
        """Forward."""
        return self.coef[ei]


def count_elements(e, max_e):
    """Count elements."""
    ret = np.zeros(max_e, dtype=np.int64)
    np.add.at(ret, e, 1)
    return ret


def linear_coef_intercept(counts, energies):
    """Auxiliary."""
    model = LinearRegression(fit_intercept=False)
    model.fit(counts, energies)
    return model.coef_
