"""Energy Shifter."""
import numpy as np
import chainer
from chainer import Link
import chainer.functions as F
from sklearn.linear_model import LinearRegression


class EnergyShifter(Link):
    """Energy Shifter."""

    def __init__(self, num_elements):
        """Initializer."""
        super().__init__()
        self.num_elements = num_elements
        self.add_persistent('coef',
                            np.zeros(num_elements, dtype=chainer.config.dtype))
        self.add_persistent('intercept', np.array(0.0))

    def setup(self, elements, energies):
        """Set up coef and intercept."""
        coef, intercept = linear_coef_intercept(
            elements, energies, self.num_elements)
        self.coef[...] = np.array(coef, dtype=chainer.config.dtype)
        self.intercept[...] = intercept

    def forward(self, ei, i1):
        """Forward."""
        return linear_predict(self.coef, self.intercept, ei, i1, xp=self.xp)


def count_elements_noconcat(elements, number_elements, xp=np):
    """Auxiliary."""
    return xp.array([F.scatter_add(xp.zeros(number_elements, dtype=xp.int32),
                                   e, xp.ones(len(e), xp.int32)).data
                     for e in elements])


def count_elements_concat(e1, i1, number_elements, xp=np):
    """Auxiliary."""
    n = (i1 * number_elements) + e1
    seed = xp.zeros(number_elements * len(xp.unique(i1)), dtype=i1.dtype)
    return F.scatter_add(seed, n, xp.ones_like(i1)).data


def linear_predict(coef, intercept, ei, i1, xp=np):
    """Auxiliary."""
    n_batch = len(xp.unique(i1))
    z = xp.zeros(n_batch, dtype=chainer.config.dtype)
    return F.scatter_add(z, i1, coef[ei]).data + intercept


def linear_coef_intercept(elements, energies, number_elements):
    """Auxiliary."""
    model = LinearRegression()
    model.fit(count_elements_noconcat(elements, number_elements), energies)
    return model.coef_, model.intercept_
