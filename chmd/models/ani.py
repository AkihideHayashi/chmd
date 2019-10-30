"""ANI-1."""
import numpy as np
import chainer
from chainer import Chain, ChainList
import chainer.links as L
import chainer.functions as F
from chmd.links.ani import ANI1AEV
from chmd.activations import gaussian
from sklearn.linear_model import LinearRegression


class AtomNN(ChainList):
    """Auxiliary function for AtomWiseNN."""

    def __init__(self, n_layers, act):
        """Initilize."""
        super().__init__()
        for nl in n_layers:
            self.add_link(L.Linear(None, nl, nobias=False))
        self.act = act

    def forward(self, x):
        """Compute for each atoms."""
        h = self[0](x)
        for l in self[1:]:
            h = l(self.act(h))
        return h


class AtomWiseNN(ChainList):
    """NN part of ANI-1."""

    def __init__(self, n_layers, act):
        """Initialize."""
        super().__init__()
        for nl in n_layers:
            self.add_link(AtomNN(nl))

    def forward(self, x, e):
        """Select and apply NN for each atoms."""
        dtype = chainer.config.dtype
        out = F.concat([n(x)[None, :, :] for n in self], axis=0)
        n = out.shape[0]
        condition = self.xp.arange(n)[:, None] == e[None, :]
        zeros = self.xp.zeros(out.shape, dtype=dtype)
        ret = F.where(condition[:, :, None], out, zeros)
        return F.sum(ret, axis=0)


class ANI1(Chain):
    """ANI-1 energy calculator."""

    def __init__(self, num_elements, aev_params, nn_params, elements, energies):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            self.aev = ANI1AEV(num_elements, **aev_params)
            self.nn = AtomWiseNN(**nn_params)
        self.coef, self.intercept = linear_coef_intercept(elements, energies,
                                                          num_elements)

    def forward(self, ci, ri, ei, i1, i2, j2, s2):
        dtype = chainer.config.dtype
        aev = self.aev(ci, ri, ei, i1, i2, j2, s2)
        atomic = self.nn(aev, ei)
        seed = self.xp.zeros((ci.shape[0], atomic.shape[1]), dtype=dtype)
        energy_nn = F.scatter_add(seed, i1, atomic)[:, 0]
        energy_linear = linear_predict(
            self.xp.array(self.coef, dtype=chainer.config.dtype),
            self.xp.array(self.intercept, dtype=chainer.config.dtype),
            ei, i1, self.xp)
        return energy_nn + energy_linear


def count_elements_noconcat(elements, number_elements, xp=np):
    return xp.array([F.scatter_add(xp.zeros(number_elements, dtype=xp.int32),
                                   e, xp.ones(len(e), xp.int32)).data
                     for e in elements])


def count_elements_concat(e1, i1, number_elements, xp=np):
    n = (i1 * number_elements) + e1
    seed = xp.zeros(number_elements * len(xp.unique(i1)), dtype=i1.dtype)
    return F.scatter_add(seed, n, xp.ones_like(i1)).data


def linear_predict(coef, intercept, ei, i1, xp=np):
    n_batch = len(xp.unique(i1))
    z = xp.zeros(n_batch, dtype=chainer.config.dtype)
    return F.scatter_add(z, i1, coef[ei]).data + intercept


def linear_coef_intercept(elements, energies, number_elements, xp=np):
    model = LinearRegression()
    model.fit(count_elements_noconcat(elements, number_elements), energies)
    return model.coef_, model.intercept_
