"""ANI-1."""
import chainer
from chainer import Chain, ChainList
import chainer.links as L
import chainer.functions as F
from chmd.links.ani import ANI1AEV
from chmd.activations import gaussian


class AtomNN(ChainList):
    """Auxiliary function for AtomWiseNN."""

    def __init__(self, n_layers):
        """Initilize."""
        super().__init__()
        for nl in n_layers:
            self.add_link(L.Linear(None, nl, nobias=False))

    def forward(self, x):
        """Compute for each atoms."""
        h = self[0](x)
        for l in self[1:]:
            h = l(gaussian(h))
        return h


class AtomWiseNN(ChainList):
    """NN part of ANI-1."""

    def __init__(self, n_layers):
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

    def __init__(self, num_elements, aev_params, nn_params):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            self.aev = ANI1AEV(num_elements, **aev_params)
            self.nn = AtomWiseNN(**nn_params)

    def forward(self, ci, ri, ei, i1, i2, j2, s2):
        dtype = chainer.config.dtype
        aev = self.aev(ci, ri, ei, i1, i2, j2, s2)
        atomic = self.nn(aev, ei)
        seed = self.xp.zeros((ci.shape[0], atomic.shape[1]), dtype=dtype)
        return F.scatter_add(seed, i1, atomic)[:, 0]