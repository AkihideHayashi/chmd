"""Atom wise linear layer."""
import chainer
from chainer import ChainList
import chainer.links as L
import chainer.functions as F


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


class AtomWiseParamNN(ChainList):
    """NN part of ANI-1."""

    def __init__(self, n_layers, act):
        """Initialize."""
        super().__init__()
        for nl in n_layers:
            self.add_link(AtomNN(nl, act))

    def forward(self, x, e):
        """Select and apply NN for each atoms."""
        dtype = chainer.config.dtype
        out = F.concat([n(x)[None, :, :] for n in self], axis=0)
        n = out.shape[0]
        condition = self.xp.arange(n)[:, None] == e[None, :]
        zeros = self.xp.zeros(out.shape, dtype=dtype)
        ret = F.where(condition[:, :, None], out, zeros)
        return F.sum(ret, axis=0)
