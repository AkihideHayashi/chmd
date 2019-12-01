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
        """Select and apply NN for each atoms.

        Parameters
        ----------
        x: AEV
        e: elements

        """
        xp = self.xp
        dtype = chainer.config.dtype
        n_elements = len(self)
        ret = None
        for i in range(n_elements):
            filt = e == i
            result = self[i](x[filt])
            if ret is None:
                ret = xp.zeros((x.shape[0], *result.shape[1:]), dtype=dtype)
            ret = F.scatter_add(ret, filt, result)
        return ret
