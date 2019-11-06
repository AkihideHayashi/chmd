"""ANI-1."""
import chainer
from chainer import Chain
import chainer.functions as F
from chmd.links.ani import ANI1AEV
from chmd.links.shifter import EnergyShifter
from chmd.links.linear import AtomWiseParamNN
from chmd.functions.neighbors import neighbor_duos, number_repeats


class AEV2Energy(Chain):
    """ANI-1 energy calculator."""

    def __init__(self, num_elements, nn_params):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            self.nn = AtomWiseParamNN(**nn_params)
            self.shift = EnergyShifter(num_elements)

    def forward(self, aev, ei, i1, n_batch):
        """Forward."""
        dtype = chainer.config.dtype
        atomic = self.nn(aev, ei)
        seed = self.xp.zeros((n_batch, atomic.shape[1]), dtype=dtype)
        energy_nn = F.scatter_add(seed, i1, atomic)[:, 0]
        energy_linear = self.shift(ei, i1)
        return energy_nn + energy_linear


class Adjacent2Energy(Chain):
    """ANI-1 energy calculator."""

    def __init__(self, num_elements, aev_params, nn_params):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            self.aev = ANI1AEV(num_elements, **aev_params)
            self.energy = AEV2Energy(num_elements, nn_params)

    def forward(self, ri, ci, ei, i1, i2, j2, s2):
        """Apply."""
        aev = self.aev(ci, ri, ei, i1, i2, j2, s2)
        return self.energy(aev, ei, i1, ci.shape[0])


class Coordinates2Energy(Chain):
    """ANI-1 energy calculator."""

    def __init__(self, num_elements, aev_params, nn_params, cutoff, pbc):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            self.energy = Adjacent2Energy(num_elements, aev_params, nn_params)
        self.cutoff = cutoff
        self.pbc = pbc

    def forward(self, ri, ci, ei, i1):
        """Apply."""
        repeats = number_repeats(ci, self.pbc, self.cutoff, self.xp)
        repeat = self.xp.max(repeats, axis=0)
        i2, j2, s2 = neighbor_duos(
            ci, ri, self.cutoff, repeat, i1, self.xp)
        return self.energy(ri, ci, ei, i1, i2, j2, s2)
