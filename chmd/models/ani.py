"""ANI-1."""
import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
from chmd.links.ani import ANI1AEV, ANI1AEV2Energy
from chmd.links.shifter import EnergyShifter
from chmd.functions.neighbors import neighbor_duos, number_repeats
from chmd.utils.devicestruct import DeviceStruct


def asarray(x):
    if isinstance(x, Variable):
        return x.data
    return x

class ANI1(Chain):
    """ANI-1 energy calculator."""

    def __init__(self, num_elements, aev_params, nn_params, cutoff, pbc):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            self.aev = ANI1AEV(num_elements, **aev_params)
            self.energy = ANI1AEV2Energy(num_elements, nn_params)
            self.shift = EnergyShifter(num_elements)
        self.cutoff = cutoff
        self.pbc = pbc

    def forward(self, cells, elements, positions, affiliations,
                adjacents1=None, adjacents2=None, shifts=None):
        """Apply."""
        if adjacents1 is None or adjacents2 is None or shifts is None:
            repeats = number_repeats(cells, self.pbc, self.cutoff)
            repeat = self.xp.max(repeats, axis=0)
            (adjacents1, adjacents2, shifts) = neighbor_duos(
                asarray(cells), asarray(positions),
                self.cutoff, repeat, affiliations,)
        aev = self.aev(cells, positions, elements, affiliations,
                       adjacents1, adjacents2, shifts)
        energy_nn = self.energy(aev, elements, affiliations, cells.shape[0])
        energy_linear = self.shift(elements, affiliations)
        return energy_nn + energy_linear
