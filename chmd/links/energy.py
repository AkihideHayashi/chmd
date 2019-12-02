import numpy as np
import chainer
from chainer import functions as F
from chainer.backend import get_device_from_array, get_array_module
from chainer.dataset import to_device


class EnergyNet(object):
    """Energy Net."""

    def __init__(self, energy_ranges):
        self.energy_ranges = energy_ranges
    
    def __call__(self, features, elements):
        xp = get_array_module(features)
        device = get_device_from_array(elements)
        dtype = chainer.config.dtype
        n_atoms, n_features = features.shape
        assert elements.shape == (n_atoms,)
        weights = F.softmax(features, axis=1)
        energies = xp.zeros((n_atoms,), dtype)
        for i, [low, high] in enumerate(self.energy_ranges):
            stop = np.log(high - low + 1)
            grid_cpu = np.linspace(0.0, stop, n_features)
            grid = to_device(device, grid_cpu)
            filt = elements == i
            log_en = F.sum(weights[filt] * grid, axis=1)
            en = F.exp(log_en) + low - 1
            energies = F.scatter_add(energies, filt, en)
        assert energies.shape == (n_atoms,), "{} != {}".format(energies.shape, (n_atoms,))
        return energies
