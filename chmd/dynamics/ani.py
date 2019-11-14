import numpy as np
import chainer
from chmd.dynamics.dynamics import MolecularDynamicsBatch
from chmd.models.ani import ANI1Batch, ANI1ForceField
from chmd.dynamics.batch import BasicBatch, AbstractBatch
from chmd.functions.neighbors import neighbor_duos_to_flatten_form
from chmd.utils.batchform import parallel_form
from chmd.preprocess import symbols_to_elements
from ase.data import atomic_masses, atomic_numbers


@np.vectorize
def get_mass(sym):
    if sym == '':
        return -1
    return atomic_masses[atomic_numbers[sym]]


class BasicNeighborList(object):
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def __call__(self, batch: AbstractBatch):
        cells = batch.cells
        positions = batch.positions
        pbc = batch.pbc
        valid = batch.valid
        i2, j2, s2 = neighbor_duos_to_flatten_form(
            cells, positions, self.cutoff, pbc, valid)
        return i2, j2, s2


class ANI1MolecularDynamicsBatch(BasicBatch, MolecularDynamicsBatch, ANI1Batch):
    def __init__(self, symbols, cells, positions, velocities, t0, params, path, masses=None):
        cutoff = params['cutoff']
        order = np.array(params['order'])
        self.model = ANI1ForceField(params, path, BasicNeighborList(cutoff))
        (sym, pos, vel), val = parallel_form.from_list(
            [symbols, positions, velocities], ['', 0.0, 0.0])
        cl = np.array(cells)
        elements = np.array([symbols_to_elements(s, order) for s in sym])
        super().__init__(elements, cl, pos, val, np.array([True, True, True]))
        dtype = chainer.config.dtype
        if masses is None:
            masses = get_mass(sym)
        else:
            masses = masses[elements]
        with self.init_scope():
            self._velocities = vel.astype(dtype)
            self._accelerations = self.xp.zeros_like(vel).astype(dtype)
            self._masses = masses.astype(dtype)
            self._forces = self.accelerations / self.masses[:, :, self.xp.newaxis]
            self._times = t0.astype(dtype)
            self._variance_potential_energies = self.xp.zeros_like(
                self.potential_energies)

    @property
    def velocities(self):
        return self._velocities

    @velocities.setter
    def velocities(self, velocities):
        self._velocities[...] = velocities

    @property
    def accelerations(self):
        return self._accelerations

    @accelerations.setter
    def accelerations(self, accelerations):
        self._accelerations[...] = accelerations

    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, forces):
        self._forces[...] = forces
    
    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, times):
        self._times[...] = times

    @property
    def variance_potential_energies(self):
        return self._variance_potential_energies

    @variance_potential_energies.setter
    def variance_potential_energies(self, vpe):
        self._variance_potential_energies[...] = vpe

    @property
    def masses(self):
        return self._masses
