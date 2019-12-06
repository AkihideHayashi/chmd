import numpy as np
import chainer
from chmd.dynamics.dynamics import MolecularDynamicsBatch
from chmd.models.ani import ANI1Batch, ANI1ForceField
from chmd.dynamics.batch import BasicBatch, AbstractBatch
from chmd.dynamics.dynamics import BasicNoseHooverBatch
from chmd.dynamics.nosehoover import setup_nose_hoover_chain_parallel
from chmd.functions.neighbors import neighbor_duos_to_flatten_form
from chmd.utils.batchform import parallel_form, series_form
from chmd.preprocess import symbols_to_elements
from ase.data import atomic_masses, atomic_numbers


@np.vectorize
def get_mass(sym):
    if sym == '':
        return -1
    return atomic_masses[atomic_numbers[sym]]


class BasicNeighborList(object):
    def __init__(self, cutoff, pbc):
        self.cutoff = cutoff
        self.pbc = pbc
    
    def __call__(self, batch: AbstractBatch):
        cells = batch.cells
        positions = batch.positions
        pbc = self.pbc
        valid = batch.is_atom
        i2, j2, s2 = neighbor_duos_to_flatten_form(
            cells, positions, self.cutoff, pbc, valid)
        return i2, j2, s2


class ANI1MolecularDynamicsBatch(BasicNoseHooverBatch, ANI1Batch):
    def __init__(self, elements, cells, positions, velocities, masses, t0, kbt, numbers, targets, is_atom, is_thermostat, model):
        super().__init__(elements, cells, positions, velocities, masses, t0, kbt, numbers, targets, is_atom, is_thermostat)
        self.model = model
        n_batch = positions.shape[0]
        n_atoms = positions.shape[1]
        dtype = chainer.config.dtype
        with self.init_scope():
            self._error = self.xp.zeros(n_batch).astype(dtype)
            self._atomic_error = self.xp.zeros((n_batch, n_atoms)).astype(dtype)

    @staticmethod
    def setup(symbols, cells, positions, velocities, t0, masses, kbt, order, model):
        (sym, pos, vel), valid = parallel_form.from_list(
            [symbols, positions, velocities], ['', 0.0, 0.0])
        cel = np.array(cells)
        elements = symbols_to_elements(sym, order)
        mas = np.where(valid, masses[elements], np.nan)
        return ANI1MolecularDynamicsBatch(
            elements, cel, pos, vel, mas[:, :, np.newaxis], t0, kbt, np.array(0), np.array(0), valid, np.array(0), model
            )
    
    @staticmethod
    def setup_nose_hoover_chain(symbols, cells, positions, velocities, t0,
                                masses, order, model,
                                numbers, targets, timeconst, kbt):
        cel = np.array(cells)
        elements = [symbols_to_elements(s, order) for s in symbols]
        masses = [masses[e] for e in elements]
        sym, pos, vel, mas, nu, ta, kbt, isa, ist = setup_nose_hoover_chain_parallel(symbols, positions, velocities, masses, numbers, targets, timeconst, kbt)
        ele = symbols_to_elements(sym, order)
        return ANI1MolecularDynamicsBatch(
            ele, cel, pos, vel, mas, t0, kbt, nu, ta, isa, ist, model)
        
    
    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error):
        self._error[...] = error

    @property
    def atomic_error(self):
        return self._atomic_error

    @atomic_error.setter
    def atomic_error(self, atomic_error):
        self._atomic_error[...] = atomic_error


# class ANI1MolecularDynamicsBatch(BasicBatch, MolecularDynamicsBatch, ANI1Batch):
#     def __init__(self, symbols, cells, positions, velocities, t0, params, path, masses=None):
#         cutoff = params['cutoff']
#         order = np.array(params['order'])
#         self.model = ANI1ForceField(params, path, BasicNeighborList(cutoff))
#         (sym, pos, vel), val = parallel_form.from_list(
#             [symbols, positions, velocities], ['', 0.0, 0.0])
#         cl = np.array(cells)
#         elements = np.array([symbols_to_elements(s, order) for s in sym])
#         super().__init__(elements, cl, pos, val, np.array([True, True, True]))
#         dtype = chainer.config.dtype
#         if masses is None:
#             masses = get_mass(sym)
#         else:
#             masses = masses[elements]
#         with self.init_scope():
#             self._velocities = vel.astype(dtype)
#             self._accelerations = self.xp.zeros_like(vel).astype(dtype)
#             self._masses = masses.astype(dtype)
#             self._forces = self.accelerations / self.masses[:, :, self.xp.newaxis]
#             self._times = t0.astype(dtype)
#             self._error = self.xp.zeros_like(self.potential_energies)
#             self._atomic_error = self.xp.zeros_like(self._masses)

#     @property
#     def velocities(self):
#         return self._velocities

#     @velocities.setter
#     def velocities(self, velocities):
#         self._velocities[...] = velocities

#     @property
#     def accelerations(self):
#         return self._accelerations

#     @accelerations.setter
#     def accelerations(self, accelerations):
#         self._accelerations[...] = accelerations

#     @property
#     def forces(self):
#         return self._forces

#     @forces.setter
#     def forces(self, forces):
#         self._forces[...] = forces
    
#     @property
#     def times(self):
#         return self._times

#     @times.setter
#     def times(self, times):
#         self._times[...] = times

#     @property
#     def error(self):
#         return self._error

#     @error.setter
#     def error(self, error):
#         self._error[...] = error

#     @property
#     def atomic_error(self):
#         return self._atomic_error

#     @atomic_error.setter
#     def atomic_error(self, atomic_error):
#         self._atomic_error[...] = atomic_error

#     @property
#     def masses(self):
#         return self._masses
