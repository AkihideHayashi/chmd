"""Batch means a set of Atoms."""
from abc import ABC, abstractproperty, abstractmethod
import contextlib
from typing import Optional
import chainer
from chainer import DeviceResident
# DOF: Degree Of Freedom.

# We use deprecated @abstractproperty instead of @property @abstractmethod.
# It is because mypy doe's not support it.

class AbstractBatch(ABC):

    @abstractproperty
    def elements(self):
        ...

    @elements.setter
    def elements(self, _):
        ...

    @abstractproperty
    def cells(self):
        ...

    @cells.setter
    def cells(self, _):
        ...

    @abstractproperty
    def positions(self):
        ...

    @positions.setter
    def positions(self, _):
        ...

    @abstractproperty
    def potential_energies(self):
        ...

    @potential_energies.setter
    def potential_energies(self, _):
        ...

    @abstractproperty
    def xp(self):
        ...

    @abstractmethod
    def to_device(self, _):
        ...

    @abstractproperty
    def pbc(self):
        """(n_dim)"""

class DeviceBatch(DeviceResident):
    def __init__(self):
        super().__init__()
        self._within_init_scope = False
        self._quantities = set()
        self.name: Optional[str] = None
        self._reserved = ['_quantities',
                          '_within_init_scope',
                          '__init_done',
                          '_reserved']
        self.__init_done = True

    def __setattr__(self, name, value):
        if hasattr(self, '__init_done') and name in self._reserved:
            raise KeyError()
        if self.within_init_scope:
            self._quantities.add(name)
        super().__setattr__(name, value)

    def __check_init_done(self):
        if not self.__init_done:
            raise RuntimeError('BasicBatch.__init__() has not been called.')

    @property
    def within_init_scope(self) -> bool:
        return getattr(self, '_within_init_scope', False)

    @contextlib.contextmanager
    def init_scope(self):
        self.__check_init_done()
        old_flag = self.within_init_scope
        self._within_init_scope = True
        try:
            yield
        finally:
            super().__setattr__('_within_init_scope', old_flag)

    def device_resident_accept(self, visitor):
        """Neccesarry for to_device, xp, and a lot of other methods."""
        super().device_resident_accept(visitor)
        for key in self._quantities:
            setattr(self, key, visitor.visit_array(getattr(self, key)))


class BasicBatch(DeviceBatch, AbstractBatch):
    def __init__(self, elements, cells, positions, valid, pbc):
        super().__init__()
        n_batch = positions.shape[0]
        dtype = chainer.config.dtype
        with self.init_scope():
            self._elements = elements
            self._cells = cells.astype(dtype)
            self._positions = positions.astype(dtype)
            self._valid = valid
            self._potential_energies = self.xp.zeros(n_batch).astype(dtype)
            self._pbc = pbc

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, elements):
        self._elements[...] = elements

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cells):
        self._cells[...] = cells

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, positions):
        self._positions[...] = positions

    @property
    def potential_energies(self):
        return self._potential_energies

    @potential_energies.setter
    def potential_energies(self, energies):
        self.potential_energies[...] = energies

    @property
    def pbc(self):
        return self._pbc

    @property
    def valid(self):
        return self._valid


# class Batch(DeviceStruct):
#     """Basic Batch.
# 
#     This class have all values that are related to Hamiltonian mechanics.
#     So, it have positions, momentums, velocity, forces.
#     However, it doe's not have acceleration.
#     """
# 
#     affiliations: np.ndarray  # i1
# 
#     elements: np.ndarray  # elemtent numbers.
#     masses: np.ndarray
#     dof: np.ndarray
#     natoms: np.ndarray
# 
#     positions: np.ndarray
#     momentums: np.ndarray
#     velocities: np.ndarray
#     forces: np.ndarray
#     times: np.ndarray
# 
#     potential_energies: np.ndarray
#     kinetic_energies: np.ndarray
#     mechanical_energies: np.ndarray
# 
#     cells: np.ndarray
#     stress: np.ndarray
# 
