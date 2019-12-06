"""Dynamics."""
from abc import ABC, abstractproperty, abstractmethod
from typing import Dict, Callable
import numpy as np
from chainer.dataset.convert import to_device
import chainer
from chainer import report, Reporter
from chmd.dynamics.batch import BasicBatch
from chmd.dynamics.nosehoover import (nose_hoover_scf,
                                      nose_hoover_conserve
                                      )
from chmd.dynamics.analyze import (calculate_kinetic_energies,
                                   calculate_temperature)


class Extension(ABC):
    """Abstract base class for Extension."""

    @abstractmethod
    def __call__(self, observation: Dict):
        """Recieve observation and do something."""
        raise NotImplementedError()

    @abstractmethod
    def setup(self, dynamics):
        """Set up persistent parameters.

        Parameters
        ----------
        dynamics: Dynamics

        """
        raise NotImplementedError()


class DynamicsBatch(ABC):
    """Abstract Dynamics.

    Optimizer and Molecualar Dynamics and MonteCarlo.
    """

    @abstractproperty
    def positions(self):
        """(n_batch, n_atoms, n_dim)."""

    @positions.setter
    def positions(self, positions):
        """(n_batch, n_atoms, n_dim)."""

    @abstractproperty
    def elements(self):
        """(n_batch, n_atoms)."""

    @abstractproperty
    def is_atom(self):
        """(n_batch, n_atoms)."""

    @abstractproperty
    def potential_energies(self):
        """(n_batch)."""


class Dynamics(ABC):
    """Base class for dynamics."""

    def __init__(self, evaluator, name='md'):
        """Initialize."""
        self.__initialized = False
        self.name = name
        self.extensions = []
        self.reporter = Reporter()
        self.observation = {}
        self.evaluator = evaluator
        self.reporter.add_observer(self.name, self)
        self.reporter.add_observer(evaluator.name, evaluator)
        self.batch = None

    @abstractmethod
    def update(self):
        """Update coordinate and other properties."""
        raise NotImplementedError()

    def initialize(self):
        """Run at once at first."""
        self.__initialized = True

    def step(self):
        """Update and report."""
        with self.reporter.scope(self.observation):
            self.update()
            for ext in self.extensions:
                ext(self.batch)

    def run(self, n):
        """Run n step."""
        if not self.__initialized:
            self.initialize()
        for _ in range(n):
            self.step()

    def extend(self, ext):
        """Add extension."""
        self.extensions.append(ext)
        ext.setup(self)


class MolecularDynamicsBatch(DynamicsBatch):
    """Molecular Dynamics requires velocities and accelerations and times."""

    @abstractproperty
    def velocities(self):
        """(n_batch, n_atoms, n_dim)."""

    @velocities.setter
    def velocities(self, velocities):
        """(n_batch, n_atoms, n_dim)."""

    @abstractproperty
    def accelerations(self):
        """(n_batch, n_atoms, n_dim)."""

    @accelerations.setter
    def accelerations(self, accelerations):
        """(n_batch, n_atoms, n_dim)."""

    @abstractproperty
    def forces(self):
        """(n_batch, n_atoms, n_dim)."""

    @abstractproperty
    def times(self):
        """(n_batch)."""

    @times.setter
    def times(self):
        """(n_batch)."""

    @abstractproperty
    def masses(self):
        """(n_batch, n_atoms)."""


class NVTBatch(MolecularDynamicsBatch):
    """MD with NVT ensemble."""
    @abstractproperty
    def kbt(self):
        """Temperature."""


class NoseHooverBatch(NVTBatch):
    """Nose Hoover Chain"""
    @abstractproperty
    def numbers(self):
        """Thermostat numbers."""

    @abstractproperty
    def targets(self):
        """Thermostat targets."""
    
    @abstractproperty
    def is_thermostat(self):
        """Valid for thermostat."""


class BasicMolecularDynamicsBatch(BasicBatch, MolecularDynamicsBatch):
    """Molecular Dynamics requires velocities and accelerations and times."""
    def __init__(self, elements, cells, positions, velocities,
                 masses, t0, is_atom):
        super().__init__(elements, cells, positions, is_atom)
        dtype = chainer.config.dtype
        with self.init_scope():
            self._velocities = velocities.astype(dtype)
            self._accelerations = self.xp.zeros_like(velocities).astype(dtype)
            self._forces = self.xp.zeros_like(positions).astype(dtype)
            self._times = t0.astype(dtype)
            self._masses = masses
        
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
    def masses(self):
        return self._masses
    
    @masses.setter
    def masses(self, masses):
        self._masses[...] = masses


class BasicNVTBatch(BasicMolecularDynamicsBatch):# (BasicMolecularDynamicsBatch, NVTBatch):
    def __init__(self, elements, cells, positions, velocities, masses,
                 t0, kbt, is_atom):
        super().__init__(elements, cells, positions, velocities, masses,
                         t0, is_atom)
        with self.init_scope():
            self._kbt = kbt

    @property
    def kbt(self):
        return self._kbt

    @kbt.setter
    def kbt(self, kbt):
        self._kbt[...] = kbt


class BasicNoseHooverBatch(BasicNVTBatch):
    def __init__(self, elements, cells, positions, velocities, masses,
                 t0, kbt, numbers, targets, is_atom, is_thermostat):
        super().__init__(elements, cells, positions, velocities, masses,
                         t0, kbt, is_atom)
        with self.init_scope():
            self._numbers = numbers
            self._targets = targets
            self._is_thermostat = is_thermostat

    @property
    def numbers(self):
        return self._numbers

    @numbers.setter
    def numbers(self, numbers):
        self._numbers[...] = numbers

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, targets):
        self._targets[...] = targets

    @property
    def is_thermostat(self):
        return self._is_thermostat

    @is_thermostat.setter
    def is_thermostat(self, is_thermostat):
        self._is_thermostat[...] = is_thermostat


class VelocityVerlet(Dynamics):
    """Normal Velocity Verlet algorithm.

    Attributes
    ----------
    batch: batch object.
    energy_forces_eval: Callable.
    delta_time: (n_batch,)

    """

    def __init__(self, batch: MolecularDynamicsBatch,
                 evaluator: Callable,
                 dt, name='md'):
        """Initializer."""
        super().__init__(evaluator, name=name)
        self.batch: MolecularDynamicsBatch = batch
        self.delta_time = dt

    def initialize(self):
        """Initialize energy, forces and accelerations."""
        super().initialize()
        xp = self.batch.xp
        self.evaluator(self.batch)
        self.batch.accelerations = (self.batch.forces /
                                    self.batch.masses[:, :, xp.newaxis])
        self.delta_time = to_device(self.batch.device, self.delta_time)

    def update(self):
        """Velocity verloet algorithm."""
        xp = self.batch.xp
        x_old = self.batch.positions
        v_old = self.batch.velocities
        a_old = self.batch.accelerations
        m = self.batch.masses[:, :, xp.newaxis]
        dt = self.delta_time[:, xp.newaxis, xp.newaxis]

        x_new = x_old + v_old * dt + 0.5 * a_old * dt * dt
        x_new = x_new - x_new // 1  # pull bach into unit cell.
        self.batch.positions = x_new
        self.batch.times = self.batch.times + self.delta_time
        self.evaluator(self.batch)
        a_new = self.batch.forces / m
        v_new = v_old + 0.5 * (a_old + a_new) * dt
        self.batch.velocities = v_new
        self.batch.accelerations = a_new


class VelocityScaling(Dynamics):
    """Normal Velocity Verlet algorithm.

    Attributes
    ----------
    batch: batch object.
    energy_forces_eval: 
    delta_time: (n_batch,)
    kbt: (n_batch,)

    """

    def __init__(self,
                 batch: MolecularDynamicsBatch,
                 evaluator: Callable,
                 dt, kbt, name='md'):
        """Initializer.

        Parameters
        ----------
        batch: Parallel form MolecularDynamicsBatch
        evaluator: energy and forces evaluator.
        dt: (n_batch,)
        kbt: (n_batch,)

        """
        super().__init__(evaluator, name=name)
        self.batch: MolecularDynamicsBatch = batch
        self.delta_time = dt
        self.kbt = kbt

    def initialize(self):
        """Initialize energy, forces and accelerations."""
        super().initialize()
        xp = self.batch.xp
        self.evaluator(self.batch)
        self.batch.accelerations = (self.batch.forces / self.batch.masses)
        self.delta_time = to_device(self.batch.device, self.delta_time)
        self.kbt = to_device(self.batch.device, self.kbt)

    def update(self):
        """Velocity Scaling Velocity Verlet algorithm."""
        xp = self.batch.xp
        x_old = self.batch.positions
        v_old = self.batch.velocities
        a_old = self.batch.accelerations
        m = self.batch.masses
        dt = self.delta_time[:, xp.newaxis, xp.newaxis]

        x_new = x_old + v_old * dt + 0.5 * a_old * dt * dt
        x_new = x_new - x_new // 1  # pull back into unit cell.
        self.batch.positions = x_new
        self.batch.times = self.batch.times + self.delta_time
        self.evaluator(self.batch)
        a_new = self.batch.forces / m
        v_new = v_old + 0.5 * (a_old + a_new) * dt

        n_dim = self.batch.positions.shape[-1]
        dof = xp.sum(self.batch.is_atom, axis=1) * n_dim
        kinetic = calculate_kinetic_energies(self.batch.cells,
                                             v_new,
                                             self.batch.masses,
                                             self.batch.is_atom)
        kbt = calculate_temperature(kinetic, dof)
        ratio = np.sqrt(self.kbt / kbt)
        self.batch.velocities = v_new * ratio[:, xp.newaxis, xp.newaxis]
        self.batch.accelerations = a_new


class NoseHooverChain(Dynamics):
    def __init__(self, batch: NoseHooverBatch,
                 energy_forces_eval: Callable, dt,
                 tol=1e-8,
                 name='md'):
        super().__init__(energy_forces_eval, name='md')
        self.batch: NoseHooverBatch = batch
        self.delta_time = dt
        self.tol = tol

    def initialize(self):
        super().initialize()
        self.evaluator(self.batch)
        self.batch.accelerations = (self.batch.forces / self.batch.masses)
        self.delta_time = to_device(self.batch.device, self.delta_time)

    def update(self):
        xp = self.batch.xp
        dt = xp.broadcast_to(self.delta_time[:, xp.newaxis, xp.newaxis], self.batch.positions.shape)
        x_old = self.batch.positions
        v_old = self.batch.velocities
        a_old = self.batch.accelerations
        m = self.batch.masses
        self.batch.positions = x_old + v_old * dt + 0.5 * a_old * dt * dt
        self.evaluator(self.batch)
        forces = self.batch.forces
        v_new, a_new = nose_hoover_scf(
            a_old.flatten(),
            v_old.flatten(),
            forces.flatten(),
            m.flatten(),
            self.batch.numbers,
            self.batch.targets,
            self.batch.kbt.flatten(),
            dt.flatten(),
            self.tol
            )
        invalid = ~(self.batch.is_atom[:, :, xp.newaxis] | self.batch.is_thermostat)
        self.batch.velocities = v_new.reshape(self.batch.positions.shape)
        self.batch.velocities[invalid] = 0.0
        self.batch.accelerations = a_new.reshape(self.batch.positions.shape)
        self.batch.accelerations[invalid] = 0.0
        self.batch.times += self.delta_time

