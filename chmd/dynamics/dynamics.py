"""Dynamics."""
import abc
from abc import ABC, abstractproperty, abstractmethod
from typing import List, Dict, Callable
import numpy as np
from chainer import report, Reporter
from chainer.backend import get_array_module
from chmd.math.xp import (repeat_interleave,
                          scatter_add_to_zero,
                          cumsum_from_zero)
from chmd.dynamics.nosehoover import (setup_nose_hoover,
                                      nose_hoover_scf,
                                      nose_hoover_conserve
                                      )
from chmd.dynamics.batch import AbstractBatch
from chmd.math.lattice import direct_to_cartesian


def calculate_kinetic_energies(cells, velocities, masses, valid):
    """All parameters are assumed to be parallel format.

    Parameters
    ----------
    cells: float(n_batch, n_dim, n_dim)
    velocities: float(n_batch, n_atoms, n_dim) direct
    masses: float(n_batch, n_atoms)
    valid: bool(n_batch, n_atoms)
    """
    n_batch, n_atoms, n_dim = velocities.shape
    xp = get_array_module(cells)
    v = direct_to_cartesian(cells, velocities)
    m = masses
    atomic = xp.where(valid[:, :, xp.newaxis],
                      0.5 * m[:, :, xp.newaxis] * v * v,
                      xp.zeros_like(v))
    return xp.sum(atomic.reshape((n_batch, n_atoms * n_dim)), axis=1)


def calculate_temperature(kinetic, dof):
    """Calculate temperature.

    Parameters
    ----------
    kinetic: kinetic energy.
    n_free: degree of freedom for each system.

    """
    return (kinetic * 2 / dof).astype(kinetic.dtype)


class Extension(abc.ABC):
    """Abstract base class for Extension."""

    @abc.abstractmethod
    def __call__(self, observation: Dict):
        """Recieve observation and do something."""
        raise NotImplementedError()

    @abc.abstractmethod
    def setup(self, dynamics):
        """Set up persistent parameters.

        Parameters
        ----------
        dynamics: Dynamics

        """
        raise NotImplementedError()


class DynamicsBatch(ABC):
    """Abstract Dynamics is Optimizer and Molecualar Dynamics and Monte Carlo"""
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
    def valid(self):
        """(n_batch, n_atoms)."""

    @abstractproperty
    def potential_energies(self):
        """(n_batch)."""


class Dynamics(abc.ABC):
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

    @abc.abstractmethod
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
        """(n_batch, n_atoms)"""


class VelocityVerlet(Dynamics):
    """Normal Velocity Verlet algorithm.

    Attributes
    ----------
    batch: batch object.
    energy_forces_eval: 
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
        self.batch: Batch = batch
        self.delta_time = dt
        self.kbt = kbt

    def initialize(self):
        """Initialize energy, forces and accelerations."""
        super().initialize()
        xp = self.batch.xp
        self.evaluator(self.batch)
        self.batch.accelerations = (self.batch.forces /
                                    self.batch.masses[:, :, xp.newaxis])

    def update(self):
        """Velocity Scaling Velocity Verlet algorithm."""
        xp = self.batch.xp
        x_old = self.batch.positions
        v_old = self.batch.velocities
        a_old = self.batch.accelerations
        m = self.batch.masses[:, :, xp.newaxis]
        dt = self.delta_time[:, xp.newaxis, xp.newaxis]

        x_new = x_old + v_old * dt + 0.5 * a_old * dt * dt
        x_new = x_new - x_new // 1  # pull back into unit cell.
        self.batch.positions = x_new
        self.batch.times = self.batch.times + self.delta_time
        self.evaluator(self.batch)
        a_new = self.batch.forces / m
        v_new = v_old + 0.5 * (a_old + a_new) * dt

        n_dim = self.batch.positions.shape[-1]
        dof = xp.sum(self.batch.valid, axis=1) * n_dim
        kinetic = calculate_kinetic_energies(self.batch.cells,
                                             v_new,
                                             self.batch.masses,
                                             self.batch.valid)
        kbt = calculate_temperature(kinetic, dof)
        ratio = np.sqrt(self.kbt / kbt)
        self.batch.velocities = v_new * ratio[:, xp.newaxis, xp.newaxis]
        self.batch.accelerations = a_new


class NoseHooverChain(Dynamics):
    def __init__(self, batch: MolecularDynamicsBatch, energy_forces_eval: Callable, dt,
                 thermostat_kbt, thermostat_timeconst,
                 thermostat_numbers, thermostat_targets,
                 tol=1e-8,
                 name='md'
                 ):
        import warnings
        warnings.warn('Now, nose hoover is assumed to handle seriese form.')
        super().__init__(energy_forces_eval, name='md')
        self.batch: Batch = batch
        self.accelerations = None
        self.delta_time = dt
        self.tol = tol
        (self.positions,
         self.velocities,
         self.masses,
         self.thermostat_numbers,
         self.thermostat_targets,
         self.thermostat_kbt,
         self.is_atom,
         self.affiliations
         ) = setup_nose_hoover(
            self.batch.positions,
            self.batch.velocities,
            self.batch.masses,
            self.batch.affiliations,
            thermostat_numbers,
            thermostat_targets,
            thermostat_kbt,
            thermostat_timeconst)

    def initialize(self):
        super().initialize()
        self.accelerations = self.batch.xp.zeros_like(self.positions)
        self.evaluator(self.batch)
        self.accelerations[self.is_atom] = (
            self.batch.forces / self.batch.masses[:, None]).flatten()

    def update(self):
        xp = self.batch.xp
        dt = self.delta_time[self.affiliations]
        x_old = self.positions
        v_old = self.velocities
        a_old = self.accelerations
        x_old[self.is_atom] = self.batch.positions.flatten()
        v_old[self.is_atom] = self.batch.velocities.flatten()
        m = self.masses
        x_new = x_old + v_old * dt + 0.5 * a_old * dt * dt
        self.batch.positions = x_new[self.is_atom].reshape(
            self.batch.positions.shape)
        self.batch.times = self.batch.times + self.delta_time
        self.evaluator(self.batch)
        forces = xp.zeros_like(self.positions)
        forces[self.is_atom] = self.batch.forces.flatten()
        v_new, a_new = nose_hoover_scf(a_old, v_old, forces, m,
                                       self.thermostat_numbers,
                                       self.thermostat_targets,
                                       self.thermostat_kbt, dt, self.tol)
        self.positions = x_new
        self.velocities = v_new
        self.accelerations = a_new
        self.batch.velocities = v_new[self.is_atom].reshape(
            self.batch.velocities.shape)
        self.batch.kinetic_energies = kinetic_energy(len(self.batch.dof),
                                                     self.batch.masses,
                                                     self.batch.velocities,
                                                     self.batch.affiliations)
        self.batch.mechanical_energies = (self.batch.kinetic_energies +
                                          self.batch.potential_energies)
        will_report = dict(self.batch.items())
        will_report['temperatures'] = temperature(
            self.batch.kinetic_energies, self.batch.dof)
        will_report['conserved'] = nose_hoover_conserve(self.positions, self.velocities, self.masses, self.thermostat_numbers,
                                                        self.thermostat_targets, self.thermostat_kbt, self.batch.potential_energies, self.affiliations)
        report(will_report)
