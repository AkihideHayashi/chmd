"""Dynamics."""
import abc
from typing import List, Dict, Callable
import numpy as np
from chainer import report, Reporter
from chainer.backend import get_array_module
from chmd.math.xp import repeat_interleave, scatter_add_to_zero, cumsum_from_zero
from chmd.dynamics.nosehoover import (setup_nose_hoover,
                                      nose_hoover_scf,
                                      nose_hoover_conserve
                                      )
from chmd.dynamics.batch import Batch
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
    atomic = xp.where(valid,
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


class Dynamics(abc.ABC):
    """Base class for dynamics."""

    def __init__(self, energy_forces_eval, name='md'):
        """Initialize."""
        self.__initialized = False
        self.name = name
        self.extensions = []
        self.reporter = Reporter()
        self.observation = {}
        self.energy_forces_eval = energy_forces_eval
        self.reporter.add_observer(self.name, self)
        self.reporter.add_observer(energy_forces_eval.name, energy_forces_eval)

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
            ext(self.observation)

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


class VelocityVerlet(Dynamics):
    def __init__(self, batch: Batch, energy_forces_eval: Callable,
                 dt, name='md'):
        """Initializer.

        batch: assumed to have positions, velocities, energies,
               forces, affiliations.
        energy_forces_eval
        dt

        """
        super().__init__(energy_forces_eval, name=name)
        self.batch: Batch = batch
        self.accelerations = None
        self.delta_time = dt

    def initialize(self):
        super().initialize()
        self.energy_forces_eval(self.batch)
        self.accelerations = self.batch.forces / self.batch.masses[:, :, None]

    def update(self):
        xp = self.batch.xp
        x_old = self.batch.positions
        v_old = self.batch.velocities
        a_old = self.accelerations
        dof = self.batch.dof
        m = self.batch.masses[:, :, xp.newaxis]
        dt = self.delta_time[:, xp.newaxis, xp.newaxis]

        x_new = x_old + v_old * dt + 0.5 * a_old * dt * dt
        x_new = x_new - x_new // 1
        self.batch.positions = x_new
        self.batch.times = self.batch.times + self.delta_time
        self.energy_forces_eval(self.batch)
        a_new = self.batch.forces / m
        v_new = v_old + 0.5 * (a_old + a_new) * dt
        self.batch.positions = x_new
        self.batch.velocities = v_new
        self.accelerations = a_new
        self.batch.kinetic_energies = calculate_kinetic_energies(
            self.batch.cells,
            v_new,
            self.batch.masses,
            self.batch.valid
        )
        self.batch.mechanical_energies = (self.batch.kinetic_energies +
                                          self.batch.potential_energies)
        will_report = dict(self.batch.items())
        will_report['temperature'] = calculate_temperature(
            self.batch.kinetic_energies,
            dof
        )
        report(will_report)


class VelocityScaling(Dynamics):
    def __init__(self, batch: Batch, energy_forces_eval: Callable,
                 dt, kbt, name='md'):
        super().__init__(energy_forces_eval, name=name)
        self.batch: Batch = batch
        self.accelerations = None
        self.delta_time = dt
        self.kbt = kbt

    def initialize(self):
        super().initialize()
        self.energy_forces_eval(self.batch)
        self.accelerations = self.batch.forces / self.batch.masses[:, None]

    def update(self):
        xp = self.batch.xp
        i1 = self.batch.affiliations
        x_old = self.batch.positions
        v_old = self.batch.velocities
        a_old = self.accelerations
        dof = self.batch.dof
        m = self.batch.masses[:, xp.newaxis]

        dt = self.delta_time[i1][:, xp.newaxis]
        x_new = x_old + v_old * dt + 0.5 * a_old * dt * dt
        x_new = x_new - x_new // 1
        self.batch.positions = x_new
        self.batch.times = self.batch.times + self.delta_time
        self.energy_forces_eval(self.batch)
        a_new = self.batch.forces / m
        v_new = v_old + 0.5 * (a_old + a_new) * dt
        kinetic = kinetic_energy(len(dof), self.batch.masses, v_new, i1)
        kbt = temperature(kinetic, dof)
        scale = np.sqrt(self.kbt / kbt)
        v_new = v_new * scale[i1][:, None]
        self.batch.positions = x_new
        self.batch.velocities = v_new
        self.accelerations = a_new
        self.batch.kinetic_energies = kinetic_energy(len(dof),
                                                     self.batch.masses,
                                                     v_new, i1)
        self.batch.mechanical_energies = (self.batch.kinetic_energies +
                                          self.batch.potential_energies)
        will_report = dict(self.batch.items())
        will_report['temperature'] = temperature(
            self.batch.kinetic_energies, dof)
        report(will_report)


class NoseHooverChain(Dynamics):
    def __init__(self, batch: Batch, energy_forces_eval: Callable, dt,
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
        self.energy_forces_eval(self.batch)
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
        self.energy_forces_eval(self.batch)
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
