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


def kinetic_energy(n, m, v, i1):
    """Calculate kinetic energy.

    Parameters
    ----------
    n: number of systems.
    m: mass
    v: velocity
    i1: index of system.57

    """
    xp = get_array_module(v)
    return scatter_add_to_zero(n, i1, xp.sum(0.5 * m[:, None] * v * v, axis=1))


def temperature(kinetic, dof):
    """Calculate temperature.

    Parameters
    ----------
    kinetic: kinetic energy.
    n_free: degree of freedom for each system.

    """
    return kinetic * 2 / dof


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

    def __init__(self):
        """Initialize."""
        self.__initialized = False
        self.extensions = []
        self.reporter = Reporter()
        self.observation = {}

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
    def __init__(self, batch: Batch, energy_force_eval: Callable, dt):
        super().__init__()
        self.batch: Batch = batch
        self.energy_forces_eval = energy_force_eval
        self.accelerations = None
        self.delta_time = dt

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
        self.batch.positions = x_new
        self.batch.times = self.batch.times + self.delta_time
        self.energy_forces_eval(self.batch)
        a_new = self.batch.forces / m
        v_new = v_old + 0.5 * (a_old + a_new) * dt
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


class VelocityScaling(Dynamics):
    def __init__(self, batch: Batch, energy_force_eval: Callable, dt, kbt):
        super().__init__()
        self.batch: Batch = batch
        self.energy_forces_eval = energy_force_eval
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
    def __init__(self, batch: Batch, energy_force_eval: Callable, dt,
                 thermostat_kbt, thermostat_timeconst,
                 thermostat_numbers, thermostat_targets,
                 tol=1e-8,
                 ):
        super().__init__()
        self.batch: Batch = batch
        self.energy_forces_eval = energy_force_eval
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
        self.accelerations[self.is_atom] = (self.batch.forces / self.batch.masses[:, None]).flatten()

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
        will_report['conserved'] = nose_hoover_conserve(self.positions, self.velocities, self.masses, self.thermostat_numbers, self.thermostat_targets, self.thermostat_kbt, self.batch.potential_energies, self.affiliations)
        report(will_report)


# class VelocityVerlet(Dynamics):
#     """Implementation of velocity verlet algorithm."""
#
#     def __init__(self, positions, velocities, masses, t0, dt, energy_forces,
#                  name='md'):
#         """Initialize."""
#         super().__init__()
#         for x, v, m in zip(positions, velocities, masses):
#             assert x.shape == v.shape == m.shape
#         assert len(positions) == len(velocities) == len(masses)
#         assert len(positions) == len(t0) == len(dt)
#         xp = get_array_module(positions[0])
#         n_free = xp.array([len(p) for p in positions])
#         self.i1 = repeat_interleave(n_free)
#         self.positions = xp.concatenate(positions)
#         self.velocities = xp.concatenate(velocities)
#         self.masses = xp.concatenate(masses)
#         self.time = t0
#         self.delta_time = dt
#         self.accelerations = None
#         self.name = name
#         self.energy_forces = energy_forces
#
#     def initialize(self):
#         """Initialize accelerations."""
#         _, forces = self.energy_forces(self.time, self.positions)
#         self.accelerations = forces / self.masses
#
#     def update(self):
#         """Update all as velocity verlet."""
#         x = self.positions
#         v = self.velocities
#         a = self.accelerations
#         m = self.masses
#         dt = self.delta_time
#         t = self.time
#         i1 = self.i1
#
#         new_x = x + v * dt[i1] + 0.5 * a * dt[i1] * dt[i1]
#         new_t = t + dt
#         potential, forces = self.energy_forces(new_t, new_x)
#         new_a = forces / m
#         new_v = v + 0.5 * (a + new_a) * dt[i1]
#         self.positions = new_x
#         self.velocities = new_v
#         self.accelerations = new_a
#         self.time = self.time + dt
#         kinetic = kinetic_energy(len(self.n_free), m, new_v, i1)
#         report({'time': self.time,
#                 'positions': self.positions,
#                 'velocities': self.velocities,
#                 'accelerations': self.accelerations,
#                 'potential': potential,
#                 'forces': forces,
#                 'kinetic': kinetic,
#                 'energy': kinetic + potential,
#                 'conserved': kinetic + potential,
#                 'temperature': temperature(kinetic, self.n_free)
#                 }, self)
#
#
# class VelocityScaling(VelocityVerlet):
#     """Scale velocity to become selected temperature."""
#
#     def __init__(self, positions, velocities, masses,
#                  t0, dt, kbt, energy_forces,
#                  extentions=None, name='md'):
#         """Initialize."""
#         super().__init__(positions, velocities, masses,
#                          t0, dt, energy_forces, extentions, name)
#         self.kbt = kbt
#
#     def update(self):
#         """Update all as velocity verlet."""
#         x = self.positions
#         v = self.velocities
#         a = self.accelerations
#         m = self.masses
#         dt = self.delta_time
#         t = self.time
#         i1 = self.i1
#
#         new_x = x + v * dt[i1] + 0.5 * a * dt[i1] * dt[i1]
#         new_t = t + dt
#         potential, forces = self.energy_forces(new_t, new_x)
#         new_a = forces / m
#         tmp_v = v + 0.5 * (a + new_a) * dt[i1]
#         tmp_kinetic = kinetic_energy(len(self.n_free), m, tmp_v, i1)
#         tmp_kbt = temperature(tmp_kinetic, self.n_free)
#         new_v = tmp_v * np.sqrt(self.kbt / tmp_kbt)[self.i1]
#         self.positions = new_x
#         self.velocities = new_v
#         self.accelerations = new_a
#         self.time = self.time + dt
#         kinetic = kinetic_energy(len(self.n_free), m, new_v, i1)
#         kbt = temperature(kinetic, self.n_free)
#
#         report({'time': self.time,
#                 'positions': self.positions,
#                 'velocities': self.velocities,
#                 'accelerations': self.accelerations,
#                 'potential': potential,
#                 'forces': forces,
#                 'kinetic': kinetic,
#                 'energy': kinetic + potential,
#                 'conserved': None,
#                 'temperature': kbt,
#                 }, self)
#
#
# class NoseHooverChain(VelocityVerlet):
#     """Nose Hoover chain."""
#
#     def __init__(self, positions, velocities, masses,
#                  t0, dt, kbt, therm_time, therm_number, therm_target,
#                  energy_forces, extentions=None, name='md', tol=1e-8):
#         """Initialize.
#
#         Parameters
#         ----------
#         tol: tolerance for accelerate-velocity scf.
#
#         """
#         (extended_postions,
#          extended_velocities,
#          extended_masses,
#          self.therm_number,
#          self.therm_target,
#          self.kbt,
#          is_atom) = setup_nose_hoover(
#              positions, velocities, masses,
#              therm_number, therm_target, kbt, therm_time)
#         super().__init__(extended_postions,
#                          extended_velocities,
#                          extended_masses,
#                          t0, dt, energy_forces, extentions, name, is_atom)
#         self.tol = tol
#
#     def update(self):
#         """Update all as velocity verlet."""
#         x = self.positions
#         v = self.velocities
#         a = self.accelerations
#         m = self.masses
#         dt = self.delta_time
#         t = self.time
#         i1 = self.extended_i1
#         is_atom = self.is_atom
#         xp = get_array_module(x)
#
#         new_x = x + v * dt[i1] + 0.5 * a * dt[i1] * dt[i1]
#         new_t = t + dt
#         potential, s_forces = self.energy_forces(new_t, new_x[is_atom])
#         forces = xp.zeros_like(x)
#         forces[is_atom] = s_forces
#         new_v, new_a = nose_hoover_scf(a, v, forces, m,
#                                        self.therm_number, self.therm_target,
#                                        self.kbt, dt[i1], self.tol)
#         self.positions = new_x
#         self.velocities = new_v
#         self.accelerations = new_a
#         self.time = self.time + dt
#         kinetic = kinetic_energy(
#             len(self.n_free), m[is_atom], new_v[is_atom], i1[is_atom])
#         conserve = nose_hoover_conserve(self.positions, self.velocities,
#                                         self.masses,
#                                         self.therm_number,
#                                         self.therm_target,
#                                         self.kbt, potential, i1)
#         report({'time': self.time,
#                 'positions': self.positions[is_atom],
#                 'velocities': self.velocities[is_atom],
#                 'accelerations': self.accelerations[is_atom],
#                 'potential': potential,
#                 'forces': forces[is_atom],
#                 'kinetic': kinetic,
#                 'energy': kinetic + potential,
#                 'conserved': conserve,
#                 'temperature': temperature(kinetic, self.n_free),
#                 },
#                self)
#
#
