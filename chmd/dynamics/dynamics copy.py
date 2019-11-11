"""Dynamics."""
import abc
from typing import List, Dict
import numpy as np
from chainer import report, Reporter
from chainer.backend import get_array_module
from chmd.math.xp import repeat_interleave, scatter_add_to_zero
from chmd.system.nosehoover import (setup_nose_hoover,
                                    nose_hoover_scf,
                                    nose_hoover_conserve
                                    )


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
    """Base class for dynamics.

    Attributes
    ----------
    i1: Which system each atom belongs.
    energy_forces: Energy and forces evaluator.
    reporter: reporter object.
    observation: Dictionary which store reported values.
    name: Name of the class. It is neccesarry for reporting.
    """
    @abc.abstractproperty
    def positions(self):
        raise NotImplementedError()
    @abc.abstractmethod
    def update(self):
        """Update coordinate and other properties."""
        raise NotImplementedError()

    def step(self):
        """Update and report."""
        with self.reporter.scope(self.observation):
            self.update()
        for ext in self.extentions:
            ext(self.observation)

    def run(self, n):
        """N step."""
        for _ in range(n):
            self.step()

    def extend(self, ext):
        """Add extension."""
        self.extentions.append(ext)
        ext.setup(self)

class Dynamics(abc.ABC):
    """Base class for dynamics.

    Attributes
    ----------
    is_atom:
    i1: Which system each atom belongs.
    n_free: Number of degree of freedom for each system.
    energy_forces: Energy and forces evaluator.
    reporter: reporter object.
    observation: Dictionary which store reported values.
    name: Name of the class. It is neccesarry for reporting.
    """

    def __init__(self, poistions: np.ndarray,
                 energy_forces,
                 extentions=None, name='md', is_atom=None):
        """Initializer."""
        self.positions = np.concatenate(poistions)
        if is_atom is None:
            self.is_atom = np.full(self.positions.shape , True)
        else:
            self.is_atom = is_atom
        self.name = name
        self.extended_i1 = repeat_interleave(self.extended_n_free)
        self.extended_n_free = np.array([len(x) for x in poistions])
        self.i1 = self.extended_i1[self.is_atom]
        self.n_free = np.unique(self.i1, return_counts=True)[1]
        assert np.all(self.extended_n_free == np.unique(self.extended_i1,
                                                        return_counts=True)[1])
        self.energy_forces = energy_forces
        self.reporter = Reporter()
        self.observation: Dict[str, None] = {}
        self.reporter.add_observer(name, self)
        if hasattr(self.energy_forces, 'setup'):
            self.energy_forces.setup(self)
        if extentions is None:
            self.extentions: List[Extension] = []
        else:
            self.extentions = extentions
        for ext in self.extentions:
            if hasattr(ext, 'setup'):
                ext.setup(self)

    @abc.abstractmethod
    def update(self):
        """Update coordinate and other properties."""
        raise NotImplementedError()

    def step(self):
        """Update and report."""
        with self.reporter.scope(self.observation):
            self.update()
        for ext in self.extentions:
            ext(self.observation)

    def run(self, n):
        """N step."""
        for _ in range(n):
            self.step()

    def extend(self, ext):
        """Add extension."""
        self.extentions.append(ext)
        ext.setup(self)


class VelocityVerlet(Dynamics):
    """Implementation of velocity verlet algorithm."""

    def __init__(self, positions, velocities, masses, t0, dt, energy_forces,
                 extentions=None, name='md', is_atom=None):
        """Initialize."""
        super().__init__(positions, energy_forces, extentions, name, is_atom)
        for x, v, m in zip(positions, velocities, masses):
            assert x.shape == v.shape == m.shape
        assert len(positions) == len(velocities) == len(masses)
        assert len(positions) == len(t0) == len(dt)
        self.velocities = np.concatenate(velocities)
        self.masses = np.concatenate(masses)
        self.time = t0
        self.delta_time = dt
        _, s_forces = self.energy_forces(self.time, self.positions[self.is_atom])
        forces = np.zeros_like(self.positions)
        forces[is_atom] = s_forces
        self.accelerations = forces / self.masses

    def update(self):
        """Update all as velocity verlet."""
        x = self.positions
        v = self.velocities
        a = self.accelerations
        m = self.masses
        dt = self.delta_time
        t = self.time
        i1 = self.i1

        new_x = x + v * dt[i1] + 0.5 * a * dt[i1] * dt[i1]
        new_t = t + dt
        potential, forces = self.energy_forces(new_t, new_x)
        new_a = forces / m
        new_v = v + 0.5 * (a + new_a) * dt[i1]
        self.positions = new_x
        self.velocities = new_v
        self.accelerations = new_a
        self.time = self.time + dt
        kinetic = kinetic_energy(len(self.n_free), m, new_v, i1)
        report({'time': self.time,
                'positions': self.positions,
                'velocities': self.velocities,
                'accelerations': self.accelerations,
                'potential': potential,
                'forces': forces,
                'kinetic': kinetic,
                'energy': kinetic + potential,
                'conserved': kinetic + potential,
                'temperature': temperature(kinetic, self.n_free)
                }, self)


class VelocityScaling(VelocityVerlet):
    """Scale velocity to become selected temperature."""

    def __init__(self, positions, velocities, masses,
                 t0, dt, kbt, energy_forces,
                 extentions=None, name='md'):
        """Initialize."""
        super().__init__(positions, velocities, masses,
                         t0, dt, energy_forces, extentions, name)
        self.kbt = kbt

    def update(self):
        """Update all as velocity verlet."""
        x = self.positions
        v = self.velocities
        a = self.accelerations
        m = self.masses
        dt = self.delta_time
        t = self.time
        i1 = self.i1

        new_x = x + v * dt[i1] + 0.5 * a * dt[i1] * dt[i1]
        new_t = t + dt
        potential, forces = self.energy_forces(new_t, new_x)
        new_a = forces / m
        tmp_v = v + 0.5 * (a + new_a) * dt[i1]
        tmp_kinetic = kinetic_energy(len(self.n_free), m, tmp_v, i1)
        tmp_kbt = temperature(tmp_kinetic, self.n_free)
        new_v = tmp_v * np.sqrt(self.kbt / tmp_kbt)[self.i1]
        self.positions = new_x
        self.velocities = new_v
        self.accelerations = new_a
        self.time = self.time + dt
        kinetic = kinetic_energy(len(self.n_free), m, new_v, i1)
        kbt = temperature(kinetic, self.n_free)

        report({'time': self.time,
                'positions': self.positions,
                'velocities': self.velocities,
                'accelerations': self.accelerations,
                'potential': potential,
                'forces': forces,
                'kinetic': kinetic,
                'energy': kinetic + potential,
                'conserved': None,
                'temperature': kbt,
                }, self)


class NoseHooverChain(VelocityVerlet):
    """Nose Hoover chain."""

    def __init__(self, positions, velocities, masses,
                 t0, dt, kbt, therm_time, therm_number, therm_target,
                 energy_forces, extentions=None, name='md', tol=1e-8):
        """Initialize.

        Parameters
        ----------
        tol: tolerance for accelerate-velocity scf.

        """
        (extended_postions,
         extended_velocities,
         extended_masses,
         self.therm_number,
         self.therm_target,
         self.kbt,
         is_atom) = setup_nose_hoover(
             positions, velocities, masses,
             therm_number, therm_target, kbt, therm_time)
        super().__init__(extended_postions,
                         extended_velocities,
                         extended_masses,
                         t0, dt, energy_forces, extentions, name, is_atom)
        self.tol = tol

    def update(self):
        """Update all as velocity verlet."""
        x = self.positions
        v = self.velocities
        a = self.accelerations
        m = self.masses
        dt = self.delta_time
        t = self.time
        i1 = self.extended_i1
        is_atom = self.is_atom
        xp = get_array_module(x)

        new_x = x + v * dt[i1] + 0.5 * a * dt[i1] * dt[i1]
        new_t = t + dt
        potential, s_forces = self.energy_forces(new_t, new_x[is_atom])
        forces = xp.zeros_like(x)
        forces[is_atom] = s_forces
        new_v, new_a = nose_hoover_scf(a, v, forces, m,
                                       self.therm_number, self.therm_target,
                                       self.kbt, dt[i1], self.tol)
        self.positions = new_x
        self.velocities = new_v
        self.accelerations = new_a
        self.time = self.time + dt
        kinetic = kinetic_energy(len(self.n_free), m[is_atom], new_v[is_atom], i1[is_atom])
        conserve = nose_hoover_conserve(self.positions, self.velocities,
                                        self.masses,
                                        self.therm_number,
                                        self.therm_target,
                                        self.kbt, potential, i1)
        report({'time': self.time,
                'positions': self.positions[is_atom],
                'velocities': self.velocities[is_atom],
                'accelerations': self.accelerations[is_atom],
                'potential': potential,
                'forces': forces[is_atom],
                'kinetic': kinetic,
                'energy': kinetic + potential,
                'conserved': conserve,
                'temperature': temperature(kinetic, self.n_free),
                },
               self)


def kinetic_energy(n, m, v, i1):
    """Calculate kinetic energy.

    Parameters
    ----------
    n: number of systems.
    m: mass
    v: velocity
    i1: index of system.57

    """
    return scatter_add_to_zero(n, i1, 0.5 * m * v * v)


def temperature(kinetic, n_free):
    """Calculate temperature.

    Parameters
    ----------
    kinetic: kinetic energy.
    n_free: degree of freedom for each system.

    """
    return kinetic * 2 / n_free
