"""Velocity Verlet Algorithm."""
from typing import Callable
import numpy as np


class NoseHoover(object):
    def __init__(self, evaluator):
        self.evaluator = evaluator


class VelocityVerlet(object):
    """Wrapping velocity_verlet."""

    def __init__(self, times, positions, velocities, timesteps, evaluator,
                 reporter):
        """Initilizer."""
        self.times = times
        self.positions = positions
        self.velocities = velocities
        self.timesteps = timesteps
        self.evaluator = evaluator
        self.reporter = reporter
        self.accelerations = None
        self.energies = None

    def initialize(self):
        """Initialize energies and acclerations."""
        self.energies, self.accelerations = self.evaluator(self.times,
                                                           self.positions)

    def step(self):
        """Move for a step."""
        (self.times,
         self.positions,
         self.velocities,
         self.accelerations,
         self.energies) = velocity_verlet_step(self.times,
                                               self.positions,
                                               self.velocities,
                                               self.accelerations,
                                               self.evaluator,
                                               self.timesteps,
                                               )

    def run(self, n):
        """Run n step."""
        self.initialize()
        self.reporter(self)
        for _ in range(n):
            self.step()
            self.reporter(self)


def velocity_verlet_step(t0: np.ndarray,
                         x0: np.ndarray,
                         v0: np.ndarray,
                         a0: np.ndarray,
                         ea_eval: Callable,
                         dt: np.ndarray,
                         ):
    """Velocity Verlet algorithm.

    Parameters
    ----------
    t0: t_i, (n_free, )
    x0: x_i, (n_free, )
    v0: x_i, (n_free, )
    a0: a_i, (n_free, )
    ea_eval: energy and acceleration from time and positions.
    dt: delta t, (n_free, )

    Returns
    -------
    t1: t_(i+1)
    x1: x_(i+1)
    v1: v_(i+1)
    a1: a_(i+1)
    e1: energy (n_free,) or (n_batch,)

    """
    x1 = x0 + v0 * dt + 0.5 * a0 * dt * dt
    t1 = t0 + dt
    e1, a1 = ea_eval(t1, x1)
    v1 = v0 + 0.5 * (a0 + a1) * dt
    return t1, x1, v1, a1, e1
