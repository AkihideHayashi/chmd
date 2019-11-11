"""Batch means a set of Atoms."""
from typing import Callable
import numpy as np
from chmd.utils.devicestruct import DeviceStruct
# DOF: Degree Of Freedom.


class Batch(DeviceStruct):
    """Basic Batch.

    This class have all values that are related to Hamiltonian mechanics.
    So, it have positions, momentums, velocity, forces.
    However, it doe's not have acceleration.
    """

    affiliations: np.ndarray  # i1

    elements: np.ndarray  # elemtent numbers.
    masses: np.ndarray
    dof: np.ndarray
    natoms: np.ndarray

    positions: np.ndarray
    momentums: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    times: np.ndarray

    potential_energies: np.ndarray
    kinetic_energies: np.ndarray
    mechanical_energies: np.ndarray

    cells: np.ndarray
    stress: np.ndarray
