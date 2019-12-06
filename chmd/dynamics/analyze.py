"""Analyze."""
from chainer.backend import get_array_module
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
                      0.5 * m * v * v,
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
