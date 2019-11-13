"""About lattice and coordinate."""
import chainer
import chainer.functions as F


def direct_to_cartesian_chainer(cells, positions):
    """Direct to Cartesian transfrom in parallel form.

    Parameters
    ----------
    positions: (n_batch x n_atoms x n_dim)
    cells: (n_batch x n_dim x n_dim)
    xpf: np, cp, F are assumed.

    Returns
    -------
    positions

    """
    return F.sum(positions[:, :, :, None] * cells[:, None, :, :], axis=-2)


def direct_to_cartesian(cells, positions):
    """Direct to Cartesian transfrom in parallel form.

    Parameters
    ----------
    positions: (n_batch x n_atoms x n_dim)
    cells: (n_batch x n_dim x n_dim)
    xpf: np, cp, F are assumed.

    Returns
    -------
    positions

    """
    xp = chainer.backend.get_array_module(cells)
    return xp.sum(positions[:, :, :, None] * cells[:, None, :, :], axis=-2)
