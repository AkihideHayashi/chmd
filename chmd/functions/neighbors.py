# *- coding:utf-8 -*-
"""Generate solo, duo and trio index."""
from typing import Tuple
import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable
from chainer.backend import get_array_module
from chmd.math.xp import repeat_interleave, cumsum_from_zero, cartesian_product
from chmd.utils.batchform import series_form


def number_repeats(cell: np.ndarray, pbc: np.ndarray,
                   cutoff: np.ndarray) -> np.ndarray:
    """Calculate max repeat number for each direction.

    Parameters
    ----------
    cell: (n, 3, 3) or (3, 3)
    pbc : (3,)
    cutoff: float

    Examples
    --------
    >>> cell = np.eye(3) * 3
    >>> pbc = np.array([True, True, True])
    >>> cutoff = 9.0
    >>> number_repeats(cell, pbc, cutoff)
    ... [3. 3. 3.]

    >>> cell = np.array([np.eye(3) * 3, np.eye(3) * 6])
    >>> pbc = np.array([True, True, True])
    >>> cutoff = 9.0
    >>> number_repeats(cell, pbc, cutoff)
    ... [[3. 3. 3.]
         [2. 2. 2.]]

    """
    assert cell.shape == (3, 3) or cell.shape[1:] == (3, 3)
    xp = get_array_module(cell)
    dtype = chainer.config.dtype
    # reciprocal_cell = xp.linalg.inv(cell)
    if cell.shape == (3, 3):
        reciprocal_cell = F.inv(cell).data
    else:
        reciprocal_cell = F.batch_inv(cell).data
    inv_lengths = xp.sqrt(xp.sum(reciprocal_cell ** 2, axis=-2))
    repeats = xp.ceil(cutoff * inv_lengths)
    pbc = xp.where(pbc, xp.ones(3, dtype=dtype), xp.zeros(3, dtype=dtype))
    return repeats * pbc


def compute_shifts(n_repeat: np.ndarray):
    """Compute shifts from the result of number_repeats or max of it.

    Parameters
    ----------
    n_repeat : (n_dim,) result of number repeats

    Returns
    -------
    shifts: (n_shifts, n_dim)

    """
    xp = get_array_module(n_repeat)
    return cartesian_product(*[xp.arange(-int(i), int(i+1)) for i in n_repeat])


def neighbor_duos_to_flatten_form(cells, positions, cutoff, pbc, valid):
    """Calculate neighbors and format them. Input must be parallel form.

    Returns
    -------
    i2, j2, s2

    """
    xp = get_array_module(cells)
    repeat = xp.max(number_repeats(cells, pbc, cutoff), axis=0)
    shifts = compute_shifts(repeat)
    n, i, j, s = neighbor_duos(cells, positions, cutoff, shifts, valid)
    n_batch, n_atoms = valid.shape
    raising_bottom = (xp.arange(n_batch) * n_atoms)[n]
    return i + raising_bottom, j + raising_bottom, s


def neighbor_duos_to_serial_form(cells, positions, cutoff, pbc, valid):
    """Calculate neighbors and format them. Input must be parallel form.

    Not Tested Yet!!!!!
    TODO: Write Test Code For Me!!!!
    """
    xp = get_array_module(cells)
    repeat = xp.max(number_repeats(cells, pbc, cutoff), axis=0)
    shifts = compute_shifts(repeat)
    n, i, j, s = neighbor_duos(cells, positions, cutoff, shifts, valid)
    natoms = xp.sum(valid, axis=1)
    raising_bottom = cumsum_from_zero(natoms)[n]
    return i + raising_bottom, j + raising_bottom, s


def neighbor_duos(cells: np.ndarray, positions: np.ndarray, cutoff: float,
                  shifts: np.ndarray, valid: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute neighbor pairs. Inputs are assumed to be parallel form.

    Parameters
    ----------
    cells: float[n_batch, n_dim, n_dim]
    positions: float[n_batch, n_atoms, n_dim] direct coordinate.
    cutoff: float
    shifts: float[n_shifts, n_dim] direct
    valid: bool[n_batch, n_atoms]

    Returns
    -------
    n, i, j, s

    The k-th bond is the bond in the n[k]-th batch,
    among which the i[k]-th bond and the j[k]-th bond.
    for all 0 <= k < len(n),
    | (positions[n[k], i[k]] - (positions[n[k], j[k]] + s[k])) | < cutoff
    where positions are cartesian.

    """
    xp = chainer.backend.get_array_module(cells)
    N = xp.newaxis
    n_batch, n_atoms, n_dim = positions.shape
    n_shifts = shifts.shape[0]
    assert cells.shape == (n_batch, n_dim, n_dim)
    assert positions.shape == (n_batch, n_atoms, n_dim)
    assert shifts.shape == (n_shifts, n_dim)
    # (n_batch, n_shifts, [n_dim], n_dim)
    # x @ A == xp.sum(x[:, N] * A[:, :], axis=0)
    full_shape = (n_batch, n_atoms, n_atoms, n_shifts, n_dim)
    cartesian_positions = xp.sum(positions[:, :, :, N] * cells[:, N, :, :], -2)
    # batch, atoms, dim, dim
    cartesian_shifts = xp.sum(shifts[N, :, :, N] * cells[:, N, :, :], -2)
    # batch, shifts, dim, dim
    vector = (cartesian_positions[:, :, N, N, :] -
              cartesian_positions[:, N, :, N, :] -
              cartesian_shifts[:, N, N, :, :])
    pow_distance = xp.sum(vector * vector, axis=-1)

    # batch, atoms, atoms, shifts, dim
    assert pow_distance.shape == (n_batch, n_atoms, n_atoms, n_shifts)
    base_shape = (n_batch, n_atoms, n_atoms, n_shifts)
    n = xp.broadcast_to(xp.arange(n_batch)[:, N, N, N], base_shape)
    i = xp.broadcast_to(xp.arange(n_atoms)[N, :, N, N], base_shape)
    j = xp.broadcast_to(xp.arange(n_atoms)[N, N, :, N], base_shape)
    s = xp.broadcast_to(cartesian_shifts[:, N, N, :, :], full_shape)
    in_cutoff = pow_distance <= cutoff * cutoff
    unique = ~((i == j) & xp.all(s == 0.0, axis=-1))
    isvalid = valid[:, :, N, N] & valid[:, N, :, N]
    enabled = in_cutoff & unique & isvalid
    return n[enabled], i[enabled], j[enabled], s[enabled]


def neighbor_trios(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Index for pair index.

    Parameters
    ----------
    i, j: concatenated result of neighbor pairs. it must be sorted.
    Returns: a, b
      assuming that i, j, k is trio (i is center)
      c: c == i3
      a: j[a] == j3
      b: j[b] == k3

    """
    xp = get_array_module(i)
    assert i.shape == j.shape
    assert xp.all(xp.diff(i) >= 0)
    center, number = xp.unique(i, return_counts=True)
    m = int(xp.max(number))
    n = len(center)
    base = cumsum_from_zero(number)[repeat_interleave(number * number)]
    center = xp.broadcast_to(center[:, xp.newaxis, xp.newaxis], (n, m, m))
    number = xp.broadcast_to(number[:, xp.newaxis, xp.newaxis], (n, m, m))
    idx = xp.broadcast_to(xp.arange(m)[xp.newaxis], (n, m))
    idx1, idx2 = xp.broadcast_arrays(idx[:, xp.newaxis, :],
                                     idx[:, :, xp.newaxis])
    filt = (idx1 < number) & (idx2 < number)

    center = center[filt]
    idx1 = idx1[filt] + base
    idx2 = idx2[filt] + base

    filt = idx1 != idx2
    return idx1[filt], idx2[filt]


def distance(ri: Variable, i2: np.ndarray, j2: np.ndarray, s2: np.ndarray):
    """Calculate distances.

    Parameters
    ----------
    r2: float[atoms, dim], cartesian, seriese form or flatten form.
    i2: int[pair]
    j2: int[pair]
    s2: float[pair, dim], cartesian

    """
    return F.sqrt(F.sum((ri[j2] + s2 - ri[i2]) ** 2, axis=1))


def distance_angle(ri: Variable,
                   i2: np.ndarray, j2: np.ndarray, s2: np.ndarray,
                   i3: np.ndarray, j3: np.ndarray):
    """Calculate distance and angles.

    Parameters
    ----------
    r2: float[atoms, dim], cartesian, seriese form or flatten form.
    i2: int[pair]
    j2: int[pair]
    s2: float[pair, dim], cartesian
    i3: int[trio]
    j3: int[trio]

    """
    rrij = (ri[j2][i3] + s2[i3] - ri[i2][i3])
    rrik = (ri[j2][j3] + s2[j3] - ri[i2][j3])
    rij = F.sqrt(F.sum(rrij ** 2, axis=1))
    rik = F.sqrt(F.sum(rrik ** 2, axis=1))
    cos = F.sum(rrij * rrik, axis=1) / (rij * rik)
    return rij, rik, cos


def duo_index(num_elements: int, xp):
    """Duo index."""
    e = xp.arange(num_elements)
    p1, p2 = cartesian_product(e, e).T
    ret = xp.zeros([num_elements, num_elements], dtype=xp.int32)
    ret[p1, p2] = xp.arange(num_elements * num_elements)
    return ret


def trio_index(num_elements: int, xp):
    """Trio index."""
    e = xp.arange(num_elements)
    p1, p2, p3 = cartesian_product(e, e, e).T
    ret = xp.zeros([num_elements, num_elements, num_elements], dtype=xp.int32)
    ret[p1, p2, p3] = xp.arange(num_elements * num_elements * num_elements)
    return ret


def concat_neighbors_flatten_form(n_batch, n_atoms, i2_seed, j2_seed, s2_seed):
    """Concatenate neighbors to flatten form."""
    raising_bottom = np.arange(n_batch) * n_atoms
    (i2_p, j2_p, s2), aff = series_form.from_list([i2_seed, j2_seed, s2_seed])
    i2 = i2_p + raising_bottom[aff]
    j2 = j2_p + raising_bottom[aff]
    return i2, j2, s2
