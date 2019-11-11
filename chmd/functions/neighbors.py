# *- coding:utf-8 -*-
"""Generate solo, duo and trio index."""
from typing import Tuple
import numpy as np
import chainer.functions as F
from chainer import Variable
from chainer.backend import get_array_module
from chmd.math.xp import repeat_interleave, cumsum_from_zero, cartesian_product


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
    reciprocal_cell = xp.linalg.inv(cell)
    inv_lengths = xp.sqrt(xp.sum(reciprocal_cell ** 2, axis=-2))
    repeats = xp.ceil(cutoff * inv_lengths)
    pbc = xp.where(pbc, xp.ones(3), xp.zeros(3))
    return repeats * pbc


def compute_shifts(n_repeat: np.ndarray):
    """Compute shifts from the result of number_repeats or max of it.

    Parameters
    ----------
    n_repeat : (3,) result of number repeats

    """
    assert n_repeat.shape == (3,)
    xp = get_array_module(n_repeat)
    return cartesian_product(*[xp.arange(-i, i+1) for i in n_repeat])


def neighbor_duos(cells: np.ndarray, positions: np.ndarray,
                  cutoff: float, repeat: np.ndarray,
                  i1: np.ndarray):
    """Concatenate version of neighbor_duos_batch."""
    xp = get_array_module(cells)
    _, count = xp.unique(i1, return_counts=True)
    n_batch = len(count)
    n_atoms = int(max(count))
    head = cumsum_from_zero(count)
    index = xp.repeat(xp.arange(n_atoms)[xp.newaxis, :], n_batch, 0)
    padding = index >= count[:, xp.newaxis]
    index_batch = (index +
                   (xp.arange(n_batch) * n_atoms)[:, xp.newaxis])[~padding]
    r = xp.zeros((n_batch * n_atoms, 3))
    r[index_batch, :] = positions
    r = r.reshape((n_batch, n_atoms, 3))
    n, i, j, shift = neighbor_duos_batch(cells, r, cutoff,
                                         repeat, padding)
    head_n = head[n]
    return i + head_n, j + head_n, shift


def neighbor_duos_batch(cells: np.ndarray, positions: np.ndarray,
                        cutoff: float, repeat: np.ndarray,
                        padding=None
                        ) -> Tuple[np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]:
    """Compute pairs that are in neighbor.

    It only support the situation that all batches has similar cell.
    And all batches
    """
    assert repeat.shape == (3,)
    shifts = compute_shifts(repeat)
    n_batch = positions.shape[0]
    n_shifts = shifts.shape[0]
    n_atoms = positions.shape[1]
    assert cells.shape == (n_batch, 3, 3)
    assert shifts.shape == (n_shifts, 3)
    assert positions.shape == (n_batch, n_atoms, 3)

    xp = get_array_module(cells)
    if padding is None:
        padding = xp.full((n_batch, n_atoms), False)
    assert padding.shape == (n_batch, n_atoms)

    shifts, positions1, positions2 = xp.broadcast_arrays(
        shifts[xp.newaxis, xp.newaxis, xp.newaxis, :, :],
        positions[:, :, xp.newaxis, xp.newaxis, :],
        positions[:, xp.newaxis, :, xp.newaxis, :],
    )
    assert shifts.shape == (n_batch, n_atoms, n_atoms, n_shifts, 3)
    assert shifts.shape == positions1.shape == positions2.shape
    real_shifts = xp.sum(cells[:, xp.newaxis, xp.newaxis, xp.newaxis, :, :] *
                         shifts[:, :, :, :, :, xp.newaxis],
                         axis=-2
                         )
    assert real_shifts.shape == shifts.shape
    vectors = positions1 - positions2 - real_shifts
    assert vectors.shape == (n_batch, n_atoms, n_atoms, n_shifts, 3)
    distances = xp.sqrt(xp.sum(vectors * vectors, axis=4))
    assert distances.shape == (n_batch, n_atoms, n_atoms, n_shifts)
    i, j, n, _ = xp.broadcast_arrays(
        xp.arange(n_atoms)[xp.newaxis, :, xp.newaxis, xp.newaxis],
        xp.arange(n_atoms)[xp.newaxis, xp.newaxis, :, xp.newaxis],
        xp.arange(n_batch)[:, xp.newaxis, xp.newaxis, xp.newaxis],
        distances
    )

    in_cutoff = distances <= cutoff
    unique = ~((i == j) & xp.all(shifts == 0.0, axis=-1))
    not_dummy = (~padding[n, i]) & (~padding[n, j])
    enabled = in_cutoff & unique & not_dummy
    assert enabled.shape == (n_batch, n_atoms, n_atoms, n_shifts)
    return n[enabled], i[enabled], j[enabled], shifts[enabled]


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
    # return center[filt], idx1[filt], idx2[filt]


def distance(cells: Variable, positions: Variable,
             i1: np.ndarray, i2: np.ndarray, j2: np.ndarray, s2: np.ndarray):
    """Distance, not tested yet."""
    xp = get_array_module(cells)
    n = i1[i2]
    n_batches = cells.shape[0]
    n_atoms = positions.shape[0]
    n_pairs = len(n)
    assert cells.shape == (n_batches, 3, 3)
    assert positions.shape == (n_atoms, 3)
    assert xp.max(n) < n_batches
    assert get_array_module(positions) == xp
    assert i2.shape == (n_pairs, )
    assert j2.shape == (n_pairs, )
    assert n.shape == (n_pairs, )
    assert s2.shape == (n_pairs, 3)
    assert isinstance(n, xp.ndarray)
    assert isinstance(i2, xp.ndarray)
    assert isinstance(j2, xp.ndarray)
    assert isinstance(s2, xp.ndarray)
    real_shifts = F.sum(cells[n, :, :] * s2[:, :, xp.newaxis], axis=1)
    return F.sqrt(F.sum((positions[j2]
                         + real_shifts
                         - positions[i2]) ** 2, axis=1))


def distance_angle(cell: Variable, positions: Variable, i1: np.ndarray,
                   i2: np.ndarray, j2: np.ndarray, s2: np.ndarray,
                   i3: np.ndarray, j3: np.ndarray):
    """Distance and angles. Not in use yet."""
    n_pairs = len(i2)
    n_trios = len(i3)
    xp = get_array_module(positions)
    n = i1[i2]
    assert n.shape == (n_pairs,)
    assert i2.shape == (n_pairs,)
    assert j2.shape == (n_pairs,)
    assert s2.shape == (n_pairs, 3)
    assert i3.shape == (n_trios,)
    assert j3.shape == (n_trios,)
    assert get_array_module(cell) == xp
    assert isinstance(i2, xp.ndarray), (xp, type(i2))
    assert isinstance(j2, xp.ndarray), (xp, type(j2))
    assert isinstance(s2, xp.ndarray), (xp, type(s2))
    assert isinstance(i3, xp.ndarray), (xp, type(i3))
    assert isinstance(j3, xp.ndarray), (xp, type(j3))
    real_shifts = F.sum(cell[n, :, :] * s2[:, :, xp.newaxis], axis=1)
    r = positions
    rrij = (r[j2][i3] + real_shifts[i3] - r[i2][i3])
    rrik = (r[j2][j3] + real_shifts[j3] - r[i2][j3])
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
