# *- coding:utf-8 -*-
"""Generate solo, duo and trio index."""
from typing import Tuple
import numpy as np
import chainer.functions as F
from chainer import Variable
from chainer.backend import get_array_module


def cartesian_product(*args, xp=np) -> np.ndarray:
    """Cartesian product of arrays.

    Examples
    --------
    >>> x = np.array([10, 20])
    >>> y = np.array([1, 2, 3])
    >>> cartesian_product(x, y)
    ... [[10  1]
         [10  2]
         [10  3]
         [20  1]
         [20  2]
         [20  3]]

    """
    n = len(args)
    shapes = [tuple(len(x) if i == j else 1
                    for i in range(n)) for j, x in enumerate(args)]
    # x = tuple(a.reshape(shape) for a, shape in zip(args, shapes))
    broad = xp.broadcast_arrays(*tuple(a.reshape(shape)
                                       for a, shape in zip(args, shapes)))
    cat = xp.concatenate([xp.expand_dims(b, -1) for b in broad], axis=-1)
    return cat.reshape((-1, n))


def number_repeats(cell: np.ndarray, pbc: np.ndarray,
                   cutoff: np.ndarray, xp=np) -> np.ndarray:
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
    reciprocal_cell = xp.linalg.inv(cell)
    inv_lengths = xp.sqrt(xp.sum(reciprocal_cell ** 2, axis=-2))
    repeats = xp.ceil(cutoff * inv_lengths)
    pbc = xp.where(pbc, xp.ones(3), xp.zeros(3))
    return repeats * pbc


def compute_shifts(n_repeat: np.ndarray, xp=np):
    """Compute shifts from the result of number_repeats or max of it.

    Parameters
    ----------
    n_repeat : (3,) result of number repeats

    """
    assert n_repeat.shape == (3,)
    return cartesian_product(*[xp.arange(-i, i+1) for i in n_repeat], xp=xp)


def neighbor_duos(cells: np.ndarray, positions: np.ndarray,
                  cutoff: float, repeat: np.ndarray,
                  i1: np.ndarray, xp=np):
    """Concatenate version of neighbor_duos_batch."""
    _, count = xp.unique(i1, return_counts=True)
    n_batch = len(count)
    n_atoms = max(count)
    head = cumsum_from_zero(count)
    index = xp.repeat(xp.arange(n_atoms)[xp.newaxis, :], n_batch, 0)
    padding = index >= count[:, xp.newaxis]
    index_batch = (index +
                   (xp.arange(n_batch) * n_atoms)[:, xp.newaxis])[~padding]
    r = xp.zeros((n_batch * n_atoms, 3))
    r[index_batch, :] = positions
    r = r.reshape((n_batch, n_atoms, 3))
    n, i, j, shift = neighbor_duos_batch(cells, r, cutoff,
                                         repeat, padding, xp)
    head_n = head[n]
    return i + head_n, j + head_n, shift


def neighbor_duos_batch(cells: np.ndarray, positions: np.ndarray,
                        cutoff: float, repeat: np.ndarray,
                        padding=None, xp=np
                        ) -> Tuple[np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]:
    """Compute pairs that are in neighbor.

    It only support the situation that all batches has similar cell.
    And all batches
    """
    assert repeat.shape == (3,)
    shifts = compute_shifts(repeat, xp=xp)
    n_batch = positions.shape[0]
    n_shifts = shifts.shape[0]
    n_atoms = positions.shape[1]
    assert cells.shape == (n_batch, 3, 3)
    assert shifts.shape == (n_shifts, 3)
    assert positions.shape == (n_batch, n_atoms, 3)

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


def neighbor_trios(i: np.ndarray, j: np.ndarray, xp=np) -> np.ndarray:
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
    assert i.shape == j.shape
    assert xp.all(xp.diff(i) >= 0)
    center, number = xp.unique(i, return_counts=True)
    m = int(xp.max(number))
    n = len(center)
    base = cumsum_from_zero(number, xp)[repeat_interleave(number * number, xp)]
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


def repeat_interleave(n: np.ndarray, xp=np):
    """Repeat arange.

    >>> n = np.array([1, 1, 2, 2, 3, 4])
    >>> center, number = np.unique(n, return_counts=True)
    >>> number
    ... [2 2 1 1]
    >>> repeat_interleave(number)
    ... [0 0 1 1 2 3]
    """
    n0 = len(n)
    n1 = int(xp.max(n))
    arange1, arange2, count = xp.broadcast_arrays(
        xp.arange(n1)[xp.newaxis, :],
        xp.arange(n0)[:, xp.newaxis],
        n[:, xp.newaxis]
    )
    mask = count > arange1
    ret = arange2[mask]
    return ret


def cumsum_from_zero(input_: np.ndarray, xp=np):
    """Like xp.cumsum. But start from 0.

    >>> n = np.array([1, 1, 2, 2, 3, 4])
    >>> center, number = np.unique(n, return_counts=True)
    >>> number
    ... [2 2 1 1]
    >>> cumsum_from_zero(number)
    ... [0 2 4 5]
    """
    cumsum = xp.cumsum(input_, axis=0)
    cumsum = xp.roll(cumsum, 1)
    cumsum[0] = 0
    return cumsum


def duo_index(num_elements: int, xp=np):
    """Duo index."""
    e = xp.arange(num_elements)
    p1, p2 = cartesian_product(e, e, xp=xp).T
    ret = xp.zeros([num_elements, num_elements], dtype=xp.int32)
    ret[p1, p2] = xp.arange(num_elements * num_elements)
    return ret


def trio_index(num_elements: int, xp=np):
    """Trio index."""
    e = xp.arange(num_elements)
    p1, p2, p3 = cartesian_product(e, e, e, xp=xp).T
    ret = xp.zeros([num_elements, num_elements, num_elements], dtype=xp.int32)
    ret[p1, p2, p3] = xp.arange(num_elements * num_elements * num_elements)
    return ret
