# *- coding:utf-8 -*-
"""Test chmd.neighbors by comparing with torchani."""
from unittest import TestCase
import numpy as np
import torch
import torchani
from chmd.utils.batchform import parallel_form
from chmd.functions.neighbors import neighbor_duos_to_flatten_form, neighbor_trios, neighbor_duos, compute_shifts, number_repeats
from chmd.math.xp import cumsum_from_zero, repeat_interleave


def fill(x, default):
    """Padding."""
    n = max(len(xi) for xi in x)
    return np.array([list(xi) + [default] * (n - len(xi)) for xi in x])


def sort2(x):
    """Sort matrix."""
    assert x.ndim == 2
    x = x.T
    for i in range(x.shape[1] - 1, -1, -1):
        if i > 5:
            raise RuntimeError()
        x = sorted(x, key=lambda xi: xi[i])
    return np.array(x).T


def neighbor_torch(mols, cutoff, order_of_symbols):
    """Calculate duo and trio using torchani."""
    cell = np.array(mols[0].cell)
    pbc = np.array(mols[0].pbc)
    positions = fill([mol.positions for mol in mols], [0.0, 0.0, 0.0])
    symbols = fill(np.array([mol.get_chemical_symbols() for mol in mols]), "")
    padding = symbols == ""
    # elements = symbols_to_elements(symbols, order_of_symbols)

    cell = torch.Tensor(cell)
    pbc = torch.BoolTensor(pbc)
    positions = torch.Tensor(positions)
    padding = torch.BoolTensor(padding)
    shifts = torchani.aev.compute_shifts(cell, pbc, cutoff)
    i2, j2, shift = torchani.aev.neighbor_pairs(
        padding, positions, cell, shifts, cutoff)
    ii = np.concatenate([i2, j2])
    jj = np.concatenate([j2, i2])
    ss = np.concatenate([shift, -shift])
    ijs2 = sort2(np.concatenate([np.array([ii, jj]), -ss.T], axis=0))
    c, a, b, s1, s2 = torchani.aev.triple_by_molecule(i2, j2)
    i3_t = c
    j3_t = np.where(s1 > 0, j2[a], i2[a])
    k3_t = np.where(s2 > 0, j2[b], i2[b])
    i3 = np.concatenate([i3_t, i3_t])
    j3 = np.concatenate([j3_t, k3_t])
    k3 = np.concatenate([k3_t, j3_t])
    ijk3 = sort2(np.array([i3, j3, k3]))
    return ijs2, ijk3

# def format_neighbor_duos(cells, positions, cutoff, shifts, valid):
#     dof = np.sum(valid, axis=1)
#     head = cumsum_from_zero(dof)
#     head = np.arange(positions.shape[0]) * positions.shape[1]
#     n2, i2, j2, s2 = neighbor_duos(cells, positions, cutoff, shifts, valid)
#     return i2 + head[n2], j2 + head[n2], s2

def transform_shifts_to_direct(cells, i2, s2, positions):
    inv = np.linalg.inv(cells)
    n_atoms = positions.shape[1]
    n2 = i2 // n_atoms
    c = inv[n2, :, :]
    return np.sum(s2[:, :, None] * c[:, :, :], axis=-2)


def neighbor_chmd(mols, cutoff, order_of_symbols):
    """Calculate duo and trio using chmd."""
    pbc = np.array(mols[0].pbc)
    positions_lst = [np.linalg.solve(atoms.cell.T, atoms.positions.T).T for atoms in mols]
    # positions_lst = [atoms.positions for atoms in mols]
    (positions,), valid = parallel_form.from_list([positions_lst], [0.0])
    cells = np.concatenate([atoms.cell[np.newaxis, :, :] for atoms in mols], axis=0)
    i2, j2, s2 = neighbor_duos_to_flatten_form(cells, positions, cutoff, pbc, valid)
    s2 = transform_shifts_to_direct(cells, i2, s2, positions)
    ijs2 = np.concatenate([np.array([i2, j2]), s2.T], axis=0)
    i3, j3 = neighbor_trios(i2, j2)
    assert np.all(i2[j3] == i2[i3])
    ijk3 = sort2(np.array([i2[i3], j2[i3], j2[j3]]))
    return ijs2, ijk3


def calc_index_batch(i1, xp=np):
    """Adapt the result of chmd and torchani.

    chmd's result is based on concatenate.
    torchani's result is based on batch and padding.
    To adsorb the difference, index is nessesary.
    It returns the difference index.

    """
    _, count = np.unique(i1, return_counts=True)
    n_batch = len(count)
    n_atoms = max(count)
    index = xp.repeat(xp.arange(n_atoms)[xp.newaxis, :], n_batch, 0)
    padding = index >= count[:, xp.newaxis]
    index_batch = (index
                   + (xp.arange(n_batch) * n_atoms)[:, xp.newaxis])[~padding]
    return index_batch


class TestPairs(TestCase):
    """Test neighbor_duo and neighbor_trio by comparing with torchani."""

    def test_pairs(self):
        """Test duo and trio."""
        from ase.build import molecule
        cell = np.array([[6.0, 2.0, 0.0],
                         [0.0, 6.0, 0.0],
                         [0.0, 0.0, 6.0]])
        cutoff = 9.0

        for x in (True, False):
            for y in (True, False):
                for z in (True, False):
                    pbc = np.array([x, y, z])
                    mols = [molecule("C2H4"), molecule("C2H6")]
                    order_of_symbols = ["C", "H"]
                    for mol in mols:
                        mol.pbc = pbc
                        mol.cell = cell
                    ijs2_torch, ijk3_torch = neighbor_torch(mols, cutoff,
                                                            order_of_symbols)

                    ijs2_chmd, ijk3_chmd = neighbor_chmd(mols, cutoff,
                                                         order_of_symbols)
                    self.assertTrue(np.allclose(ijs2_chmd, ijs2_torch))
                    self.assertTrue(np.allclose(ijk3_chmd, ijk3_torch))
