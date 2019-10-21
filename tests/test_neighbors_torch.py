# *- coding:utf-8 -*-
"""Test chmd.neighbors by comparing with torchani."""
from unittest import TestCase
import numpy as np
from chmd.neighbors import number_repeats, neighbor_duos, neighbor_trios
from chmd.atoms import Atoms, get_items
import torch
import torchani


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


def neighbor_chmd(mols, cutoff, order_of_symbols):
    """Calculate duo and trio using chmd."""
    def inner(cells, positions, elements, solo):
        i1 = solo
        index_batch = calc_index_batch(i1)
        pbc = mols[0].pbc
        repeat = np.max(number_repeats(cells, pbc, cutoff), axis=0)
        i2, j2, s2 = neighbor_duos(cells, positions, cutoff, repeat, i1)
        ijs2 = sort2(np.concatenate(
            [np.array([index_batch[i2], index_batch[j2]]), s2.T], axis=0))
        a, b = neighbor_trios(i2, j2)
        c = i2[a]
        i3, j3, k3 = c, j2[a], j2[b]
        ijk3 = sort2(
            np.array([index_batch[i3], index_batch[j3], index_batch[k3]]))
        return ijs2, ijk3

    for mol in mols:
        mol.set_elements(order_of_symbols)
        # mol.set_pairs(cutoff)
    return inner(**get_items(mols, ["cells", "positions", "elements", "solo"]))


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

                    mols_chmd = [Atoms.from_ase(mol) for mol in mols]
                    ijs2_chmd, ijk3_chmd = neighbor_chmd(mols_chmd, cutoff,
                                                         order_of_symbols)
                    self.assertTrue(np.allclose(ijs2_chmd, ijs2_torch))
                    self.assertTrue(np.allclose(ijk3_chmd, ijk3_torch))
