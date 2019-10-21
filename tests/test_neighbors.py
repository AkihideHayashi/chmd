
# *- coding:utf-8 -*-
"""Test chmd.neighbors."""
from unittest import TestCase
import numpy as np
from chainer import Variable
from chmd.neighbors import neighbor_trios, distance, distance_angle
from chmd.atoms import Atoms, get_items


class TestPairs(TestCase):
    """Test."""

    def test_distance(self):
        """Check distance and angles is valid."""
        from ase.build import molecule

        def inner(cells, elements, positions, solo, duo):
            i1 = solo
            i2, j2, s2 = duo
            a3, b3 = neighbor_trios(i2, j2)
            self.assertTrue(np.allclose(i2[a3], i2[b3]))
            c3 = i2[a3]
            cells = Variable(cells)
            positions = Variable(positions)
            rij2 = distance(cells, positions, i1, i2, j2, s2)
            self.assertTrue(np.all(rij2.data < cutoff))
            rij3, rik3, costhetaijk3 = distance_angle(cells, positions,
                                                      i1,
                                                      i2, j2, s2,
                                                      c3, a3, b3
                                                      )
            self.assertTrue(np.allclose(rij3.data, rij2.data[a3]))
            self.assertTrue(np.allclose(rik3.data, rij2.data[b3]))
            self.assertTrue(np.all(costhetaijk3.data <= 1.0000000001))
            self.assertTrue(np.all(costhetaijk3.data >= -1.0000000001))

        cells = np.array([
            [[6.0, 2.0, 0.0],
             [0.0, 6.0, 0.0],
             [0.0, 0.0, 6.0]],
            [[6.0, 2.0, 0.0],
             [0.0, 6.0, 0.0],
             [0.0, 0.0, 6.0]],
        ])
        cutoff = 9.0
        pbc = np.array([True, True, True])
        mols_ase = [molecule("C2H4"), molecule("C2H6")]
        order_of_symbols = ["C", "H"]
        for mol, cell in zip(mols_ase, cells):
            mol.cell = cell
            mol.pbc = pbc
        mols = [Atoms.from_ase(atoms) for atoms in mols_ase]
        for mol in mols:
            mol.set_elements(order_of_symbols)
            mol.set_pairs(cutoff)
        inner(**get_items(mols,
                          ["cells", "elements", "positions", "solo", "duo"]))
