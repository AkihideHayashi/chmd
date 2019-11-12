# *- coding:utf-8 -*-
"""Test chmd.neighbors."""
from unittest import TestCase
import numpy as np
from chmd.utils.batchform import parallel_form, series_form, list_form


class TestPairs(TestCase):
    def test(self):
        n_dim = 3
        max_n_atoms = 5
        n_batch = 10
        n_atoms = np.random.choice(
            np.arange(max_n_atoms // 2, max_n_atoms), n_batch)
        list_positions = [np.random.random((n_a, n_dim))
                          for n_a in n_atoms]
        list_elements = [np.random.random((n_a)) for n_a in n_atoms]
        (para_pos, para_el), valid = parallel_form.from_list(
            [list_positions, list_elements], 0.0)
        (seri_pos, seri_el), affil = series_form.from_list(
            [list_positions, list_elements])
        (x, y), z = series_form.from_parallel([para_pos, para_el], valid)
        self.assertTrue(np.allclose(seri_pos, x))
        self.assertTrue(np.allclose(seri_el, y))
        self.assertTrue(np.allclose(affil, z))
        (x, y), z = parallel_form.from_series((seri_pos, seri_el), affil, 0.0)
        self.assertTrue(np.allclose(para_pos, x))
        self.assertTrue(np.allclose(para_el, y))
        self.assertTrue(np.allclose(valid, z))
        x, y = list_form.from_series((seri_pos, seri_el), affil)
        for i, j in zip(list_positions, x):
            self.assertTrue(np.allclose(i, j))
        for i, j in zip(list_elements, y):
            self.assertTrue(np.allclose(i, j))
        x, y = list_form.from_parallel((para_pos, para_el), valid)
        for i, j in zip(list_positions, x):
            self.assertTrue(np.allclose(i, j))
        for i, j in zip(list_elements, y):
            self.assertTrue(np.allclose(i, j))
