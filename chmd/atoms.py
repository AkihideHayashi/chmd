# *- coding:utf-8 -*-
"""Atoms class and formatting functions."""
from typing import List, Tuple, Dict
import numpy as np
from chmd.neighbors import number_repeats, neighbor_duos
from chmd.neighbors import cumsum_from_zero


def symbols_to_elements(symbols: np.ndarray,
                        order_of_symbols: np.ndarray) -> np.ndarray:
    """Convert symbols to uniqued elements number.

    Parameters
    ----------
    symbols         : symbols that consists molecular
    order_of_symbols: unique symbols

    Returns
    -------
    elements

    """
    ret = np.full(symbols.shape, -1)
    for i, s in enumerate(order_of_symbols):
        ret = np.where(symbols == s, i, ret)
    return ret


class Atoms(object):
    """The atoms class that hold infomations including adjacent."""

    def __init__(self, symbols: np.ndarray, positions: np.ndarray,
                 cell: np.ndarray, pbc: np.ndarray):
        """Initialize.

        Parameters
        ----------
        symbols : str[n_atoms]
        positions : float[n_atoms, 3]
        cell: float[3, 3]
        pbc : bool[3]

        """
        n_atoms = len(symbols)
        assert symbols.shape == (n_atoms,)
        assert positions.shape == (n_atoms, 3)
        assert cell.shape == (3, 3)
        assert pbc.shape == (3,)
        self.symbols = symbols
        self.positions = positions
        self.cell = cell
        self.pbc = pbc
        self.elements = None
        self.i = None
        self.j = None
        self.shift = None

    def set_elements(self, order_of_symbols: np.ndarray):
        """Set elements that are not initialized in __init__."""
        self.elements = symbols_to_elements(self.symbols, order_of_symbols)

    def set_pairs(self, cutoff: float):
        """Set pairs using neighbor pairs. It is usefull in learning stage."""
        repeat = number_repeats(self.cell, self.pbc, cutoff)
        self.i, self.j, self.shift = neighbor_duos(
            self.cell[np.newaxis, :, :],
            self.positions,
            cutoff,
            repeat,
            np.zeros(self.symbols.shape, dtype=np.int32),
        )

    @staticmethod
    def from_ase(atoms):
        """Initialize Atoms class from ase.Atoms.

        Parameters
        ----------
        atoms : ase.Atoms

        """
        return Atoms(np.array(atoms.get_chemical_symbols()),
                     atoms.positions, np.array(atoms.cell), atoms.pbc)

    def __len__(self):
        """Return the number of atoms that consist it."""
        return len(self.symbols)


def get_solo_index(mols: List[Atoms]) -> np.ndarray:
    """Get solo index.

    Parameters
    ----------
    mols : List of Atoms.

    Returns
    -------
    i1: Index that maps each atoms to minibatch

    """
    i1 = np.concatenate([i * np.ones_like(atoms.elements)
                         for i, atoms in enumerate(mols)])
    return i1


def get_duo_index(mols: List[Atoms]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get solo index and pair index from list of atoms.

    Parameters
    ----------
    mols : List of Atoms.

    Returns
    -------
    (i1, i2, j2, s2)
    i2: Index that maps head of bond to atoms.
    j2: Index that maps tail of bond to atoms.
    s2: Shift of bond

    """
    head = cumsum_from_zero(np.array([len(atoms) for atoms in mols]))
    n2 = np.concatenate([i * np.ones_like(atoms.i)
                         for i, atoms in enumerate(mols)])
    i2 = np.concatenate([atoms.i for atoms in mols]) + head[n2]
    j2 = np.concatenate([atoms.j for atoms in mols]) + head[n2]
    s2 = np.concatenate([atoms.shift for atoms in mols])
    return i2, j2, s2


def get_items(mols: List[Atoms], requires: List[str]) -> Dict:
    """Get cells, positions, elements and indices from list of atoms.

    Parameters
    ----------
    mols : List of Atoms.
    requres : List of keys.
      cells, positions, elements, solo, duo are acceptable.

    Returns
    -------
    dict

    """
    items = {}
    for req in requires:
        if req == 'cells':
            items[req] = np.concatenate(
                [atoms.cell[np.newaxis, :, :] for atoms in mols]
                )
        elif req == 'positions':
            items[req] = np.concatenate([atoms.positions for atoms in mols])
        elif req == 'elements':
            items[req] = np.concatenate([atoms.elements for atoms in mols])
        elif req == 'solo':
            items[req] = get_solo_index(mols)
        elif req == 'duo':
            items[req] = get_duo_index(mols)
        else:
            KeyError(req)
    return items
