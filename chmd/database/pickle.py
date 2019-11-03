"""Read vasprun and dump to pickle."""
import os
import logging
import json
import numpy as np
from chainer.backend import get_array_module
from chainer.datasets import open_pickle_dataset_writer, open_pickle_dataset
from chmd.database.vasprun import read_trajectory
from chmd.preprocess import symbols_to_elements
from chmd.neighbors import number_repeats, neighbor_duos


def neighbor_duos_serial(cell: np.ndarray, positions: np.ndarray,
                         pbc: np.ndarray, cutoff: float):
    """Compute pairs that are in neighbor in an atoms."""
    xp = get_array_module(cell)
    repeat = number_repeats(cell, pbc, cutoff)
    i2, j2, s2 = neighbor_duos(
        cell[xp.newaxis, :, :],
        positions,
        cutoff,
        repeat,
        xp.zeros(positions.shape[0], dtype=xp.int32))
    return i2, j2, s2


def preprocess(cell, pbc, positions, energy, forces, symbols,
               order, cutoff):
    """Preprocess."""
    elements = symbols_to_elements(symbols, order)
    i2, j2, s2 = neighbor_duos_serial(cell, positions, pbc, cutoff)
    return {'cell': cell,
            'positions': positions,
            'elements': elements,
            'i2': i2,
            'j2': j2,
            's2': s2,
            'energy': energy,
            'forces': forces,
            }


def initialize_manager(cutoff, symbols):
    """Leave only cutoff and symbols."""
    manage = dict()
    manage['cutoff'] = cutoff
    manage['symbols'] = symbols
    manage['remain'] = []
    manage['train'] = []
    manage['drain'] = []
    manage['parsed'] = []
    return manage


def make_dataset(vaspruns_dir, pklout, manout, pklinp=None, maninp=None):
    """Build data set from scratch or from pklinp and unread files."""
    with open(maninp, 'r') as f:
        manage = json.load(f)
    cutoff = manage['cutoff']
    symbols = np.array(manage['symbols'])
    pbc = np.array([True, True, True])
    vaspruns = os.listdir(vaspruns_dir)
    with open_pickle_dataset_writer(pklout) as po:
        if pklinp:
            with open_pickle_dataset(pklinp) as pi:
                n = len(pi)  # index in po.
                for data in pi:
                    po.write(data)
        else:
            manage = initialize_manager(cutoff, symbols.tolist())
            n = 0  # index in po.
        for vr in vaspruns:
            if vr in manage['parsed']:
                logging.info('Skip %s', vr)
                continue
            logging.info('Parsing %s', vr)
            manage['parsed'].append(vr)
            trj = read_trajectory(os.path.join(vaspruns_dir, vr))
            for atoms in trj:
                manage['remain'].append(n)
                n += 1  # index in po.
                processed = preprocess(**atoms,
                                       pbc=pbc, order=symbols, cutoff=cutoff)
                po.write(processed)
    with open(manout, 'w') as f:
        json.dump(manage, f)
