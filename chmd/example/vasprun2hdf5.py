"""Parse vasprun.xml and dump to hdf5."""
import os
import argparse
import logging
import numpy as np
from chainer.backend import get_array_module
from chmd.database.vasprun import read_trajectory
from chmd.database.hdf5 import HDF5Recorder
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


def main():
    """Main."""
    join = os.path.join
    parser = argparse.ArgumentParser(description='Convert vasprun.xml to hdf.')
    parser.add_argument('--inp', required=True,
                        help='Directory where vaspruns are puted.')
    parser.add_argument('--out', required=True,
                        help='The name of output hdf5.')
    parser.add_argument('--cutoff', type=float, required=True,
                        help='Cutoff radius.')
    parser.add_argument('--symbols', nargs='*', required=True,
                        help='The order of symbols.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    inp = args.inp
    out = args.out
    cutoff = args.cutoff
    symbols = np.array(args.symbols)
    vaspruns = os.listdir(inp)
    with HDF5Recorder(out, 'w') as f:
        f.attrs['cutoff'] = cutoff
        f.attrs['symbols'] = ' '.join(symbols)
        f.attrs['vaspruns'] = ' '.join(vaspruns)
        for path in vaspruns[:2]:
            logging.info('Parsing %s', path)
            trajectory = read_trajectory(join(inp, path))
            for atoms in trajectory:
                f.append(**preprocess(**atoms, order=symbols, cutoff=cutoff,
                                      pbc=np.array([True, True, True])))


if __name__ == '__main__':
    main()
