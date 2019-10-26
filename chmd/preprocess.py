import numpy as np
from chmd.neighbors import number_repeats, neighbor_duos
from chainer.backend import get_array_module


def symbols_to_elements(symbols: np.ndarray,
                        order_of_symbols: np.ndarray) -> np.ndarray:
    """Convert symbols to uniqued elements number.

    Parameters
    ----------
    symbols         : symbols that consists molecular
    order_of_symbols: unique symbols

    Returns
    -------
    elements: where order_of_symbols[elements] == symbols

    """
    unique, inverse = np.unique(symbols, return_inverse=True)
    elements = np.argsort(unique)[np.argsort(order_of_symbols)][inverse]
    assert np.all(order_of_symbols[elements] == symbols)
    assert np.all(np.sort(unique) == np.sort(order_of_symbols))
    return elements
    # ret = np.full(symbols.shape, -1)
    # for i, s in enumerate(order_of_symbols):
    #     ret = np.where(symbols == s, i, ret)
    # return ret


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


def preprocess_energy_forces(cell, pbc, positions, energy, forces, symbols, order, cutoff):
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


def dict_to_hdf5(root, name, dic):
    grp = root.create_group(name)
    for key, val in dic.items():
        dataset = grp.create_dataset(key, val.shape, dtype=val.dtype)
        dataset[...] = dic[key]


def hdf5_to_dict(hdf5):
    dic = dict()
    for key in hdf5:
        dic[key] = hdf5[key][()]
    return dic
