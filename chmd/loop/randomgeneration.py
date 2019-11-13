import numpy as np
import chainer
from chmd.functions.neighbors import number_repeats, compute_shifts


def random_coordinates(cell, min_distance, n_atoms, max_cycle):
    pbc = np.array([True, True, True])
    repeat = number_repeats(cell, pbc, min_distance * 2)
    shifts = compute_shifts(repeat)  # (n_shifts, n_dim)
    n_dim = 3
    dtype = chainer.config.dtype
    positions = np.array([np.random.random(n_dim).astype(dtype)]) # (n_atoms, n_dim)
    for _ in range(max_cycle):
        new_positions = np.random.random(n_dim).astype(dtype)
        # (n_atoms, n_shifts, n_dim)
        direct_vec = positions[:, np.newaxis, :] - new_positions[np.newaxis, np.newaxis, :] - shifts[np.newaxis, :, :]
        # (n_atoms, n_shifts, n_dim)
        cartesian_vec = np.sum(direct_vec[:, :, np.newaxis, :] * cell[np.newaxis, np.newaxis, :, :], axis=-2)
        distances = np.sqrt(np.sum(cartesian_vec * cartesian_vec, axis=2))
        minimum_distances = np.min(distances)
        if minimum_distances < min_distance:
            continue
        positions = positions.tolist()
        positions.append(new_positions)
        positions = np.array(positions)
    return positions

def random_symbols(order, ratio, n):
    seed = np.random.random((len(order), n))  # (n_order, n_atoms)
    rand = ratio[:, np.newaxis] * seed
    return order[np.sort(np.argmax(rand, axis=0))]


