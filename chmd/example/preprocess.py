import numpy as np
from chainer.datasets import open_pickle_dataset, open_pickle_dataset_writer
from chmd.preprocess import symbols_to_elements
from chmd.functions.neighbors import neighbor_duos, number_repeats, compute_shifts

def batch_neighbor_duos(cell, positions, cutoff):
    pbc = np.array([True, True, True])
    repeat = number_repeats(cell, pbc, cutoff)
    shifts = compute_shifts(repeat)
    valid = np.full(positions.shape[:-1], True)
    n2, i2, j2, s2 = neighbor_duos(cell[None, :, :], positions[None, :, :], cutoff, shifts, valid[None, :])
    assert np.all(n2 == 0)
    return i2, j2, s2


def main():
    order = np.array(['H', 'C', 'Pt'])
    inp_path = '../../../data/seed.pkl'
    out_path = '../../../data/processed.pkl'
    cutoff = 9.0
    with open_pickle_dataset(inp_path) as fi:
        with open_pickle_dataset_writer(out_path) as fo:
            for i, data in enumerate(fi):
                if i % 10 == 0:
                    print(i, flush=True)
                data['elements'] = symbols_to_elements(data['symbols'], order)
                i2, j2, s2 = batch_neighbor_duos(data['cell'], data['positions'], cutoff)
                data['i2'] = i2
                data['j2'] = j2
                data['s2'] = s2
                fo.write(data)

if __name__ == "__main__":
    main()