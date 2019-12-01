"""New ANI-1."""
import numpy as np
import chainer
from chainer import Chain, functions as F
from chainer.datasets import open_pickle_dataset, open_pickle_dataset_writer
from chainer.dataset.convert import to_device
from chainer.backend import get_array_module
from chmd.utils.batchform import parallel_form, series_form
from chmd.functions.neighbors import (neighbor_duos,
                                      number_repeats,
                                      compute_shifts,
                                      neighbor_duos_to_flatten_form
                                      )
from chmd.links.ani import ANI1AEV
from chmd.preprocess import symbols_to_elements


def concat_neighbors(n_batch, n_atoms, i2_seed, j2_seed, s2_seed):
    """Concatenate neighbors to flatten form."""
    raising_bottom = np.arange(n_batch) * n_atoms
    (i2_p, j2_p, s2), aff = series_form.from_list([i2_seed, j2_seed, s2_seed])
    i2 = i2_p + raising_bottom[aff]
    j2 = j2_p + raising_bottom[aff]
    return i2, j2, s2


class AddAEV(object):
    def __init__(self, params):
        self.aev_calc = ANI1AEV(params['num_elements'], **params['aev_params'])
        self.order = np.array(params['order'])
        self.pbc = params['pbc']
        self.cutoff = params['cutoff']
    
    def __call__(self, datas, device):
        add_elements(datas, self.order)
        add_neighbors(datas, self.cutoff, self.pbc, device)
        add_aev(datas, self.aev_calc, device)
        for data in datas:
            del data['i2']
            del data['j2']
            del data['s2']

def add_elements(datas, order):
    symbols = [data['symbols'] for data in datas]
    [sym], valid = parallel_form.from_list([symbols], [''])
    elem = symbols_to_elements(sym, order)
    for i, data in enumerate(datas):
        data['elements'] = elem[i][valid[i]]


def add_neighbors(datas, cutoff, pbc, device):
    direct = [data['positions'] for data in datas]
    cells = np.array([data['cell'] for data in datas])
    [direct], valid = parallel_form.from_list([direct], [0.0])
    cells = to_device(device, cells)
    direct = to_device(device, direct)
    pbc = to_device(device, pbc)
    valid = to_device(device, valid)
    xp = get_array_module(direct)
    repeat = xp.min(number_repeats(cells, pbc, cutoff), axis=0)
    shifts = compute_shifts(repeat)
    n2, i2, j2, s2 = neighbor_duos(cells, direct, cutoff, shifts, valid)
    for i, data in enumerate(datas):
        selecter = n2 == i
        data['i2'] = i2[selecter]
        data['j2'] = j2[selecter]
        data['s2'] = s2[selecter]


def add_aev(datas, aev_calc, device):
    direct = [data['positions'] for data in datas]
    cells = np.array([data['cell'] for data in datas])
    elements = [data['elements'] for data in datas]
    [direct, elements], valid = parallel_form.from_list(
        [direct, elements], [0.0, -1])
    cells = to_device(device, cells)
    direct = to_device(device, direct)
    elements = to_device(device, elements)
    valid = to_device(device, valid)
    xp = get_array_module(direct)
    cartesian = xp.sum(direct[:, :, :, None] * cells[:, None, :, :], axis=-2)
    n_batch, n_atoms, n_dim = cartesian.shape
    ri = cartesian.reshape((n_batch * n_atoms, n_dim))
    ei = elements.reshape((n_batch * n_atoms, ))
    i2_seed = [data['i2'] for data in datas]
    j2_seed = [data['j2'] for data in datas]
    s2_seed = [data['s2'] for data in datas]
    i2, j2, s2 = concat_neighbors(n_batch, n_atoms, i2_seed, j2_seed, s2_seed)
    aev = aev_calc(ri, ei, i2, j2, s2)  # (batch * atoms, features)
    n_feature = aev.shape[-1]
    aev_parallel = aev.reshape((n_batch, n_atoms, n_feature))
    aev_cpu = to_device(-1, aev_parallel.data)
    valid = to_device(-1, valid)
    for i, data in enumerate(datas):
        data['aev'] = aev_cpu[i][valid[i]]


# def add_elements_aev(aev_calc, datas, order, device, cutoff, pbc):
#     """Add aev to data."""
#     positions = [d['positions'] for d in datas]
#     symbols = [d['symbols'] for d in datas]
#     lst, valid = parallel_form.from_list([positions, symbols], [0.0, ''])

#     cells = to_device(device, np.array([d['cell'] for d in datas]))
#     positions = to_device(device, lst[0])
#     elements = to_device(device, symbols_to_elements(lst[1], order))
#     valid = to_device(device, valid)
#     n_batch, n_atoms, n_dim = positions.shape
#     i2, j2, s2 = neighbor_duos_to_flatten_form(
#         cells, positions, cutoff, pbc, valid
#     )
#     xp = get_array_module(positions)
#     cart = xp.sum(
#         positions[:, :, :, None] * cells[:, None, :, :],
#         axis=-2)
#     # batch, atoms, dim, dim
#     ri = cart.reshape((n_batch * n_atoms, n_dim))
#     ei = elements.reshape((n_batch * n_atoms))
#     aev = aev_calc(ri, ei, i2, j2, s2)  # (batch * atoms, features)
#     _, n_feature = aev.shape
#     aev_parallel = aev.reshape((n_batch, n_atoms, n_feature))
#     aev_cpu = to_device(-1, aev_parallel.data)
#     valid = to_device(-1, valid)
#     elements = to_device(-1, elements)
#     for i, data in enumerate(datas):
#         data['aev'] = aev_cpu[i][valid[i]]
#         data['elements'] = elements[i][valid[i]]


def preprocess(inp_path, out_path, batch_size, pbc, cutoff, device, trans):
    """Preprocess data."""
    datasets = {}
    with open_pickle_dataset(inp_path) as fi:
        with open_pickle_dataset_writer(out_path) as fo:
            for i, data in enumerate(fi):
                cell = data['cell']
                repeats = tuple(number_repeats(
                    cell, pbc, cutoff).astype(np.int64).tolist())
                if repeats not in datasets:
                    datasets[repeats] = [data]
                else:
                    datasets[repeats].append(data)
                    if len(datasets[repeats]) > batch_size:
                        datas = datasets.pop(repeats)
                        print('{} process for {}'.format(i, repeats))
                        trans(datas, device)
                        for d in datas:
                            fo.write(d)
            keys = list(datasets.keys())
            for repeats in keys:
                datas = datasets.pop(repeats)
                print('process for {}'.format(repeats))
                trans(datas, device)
                for d in datas:
                    fo.write(d)


class ANI1EnergyLoss(Chain):
    """ANI1 Energy loss only from aevs."""

    def __init__(self, models):
        """Initialize."""
        super().__init__()
        with self.init_scope():
            self.nn = models

    def forward(self, aevs, elements, energies, valid):
        """Calculate energy loss from aev. Input is assumed to be parallel form.

        Parameters
        ----------
        aevs: float(n_batch, n_atoms, n_features)
        elements: int(n_batch, n_atoms)
        energies: float(n_batch,)
        valids: bool(n_batch, n_atoms)

        """
        xp = self.xp
        n_ensemble = len(self.nn)
        n_batch, n_atoms, n_features = aevs.shape
        flatten_aev = F.reshape(aevs, (n_batch * n_atoms, n_features))
        flatten_elements = F.reshape(elements, (n_batch * n_atoms))
        flatten_predict = self.nn(flatten_aev, flatten_elements)
        predict_atomwise = F.reshape(flatten_predict,
                                     (n_ensemble, n_batch, n_atoms))
        predict = F.sum(predict_atomwise, 2)
        assert xp.all(predict.data[~valid] == 0.0)
        assert predict.shape == (n_ensemble, n_batch)
        diff = predict - energies[xp.newaxis, :, :]
        power_diff = diff * diff
        mean_power_diff = F.mean(power_diff, 1)
        assert mean_power_diff.shape == (n_ensemble,)
        return F.sum(mean_power_diff)


@chainer.dataset.converter()
def converter_concat_neighbors(batch, device):
    """Not tested yet."""
    lst_keys = ['aev', 'elements']
    paddings = [0.0, -1]
    lst = [[atoms[key] for atoms in batch] for key in lst_keys]
    parallels, valid = parallel_form.from_list(lst, paddings)
    aevs = parallels[0]
    elements = parallels[1]
    energies = np.array([atoms['energy'] for atoms in batch])

    dtype = chainer.config.dtype
    return_dict = {
        'aevs': aevs.astype(dtype),
        'elements': elements,
        'valid': valid,
        'energies': energies.astype(dtype),
    }
    for key in return_dict:
        return_dict[key] = to_device(device, return_dict[key])
    return return_dict
