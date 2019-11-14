"""Converter."""
import numpy as np
import chainer
from chainer.dataset.convert import to_device
from chmd.functions.neighbors import neighbor_duos_to_flatten_form
from chmd.utils.batchform import parallel_form, flatten_form, series_form


def concat_neighbors(n_batch, n_atoms, i2_seed, j2_seed, s2_seed):
    """Concatenate neighbors in parallel form."""
    raising_bottom = np.arange(n_batch) * n_atoms
    (i2_p, j2_p, s2), aff = series_form.from_list([i2_seed, j2_seed, s2_seed])
    i2 = i2_p + raising_bottom[aff]
    j2 = j2_p + raising_bottom[aff]
    return i2, j2, s2


class ConverterGenerateNeighbors(object):
    """Not Tested Yet."""

    def __init__(self, cutoff, pbc, **kwargs):
        """Not Tested Yet."""
        import warnings
        warnings.warn('ConvergerGenerateNeighbors have been not tested yet.')
        self.cutoff = cutoff
        self.pbc = np.array(pbc)

    @chainer.dataset.converter()
    def __call__(self, batch, device):
        """Not tested yet."""
        lst_keys = ['positions', 'elements', 'forces']
        paddings = [0.0, -1, 0.0]
        lst = [[atoms[key] for atoms in batch] for key in lst_keys]
        parallels, valid = parallel_form.from_list(lst, paddings)
        positions = parallels[0]
        elements = parallels[1]
        forces = parallels[2]
        cells = np.array([atoms['cell'] for atoms in batch])
        energies = np.array([atoms['energy'] for atoms in batch])
        dtype = chainer.config.dtype
        return_dict = {
            'positions': positions.astype(dtype),
            'forces': forces.astype(dtype),
            'elements': elements,
            'valid': valid,
            'cells': cells.astype(dtype),
            'energies': energies.astype(dtype),
        }
        for key in return_dict:
            return_dict[key] = to_device(device, return_dict[key])
        i2, j2, s2 = neighbor_duos_to_flatten_form(return_dict['cells'],
                                                   return_dict['positions'],
                                                   self.cutoff,
                                                   self.pbc,
                                                   return_dict['valid'],
                                                   )
        return_dict['i2'] = i2
        return_dict['j2'] = j2
        return_dict['s2'] = s2
        return return_dict


@chainer.dataset.converter()
def converter_concat_neighbors(batch, device):
    """Not tested yet."""
    lst_keys = ['positions', 'elements', 'forces']
    paddings = [0.0, -1, 0.0]
    lst = [[atoms[key] for atoms in batch] for key in lst_keys]
    parallels, valid = parallel_form.from_list(lst, paddings)
    positions = parallels[0]
    elements = parallels[1]
    forces = parallels[2]
    cells = np.array([atoms['cell'] for atoms in batch])
    energies = np.array([atoms['energy'] for atoms in batch])
    n_batch, n_atoms = parallels[1].shape[:2]
    i2, j2, s2 = concat_neighbors(n_batch, n_atoms,
                                  [atoms['i2'] for atoms in batch],
                                  [atoms['j2'] for atoms in batch],
                                  [atoms['s2'] for atoms in batch],
                                  )

    dtype = chainer.config.dtype
    return_dict = {
        'positions': positions.astype(dtype),
        'forces': forces.astype(dtype),
        'elements': elements,
        'valid': valid,
        'cells': cells.astype(dtype),
        'energies': energies.astype(dtype),
        'i2': i2,
        'j2': j2,
        's2': s2.astype(dtype),
    }
    for key in return_dict:
        return_dict[key] = to_device(device, return_dict[key])
    return return_dict


# def concat_converter(batch, device=None, padding=None):
#     """Not tested yet."""
#     lst_keys = ['positions', 'elements', 'forces']
#     paddings = [0.0, -1, 0.0]
#     lst = [[atoms[key] for atoms in batch] for key in lst_keys]
#     cells = np.array([atoms['cell'] for atoms in batch])
#     energies = np.array([atoms['energy'] for atoms in batch])
#     parallels, valid = parallel_form.from_list(lst, paddings)
#     return_dict = {key: x for key, x in zip(lst_keys, parallels)}
#     return_dict['valid'] = valid
#     return_dict['cells'] = cells
#     return_dict['energies'] = energies
#     if 'i2' in batch[0] and 'j2' in batch[0] and 's2' in batch[0]:
#         n_batch, n_atoms = parallels[1].shape[:2]
#         raising_bottom = np.arange(n_batch) * n_atoms
#         (i2_seed, j2_seed, s2), aff = series_form.from_list([
#             [atoms['i2'] for atoms in batch],
#             [atoms['j2'] for atoms in batch],
#             [atoms['s2'] for atoms in batch],
#         ])
#         i2 = i2_seed + raising_bottom[aff]
#         j2 = j2_seed + raising_bottom[aff]
#         return_dict['i2'] = i2
#         return_dict['j2'] = j2
#         return_dict['s2'] = s2
#     dtype = chainer.config.dtype
#     for key in return_dict:
#         if return_dict[key].dtype in (np.float16, np.float32, np.float64):
#             return_dict[key] = return_dict[key].astype(dtype)
#         return_dict[key] = to_device(device, return_dict[key])
#     return return_dict
