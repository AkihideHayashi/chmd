"""Converter."""
import numpy as np
import chainer
from chainer.dataset.convert import to_device
from chmd.functions.neighbors import cumsum_from_zero, repeat_interleave
from chmd.utils.batchform import parallel_form, flatten_form, series_form


# def concat_converter(batch, device=None, padding=None):
#     assert padding is None
#     n_atoms = np.array([len(atoms['elements']) for atoms in batch])
#     affiliations = repeat_interleave(n_atoms)
#     positions = np.concatenate([atoms['positions'] for atoms in batch])
#     cells = np.concatenate([atoms['cell'][None, :, :] for atoms in batch])
#     elements = np.concatenate([atoms['elements'] for atoms in batch])
#     energies = np.array([atoms['energy'] for atoms in batch])
#     forces = np.concatenate([atoms['forces'] for atoms in batch])
#     dtype = chainer.config.dtype
#     ret = {'elements': elements,
#            'cells': cells.astype(dtype),
#            'positions': positions.astype(dtype),
#            'affiliations': affiliations,
#            'energies': energies.astype(dtype),
#            'forces': forces.astype(dtype)}
#     for key in ret:
#         ret[key] = to_device(device, ret[key])
#     return ret

def concat_converter(batch, device=None, padding=None):
    """Not tested yet."""
    lst_keys = ['positions', 'elements', 'energies', 'forces']
    paddings = [0.0, -1, 0.0, 0.0]
    lst = [[atoms[key] for atoms in batch] for key in lst_keys]
    cells = np.array([atoms['cell'] for atoms in batch])
    parallels, valid = parallel_form.from_list(lst, paddings)
    return_dict = {key: x for key, x in zip(lst_keys, parallels)}
    return_dict['valid'] = valid
    return_dict['cells'] = cells
    if 'i2' in batch and 'j2' in batch and 's2' in batch[0]:
        n_batch, n_atoms = parallels[1].shape[:2]
        raising_bottom = np.arange(n_batch) * n_atoms
        (i2_seed, j2_seed, s2), aff = series_form.from_list([
            [atoms['i2'] for atoms in batch],
            [atoms['j2'] for atoms in batch],
            [atoms['s2'] for atoms in batch],
        ])
        i2 = i2_seed + raising_bottom[aff]
        j2 = j2_seed + raising_bottom[aff]
        return_dict['i2'] = i2
        return_dict['j2'] = j2
        return_dict['s2'] = s2
    dtype = chainer.config.dtype
    for key in return_dict:
        if return_dict[key].dtype in (np.float16, np.float32, np.float64):
            return_dict[key] = return_dict[key].astype(dtype)
        return_dict[key] = to_device(device, return_dict[key])
    return return_dict


# def concat_converter(batch, device=None, padding=None):
#     """Concat converter which uses index."""
#     assert padding is None
#     i1 = np.concatenate([i * np.ones_like(atoms['elements'])
#                          for i, atoms in enumerate(batch)])
#     head = cumsum_from_zero(np.array([len(atoms['elements'])
#                                       for atoms in batch]))
#     n2 = np.concatenate([i * np.ones_like(atoms['i2'])
#                          for i, atoms in enumerate(batch)])
#     i2 = np.concatenate([atoms['i2'] for atoms in batch]) + head[n2]
#     j2 = np.concatenate([atoms['j2'] for atoms in batch]) + head[n2]
#     s2 = np.concatenate([atoms['s2'] for atoms in batch])
#     positions = np.concatenate([atoms['positions'] for atoms in batch])
#     cells = np.concatenate([atoms['cell'][None, :, :] for atoms in batch])
#     elements = np.concatenate([atoms['elements'] for atoms in batch])
#     energies = np.array([atoms['energy'] for atoms in batch])
#     forces = np.concatenate([atoms['forces'] for atoms in batch])
#     dtype = chainer.config.dtype
#     ret = {
#         'elements': elements,
#         'cells': cells.astype(dtype),
#         'positions': positions.astype(dtype),
#         'i1': i1,
#         'i2': i2,
#         'j2': j2,
#         's2': s2.astype(dtype),
#         'energies': energies.astype(dtype),
#         'forces': forces.astype(dtype),
#     }
#     for key in ret:
#         ret[key] = to_device(device, ret[key])
#     return ret
