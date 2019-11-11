"""Converter."""
import numpy as np
import chainer
from chainer.dataset.convert import to_device
from chmd.functions.neighbors import repeat_interleave


def concat_converter(batch, device=None, padding=None):
    assert padding is None
    n_atoms = np.array([len(atoms['elements']) for atoms in batch])
    affiliations = repeat_interleave(n_atoms)
    positions = np.concatenate([atoms['positions'] for atoms in batch])
    cells = np.concatenate([atoms['cell'][None, :, :] for atoms in batch])
    elements = np.concatenate([atoms['elements'] for atoms in batch])
    energies = np.array([atoms['energy'] for atoms in batch])
    forces = np.concatenate([atoms['forces'] for atoms in batch])
    dtype = chainer.config.dtype
    ret = {'elements': elements,
           'cells': cells.astype(dtype),
           'positions': positions.astype(dtype),
           'affiliations': affiliations,
           'energies': energies.astype(dtype),
           'forces': forces.astype(dtype)}
    for key in ret:
        ret[key] = to_device(device, ret[key])
    return ret

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
#         'ei': elements,
#         'ci': cells.astype(dtype),
#         'ri': positions.astype(dtype),
#         'i1': i1,
#         'i2': i2,
#         'j2': j2,
#         's2': s2.astype(dtype),
#         'e': energies.astype(dtype),
#         'f': forces.astype(dtype),
#     }
#     for key in ret:
#         ret[key] = to_device(device, ret[key])
#     return ret
# 