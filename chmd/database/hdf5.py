"""Use hdf5 as database.

All datas are expressed as group.
"""
import numpy as np
import h5py


class HDF5Recorder(object):
    """HDF5 atoms recorder."""

    def __init__(self, path, mode, complib='gzip', complevel=6):
        """Initializer."""
        self.path = path
        self.mode = mode
        self.clib = complib
        self.clev = complevel
        self.file = None

    @property
    def attrs(self):
        """Attrs of hdf5 file it self."""
        return self.file.attrs

    def __enter__(self):
        """Hold hdf5 file."""
        self.file = h5py.File(self.path, self.mode)
        return self

    def __exit__(self, ex_type, ex_value, trace):
        """Release hdf5 file."""
        self.file.close()
        self.file = None

    def readable_check(self):
        """Readable now."""
        assert self.file is not None

    def writable_check(self):
        """Writable now."""
        assert self.mode in ('w', 'a', 'r+', 'w-') and self.file is not None

    def __len__(self):
        """Return number of files."""
        self.readable_check()
        return len(self.file)

    def append(self, **kwargs):
        """Add atoms."""
        self.writable_check()
        name = str(len(self))
        grp = self.file.create_group(name)
        for key, val in kwargs.items():
            if np.issubdtype(val.dtype, np.str_):
                grp.create_dataset(key, data=val.astype(
                    h5py.special_dtype(vlen=str)))
            else:
                grp.create_dataset(key, data=val)

    def __getitem__(self, key):
        """Get item from key."""
        if isinstance(key, (list,)):
            return (read_hdf5(self.file[k]) for k in key)
        else:
            return read_hdf5(self.file[key])

    def __iter__(self):
        """Iterate over key."""
        yield from self.file


def read_hdf5(group):
    """Convert group to dict[str, ndarray]."""
    return {key: val[()] for key, val in group.items()}

# def dict_to_hdf5(root, name, dic, **attrs):
#     """Convert dict to hdf5 group."""
#     grp = root.create_group(name)
#     for key in attrs:
#         grp.attrs[key] = attrs[key]
#     for key, val in dic.items():
#         dataset = grp.create_dataset(key, val.shape, dtype=val.dtype)
#         dataset[...] = dic[key]
# 
# 
# def hdf5_to_dict(path):
#     """Read hdf5 group and convert to dict."""
#     dic = dict()
#     for key in path:
#         dic[key] = path[key][()]
#     return dic, dict(path.attrs)
# 
# 
# def get_keys(path, **kwargs):
#     """Read and get hdf5 groups."""
#     with h5py.File(path, 'r') as f:
#         for key in f:
#             data = f[key]
#             for k, v in kwargs.items():
#                 if data.attrs[k] != v:
#                     break
#             else:
#                 yield key
# 
# 
# def set_attrs(path, keys, **kwargs):
#     """Set attrs for keys."""
#     with h5py.File(path, 'r') as f:
#         for key in keys:
#             for k, v in kwargs.items():
#                 f[key].attrs[k] = v
# 
# 
# def get_datas(path, keys):
#     """Read and get hdf5 data."""
#     with h5py.File(path, 'r') as f:
#         for key in keys:
#             data, _ = hdf5_to_dict(f[key])
#             yield data
# 
# 
# def len_hdf5(path):
#     """Return the number of data."""
#     with h5py.File(path, 'r') as f:
#         return len(f)
# 
# 
# class ConcatHDF5(object):
#     """Concat."""
# 
#     def __init__(self, hdf5):
#         """Initializer."""
#         self.hdf5 = hdf5
# 
#     def __call__(self, batch, device=None, padding=None):
#         """Get data from hdf5 and concat.
# 
#         Parameters
#         ----------
#         batch: keys.
# 
#         """
#         assert padding is None
#         data = get_datas(self.hdf5, batch)
#         return concat_converter(data, device=device, padding=padding)
# 