r"""Support transform between some styles of having batch.

listform: List[np.ndarray(n_atoms, [n_dims])]
parallelform: np.ndarray(n_batch, n_atoms, [n_dims]), valid
seriesform: np.ndarray(\sum_{i}{n_atoms_i}, [n_dims]), affiliation
"""

import numpy as np
from chainer.backend import get_array_module
from chmd.math.xp import repeat_interleave, scatter_add


class parallel_form(object):
    """Compute valid from several form."""

    @staticmethod
    def valid_from_listform(listform):
        """Compute valid from listform."""
        n_atoms = np.array([len(x) for x in listform])
        arange_batch = np.arange(np.max(n_atoms))
        return arange_batch[np.newaxis, :] < n_atoms[:, np.newaxis]

    @staticmethod
    def valid_from_affiliation(affiliation):
        xp = get_array_module(affiliation)
        aff, n_atoms = xp.unique(affiliation, return_counts=True)
        assert xp.all(aff == xp.arange(len(aff)))
        arange_batch = xp.arange(xp.max(n_atoms))
        return arange_batch[xp.newaxis, :] < n_atoms[:, xp.newaxis]

    @staticmethod
    def parallel_from_list(listform, valid, padding):
        xp = get_array_module(valid)
        seriesform = series_form.series_from_listform(listform)
        tmp = xp.full((*valid.shape, *listform[0].shape[1:]),
                      padding, dtype=seriesform.dtype)
        tmp[valid] = seriesform
        return tmp

    @staticmethod
    def parallel_from_series(seriesform, affiliation, valid, padding):
        xp = get_array_module(affiliation)
        aff, n_atoms = xp.unique(affiliation, return_counts=True)
        tmp = xp.full((
            len(aff), max(n_atoms), *seriesform.shape[1:]),
            padding, dtype=seriesform.dtype)
        tmp[valid] = seriesform
        return tmp

    @staticmethod
    def from_list(lists, padding):
        valid = __class__.valid_from_listform(lists[0])
        for lst in lists:
            assert np.allclose(valid, __class__.valid_from_listform(lst))
        parallels = [__class__.parallel_from_list(
            lst, valid, padding) for lst in lists]
        return parallels, valid

    @staticmethod
    def from_series(series, affiliations, padding):
        valid = __class__.valid_from_affiliation(affiliations)
        parallels = [__class__.parallel_from_series(
            s, affiliations, valid, padding) for s in series]
        return parallels, valid


class series_form(object):
    """Compute affiliation from several form."""

    @staticmethod
    def affiliations_from_valid(valid):
        xp = get_array_module(valid)
        n_atoms = xp.sum(valid, axis=1)
        return repeat_interleave(n_atoms)

    @staticmethod
    def affiliations_from_listform(listform):
        xp = get_array_module(listform[0])
        n_atoms = xp.array([len(x) for x in listform])
        return repeat_interleave(n_atoms)

    @staticmethod
    def series_from_listform(listform):
        xp = get_array_module(listform[0])
        return xp.concatenate(listform, axis=0)

    @staticmethod
    def series_from_parallel(parallelform, valid):
        return parallelform[valid]

    @staticmethod
    def from_list(lists):
        affiliations = __class__.affiliations_from_listform(lists[0])
        for lst in lists:
            assert np.allclose(affiliations,
                               __class__.affiliations_from_listform(lst))
        series = [__class__.series_from_listform(lst) for lst in lists]
        return series, affiliations

    @staticmethod
    def from_parallel(parallels, valid):
        affiliations = __class__.affiliations_from_valid(valid)
        series = [__class__.series_from_parallel(para, valid)
                  for para in parallels]
        return series, affiliations


class list_form(object):
    """Cumpute list form."""
    @staticmethod
    def from_series(series, affiliation):
        xp = get_array_module(series)
        n = len(xp.unique(affiliation))
        return [[s[affiliation == i] for i in range(n)] for s in series]
    
    @staticmethod
    def from_parallel(parallel, valid):
        xp = get_array_module(parallel_form)
        return [[p[v] for p, v in zip(pa, valid)] for pa in parallel]
