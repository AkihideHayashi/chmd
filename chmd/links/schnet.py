# *- coding:utf-8 -*-
"""Link."""
# %%
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Link, ChainList
from chmd.neighbors import neighbor_trios
from chmd.atoms import Atoms, get_items
from chmd.activations import shifted_softplus

# %%


def cfconv(xi, cij, wij, i2, j2, eps=1.0, xp=np):
    """CFConv.

    Parameters
    ----------
    xi  : feature vector (n_atoms, n_feature)
    cij : cutoff (n_duo,)
    wij : filter (n_duo, n_feature)
    i2  : (n_duo,)
    j2  : (n_duo,)

    """
    num = F.scatter_add(
        xp.zeros(xi.shape),
        i2,
        xi[j2] * wij * cij)
    den = F.scatter_add(
        xp.zeros_like(xi.shape[0]),
        i2,
        cij
    )
    return num / (den + eps)


def rbf(rij):
    """RBF function."""
    pass
