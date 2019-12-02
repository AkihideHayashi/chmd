# *- coding:utf-8 -*-
"""Functions."""
import chainer.functions as F
from chainer import Variable
import numpy as np


def parse_act(act):
    """Parse string act."""
    if not isinstance(act, str):
        return act
    elif act == 'gaussian':
        return gaussian
    elif act == 'shifted_softplus':
        return shifted_softplus
    else:
        raise NotImplementedError(act)


def shifted_softplus(x: Variable):
    """Compute shifted soft-plus activation function."""
    return F.softplus(x) - np.log(2.0)


def gaussian(x: Variable):
    """Gaussian activation function."""
    return F.exp(- x * x)
