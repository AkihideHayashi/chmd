# *- coding:utf-8 -*-
"""Functions."""
import chainer.functions as F
from chainer import Variable
import numpy as np


def shifted_softplus(x: Variable):
    """Compute shifted soft-plus activation function."""
    return F.softplus(x) - np.log(2.0)


def gaussian(x: Variable):
    """Gaussian activation function."""
    return F.exp(- x * x)
