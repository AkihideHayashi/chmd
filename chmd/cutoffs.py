# *- coding:utf-8 -*-
"""Functions."""
import chainer.functions as F
from chainer import Variable, Link
import numpy as np


class CosineCutoff(Link):
    """Behler cos cutoff."""

    def __init__(self, cutoff=5.0):
        """Initialize."""
        super().__init__()
        self.add_persistent('cutoff', cutoff)

    def forward(self, distances: Variable):
        """Forward."""
        cutoff = self.cutoff
        cutoffs = 0.5 * (F.cos(distances * np.pi / cutoff) + 1.0)
        cutoffs *= distances.data < cutoff
        return cutoffs


class MollifierCutoff(Link):
    """Mollifier cutoff."""

    def __init__(self, cutoff=5.0, eps=1.0e-7):
        """Initialize."""
        super().__init__()
        self.add_persistent('cutoff', cutoff)
        self.add_persistent('eps', eps)

    def forward(self, distances: Variable):
        """Forward."""
        cutoff = self.cutoff
        eps = self.eps
        mask = (distances.data + eps < cutoff)
        exponent = 1.0 - 1.0 / (1.0 - (distances * mask / cutoff) ** 2)
        cutoffs = F.exp(exponent)
        cutoffs = cutoffs * mask
        return cutoffs
