# *- coding:utf-8 -*-
"""Dump rij to peak."""
import chainer.functions as F
from chainer import Link


class GaussianSmearing(Link):
    """Gaussian Smearing."""

    def __init__(self, start=0.0, stop=5.0, n=50):
        """Initialize."""
        super().__init__()
        xp = self.xp
        offset = xp.linspace(start, stop, n)
        widths = (offset[1] - offset[0]) * xp.ones_like(offset)
        self.add_persistent('offset', offset)
        self.add_persistent('widths', widths)

    def forward(self, rij):
        """Forward."""
        xp = self.xp
        offset = self.offset
        widths = self.widths
        coeff = -0.5 / (widths ** 2)
        diff = rij[:, :, xp.newaxis] - offset[xp.newaxis, xp.newaxis, :]
        gauss = F.exp(coeff * (diff ** 2))
        return gauss
