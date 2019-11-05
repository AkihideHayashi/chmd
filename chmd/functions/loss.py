"""Define Loss functions."""
from chainer import Variable
import chainer.functions as F
from chainer import reporter


def energy_accuracy(ei, ci, ri, i1, i2, j2, s2, e, f, target):
    """Atoms wise energy loss."""
    assert f.shape == ri.shape
    target.cleargrads()
    en = target(ei=ei, ci=ci, ri=ri, i1=i1, i2=i2, j2=j2, s2=s2)
    loss = (en.data - e) ** 2
    return loss
