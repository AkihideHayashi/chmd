"""Loss."""
from chainer import Variable, Chain, ChainList, grad
import chainer.functions as F
from chainer import reporter


class SummaryLoss(Chain):
    """Summation of many loss."""

    def __init__(self, models, loss_fun, *args, **kwargs):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            self.losses = ChainList(*[loss_fun(model, *args, **kwargs)
                                      for model in models])

    def forward(self, *args, **kwargs):
        """Forward."""
        return sum(l(*args, **kwargs) for l in self.losses)


class EnergyGradLoss(Chain):
    """Energy + Grad."""

    def __init__(self, target, ce, cf):
        """Initializer.

        Paramters
        ---------
        ce: coeffient for energy.
        cf: coeffient for forces.

        """
        super().__init__()
        with self.init_scope():
            self.target = target
        self.ce = ce
        self.cf = cf

    def __call__(self, ri, e, f, *args, **kwargs):
        """Loss.

        Parameters
        ----------
        target: Chain.
        ri: positions.
        e: Energy (ground truth.)
        f: Force (ground truth.)

        """
        ri = Variable(ri)
        en = self.target(ri, *args, **kwargs)
        fi, = grad([-en], [ri], enable_double_backprop=True)
        loss_e = F.mean_squared_error(en, e)
        reporter.report({'loss_e': loss_e.data}, self)
        loss_f = F.mean_squared_error(fi, f)
        reporter.report({'loss_f': loss_f.data}, self)
        loss = self.ce * loss_e + self.cf * loss_f
        reporter.report({'loss': loss.data}, self)
        return loss


class EachEnergyGradLoss(Chain):
    """Energy + Grad."""

    def __init__(self, target, ce, cf):
        """Initializer.

        Paramters
        ---------
        ce: coeffient for energy.
        cf: coeffient for forces.

        """
        super().__init__()
        with self.init_scope():
            self.target = target
        self.ce = ce
        self.cf = cf

    def __call__(self, ri, e, f, i1, *args, **kwargs):
        """Loss.

        Parameters
        ----------
        target: Chain.
        ri: positions.
        e: Energy (ground truth.)
        f: Force (ground truth.)

        """
        ri = Variable(ri)
        en = self.target(ri, *args, **kwargs)
        fi, = grad([-en], [ri])
        _, n_atoms = self.xp.unique(i1, return_count=True)
        loss_e = (en.data - e) ** 2 / n_atoms
        loss_f_atom = F.sum((fi - f) ** 2, axis=1)
        loss_f = F.scatter_add(
            self.xp.zeros_like(loss_e), i1, loss_f_atom) / n_atoms
        loss = self.ce * loss_e + self.cf * loss_f
        return loss
