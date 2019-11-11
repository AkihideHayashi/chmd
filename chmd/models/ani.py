"""ANI-1."""
import numpy as np
import chainer
from chainer import Chain, Variable, ChainList, grad, report
import chainer.functions as F
from chmd.links.ani import ANI1AEV, ANI1AEV2Energy
from chmd.links.shifter import EnergyShifter
from chmd.functions.neighbors import neighbor_duos, number_repeats
from chmd.utils.devicestruct import DeviceStruct


def asarray(x):
    if isinstance(x, Variable):
        return x.data
    return x


class ANI1(Chain):
    """ANI-1 energy calculator."""

    def __init__(self, num_elements, aev_params, nn_params,
                 cutoff, pbc, n_agents):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            self.aev = ANI1AEV(num_elements, **aev_params)
            self.energies = ChainList(*[ANI1AEV2Energy(num_elements, nn_params)
                                        for _ in range(n_agents)])
            self.shift = EnergyShifter(num_elements)
        self.add_persistent('pbc', pbc)
        self.cutoff = cutoff
        self.n_agents = n_agents

    def forward(self, cells, elements, positions,
                i1, i2=None, j2=None, s2=None):
        """Apply."""
        if i2 is None or j2 is None or s2 is None:
            repeats = number_repeats(cells, self.pbc, self.cutoff)
            repeat = self.xp.max(repeats, axis=0)
            (i2, j2, s2) = neighbor_duos(
                asarray(cells), asarray(positions),
                self.cutoff, repeat, i1,)
        aev = self.aev(cells, positions, elements, i1, i2, j2, s2)
        # parallel x batch
        energy_nn = F.concat([en(aev, elements, i1, cells.shape[0])[None, :]
                              for en in self.energies], axis=0)
        assert energy_nn.shape[0] == self.n_agents
        assert energy_nn.shape[1] == cells.shape[0]
        assert energy_nn.ndim == 2
        energy_linear = self.shift(elements, i1)
        return energy_nn + energy_linear[None, :]


class ANI1EnergyGradLoss(Chain):
    """Energy + Grad."""

    def __init__(self, predictor, ce, cf):
        """Initializer.

        Paramters
        ---------
        ce: coeffient for energy.
        cf: coeffient for forces.

        """
        super().__init__()
        with self.init_scope():
            self.predictor = predictor
        self.ce = ce
        self.cf = cf

    def __call__(self, positions, energies, forces, *args, **kwargs):
        """Loss.

        Parameters
        ----------
        target: Chain.
        ri: positions.
        e: Energy (ground truth.)
        f: Force (ground truth.)

        """
        ri = Variable(positions)
        # n_agents x n_batch
        en = self.predictor(positions=ri, *args, **kwargs)
        assert en.ndim == 2
        loss_e = 0.0
        loss_f = 0.0
        for i in range(self.predictor.n_agents):
            fi, = grad([-en[i, :]], [ri], enable_double_backprop=True)
            assert ri.shape == fi.shape
            loss_e += F.mean_squared_error(en[i, :], energies)
            loss_f += F.mean_squared_error(fi, forces)
        loss_e /= self.predictor.n_agents
        loss_f /= self.predictor.n_agents
        report({'loss_e': loss_e.data}, self)
        report({'loss_f': loss_f.data}, self)
        loss = self.ce * loss_e + self.cf * loss_f
        report({'loss': loss.data}, self)
        return loss


class EnergyForceVar(Chain):
    """Energy + Grad."""

    def __init__(self, predictor):
        """Initializer.

        Paramters
        ---------
        ce: coeffient for energy.
        cf: coeffient for forces.

        """
        super().__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, positions, *args, **kwargs):
        """Loss.

        Parameters
        ----------
        target: Chain.
        ri: positions.
        e: Energy (ground truth.)
        f: Force (ground truth.)

        """
        ri = Variable(positions)
        # n_agents x n_batch
        en = self.predictor(positions=ri, *args, **kwargs)
        mean = F.mean(en, axis=0)
        n2 = F.mean(en * en, axis=0)
        var = n2 - mean * mean
        force, = grad([-mean], [ri])
        return mean.data, force.data, var.data
