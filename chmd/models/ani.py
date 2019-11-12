"""ANI-1."""
import numpy as np
import chainer
from chainer import Chain, Variable, ChainList, grad, report
import chainer.functions as F
from chmd.links.ani import ANI1AEV, ANI1AEV2EnergyFlattenForm
from chmd.links.linear import AtomWiseParamNN
from chmd.links.shifter import EnergyShifter
from chmd.functions.neighbors import neighbor_duos_to_flatten_form
from chmd.utils.batchform import flatten_form


def asarray(x):
    if isinstance(x, Variable):
        return x.data
    return x

# 中の処理をseries formで処理するには隣接をseriese formで渡す必要があるが、（本当？）
# parallel formでpositions, cellsを渡して置いて、隣接だけseriese formというのはややこしすぎる。
# 従って、ani1の集計処理をflatten formにすることで隣接も座標もparallel formで渡すことにする。
# TODO Energy shifterをparallel formかflatten formに対応させる。


class ANI1(Chain):
    """ANI-1 energy calculator."""

    def __init__(self, num_elements, aev_params, nn_params,
                 cutoff, pbc, n_agents):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            self.aev = ANI1AEV(num_elements, **aev_params)  # recieve flatten form and return flatten form.
            self.energies = ChainList(*[AtomWiseParamNN(**nn_params)  
                                        for _ in range(n_agents)])  # reciece flatten form.
            self.shift = EnergyShifter(num_elements)
        self.add_persistent('pbc', pbc)
        self.cutoff = cutoff
        self.n_agents = n_agents

    def forward(self, cells, elements, positions, valid, i2=None, j2=None, s2=None):
        """Apply. All inputs are assumed to be passed as parallel form.

        However, i2, j2, s2 are assumed to be flatten from.
        """
        xp = self.xp
        if False:
            cartesian_positions = F.sum(positions[:, :, :, None] * cells[:, None, :, :], axis=-2)
        n_batch, n_atoms = elements.shape
        # Fist, make all to flatten form.
        v1, i1 = flatten_form.valid_affiliation_from_parallel(valid)
        (ei, ri), v1, i1 = flatten_form.from_parallel([elements, positions], valid)
        if i2 is None or j2 is None or s2 is None:
            i2, j2, s2 = neighbor_duos_to_flatten_form(asarray(cells), asarray(positions),
                                                       self.cutoff,
                                                       self.pbc, valid)
            assert False
        # Second calculate AEV.
        aev = self.aev(cells, ri, ei, i1, i2, j2, s2)
        # Third calculate energies
        shift = self.shift(ei)
        # List[(n_batch * n_atoms)]

        def calculate_atomic_valid(en):
            atomic_novalid = en(aev, ei) + shift
            atomic_valid = F.where(v1,
                                   atomic_novalid,
                                   xp.zeros_like(atomic_novalid.data))
            return F.expand_dims(atomic_valid, 0)

        # (n_parallel x n_batch * n_atoms)
        atomic_energies = F.concat([calculate_atomic_valid(e)
                                    for e in self.energies], axis=0)
        molecular_energies = F.sum(
            F.reshape(atomic_energies, (self.n_agents, n_batch, n_atoms)),
            axis=2
            )
        return molecular_energies


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
