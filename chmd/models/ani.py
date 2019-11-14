"""ANI-1."""
from abc import ABC, abstractproperty, abstractmethod
import numpy as np
import chainer
from chainer import Chain, Variable, ChainList, grad, report
import chainer.functions as F
from chmd.links.ani import ANI1AEV, ANI1AEV2EnergyFlattenForm
from chmd.links.linear import AtomWiseParamNN
from chmd.links.shifter import EnergyShifter
from chmd.functions.neighbors import neighbor_duos_to_flatten_form
from chmd.utils.batchform import flatten_form
from chmd.math.lattice import direct_to_cartesian_chainer
from chmd.dynamics.batch import AbstractBatch


def asarray(x):
    if isinstance(x, Variable):
        return x.data
    return x

# 中の処理をseries formで処理するには隣接をseriese formで渡す必要があるが、（本当？）
# parallel formでpositions, cellsを渡して置いて、隣接だけseriese formというのはややこしすぎる。
# 従って、ani1の集計処理をflatten formにすることで隣接も座標もparallel formで渡すことにする。
# TODO Energy shifterをparallel formかflatten formに対応させる。


class ANI1Batch(AbstractBatch):

    @abstractproperty
    def positions(self):
        ...

    @abstractproperty
    def elements(self):
        ...

    @abstractproperty
    def cells(self):
        ...

    @abstractproperty
    def valid(self):
        ...

    @abstractproperty
    def potential_energies(self):
        ...

    @potential_energies.setter
    def potential_energies(self, _):
        ...

    @abstractproperty
    def forces(self):
        ...

    @forces.setter
    def forces(self, _):
        ...

    @abstractproperty
    def variance_potential_energies(self):
        ...

    @variance_potential_energies.setter
    def variance_potential_energies(self, _):
        ...

    @abstractmethod
    def xp(self):
        ...


class ANI1ForceField(object):
    def __init__(self, params, path, neighbor_list, name='ANI1'):
        self.model = ANI1(**params)
        chainer.serializers.load_npz(path, self.model)
        self.neighbor_list = neighbor_list
        self.name = name
    
    def __call__(self, batch: ANI1Batch):
        i2, j2, s2 = self.neighbor_list(batch)
        positions = Variable(batch.positions)
        elements = batch.elements
        cells = Variable(batch.cells)
        valid = batch.valid
        energies = self.model(cells, elements, positions, valid, i2, j2, s2)
        mean = F.sum(energies, axis=0)
        mean2 = F.sum(energies * energies, axis=0)
        var = mean2 - mean
        forces, = grad([-mean], [positions])
        batch.potential_energies = mean.data
        batch.forces = forces.data
        batch.variance_potential_energies = var.data


class ANI1(Chain):
    """ANI-1 energy calculator."""

    def __init__(self, num_elements, aev_params, nn_params,
                 cutoff, pbc, n_agents, order):
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
        self.order = order

    def forward(self, cells, elements, positions, valid, i2, j2, s2):
        """Apply. All inputs are assumed to be passed as parallel form.

        However, i2, j2, s2 are assumed to be flatten from.

        Parameters
        ----------
        cells: (n_batch x n_dim x n_dim)
        elements: (n_batch x n_atoms)
        positions: (n_batch x n_atoms x n_dim) direct coordinate.
        valid: bool(n_batch x n_atoms) True if is_atom. False if dummy.
        i2, j2: (n_bond) flatten form based.
        s2: (n_bond, n_free) flatten form based.

        Returns
        -------
        energies: (n_parallel, n_batch)

        """
        xp = self.xp
        in_cell_positions = positions - positions // 1
        cartesian_positions = direct_to_cartesian_chainer(
            cells, in_cell_positions)
        n_batch, n_atoms = elements.shape
        # Fist, make all to flatten form.
        v1, i1 = flatten_form.valid_affiliation_from_parallel(valid)
        (ei, ri), v1, i1 = flatten_form.from_parallel(
            [elements, cartesian_positions], valid)
        # Second calculate AEV.
        aev = self.aev(cells, ri, ei, i1, i2, j2, s2)
        # Third calculate energies
        shift = self.shift(ei)
        # List[(n_batch * n_atoms)]

        def calculate_atomic_valid(en):
            atomic_novalid = F.squeeze(en(aev, ei), axis=1) + shift
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

    def __call__(self, cells, positions, energies, forces, *args, **kwargs):
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
        en = self.predictor(cells=cells, positions=ri, *args, **kwargs)
        assert en.ndim == 2
        loss_e = 0.0
        loss_f = 0.0
        for i in range(self.predictor.n_agents):
            fi_direct, = grad([-en[i, :]], [ri], enable_double_backprop=True)
            fi_cartesian = direct_to_cartesian_chainer(cells, fi_direct)
            forces_cartesian = direct_to_cartesian_chainer(cells, forces)
            loss_e += F.mean_squared_error(en[i, :], energies)
            loss_f += F.mean_squared_error(fi_cartesian, forces_cartesian)
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


class ANI1EachEnergyGradLoss(Chain):
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

    def __call__(self, cells, positions, energies, forces, *args, **kwargs):
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
        en = self.predictor(cells=cells, positions=ri, *args, **kwargs)
        assert en.ndim == 2
        mean = F.mean(en, axis=0)
        force, = grad([-mean], [ri])
        loss_e = (mean - energies) ** 2
        loss_f = F.mean(F.mean((force - forces) ** 2, axis=-1), axis=-1)
        return self.ce * loss_e.data + self.cf * loss_f.data
