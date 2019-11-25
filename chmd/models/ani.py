"""ANI-1."""
import math
from abc import ABC, abstractproperty, abstractmethod
import numpy as np
import chainer
from chainer.backend import get_array_module
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
    def error(self):
        ...

    @error.setter
    def error(self, _):
        ...

    @abstractproperty
    def atomic_error(self):
        ...

    @atomic_error.setter
    def atomic_error(self, _):
        ...

    @abstractmethod
    def xp(self):
        ...


def grad_to_force(grads, cells):
    """Calculate forces from gradients in direct positoins.

    Parameters
    ----------
    grads: (batch, atoms, dim)
    cells: (batch, dim, dim)

    """
    xp = get_array_module(grads)
    L = xp.linalg.inv(cells)  # (batch x dim x dim)
    LT = L.transpose((0, 2, 1))  # (batch x dim x dim)
    G = xp.sum(LT[:, :, :, None] * L[:, None, :, :], axis=-2)
    return xp.sum(G[:, None, :, :] * grads[:, :, None, :], axis=-1)


class ANI1ForceField(object):
    def __init__(self, params, path, neighbor_list, name='ANI1'):
        self.model = ANI1(**params)
        chainer.serializers.load_npz(path, self.model)
        self.neighbor_list = neighbor_list
        self.name = name

    def __call__(self, batch: ANI1Batch):
        xp = batch.xp
        n_batch, n_atoms, n_dim = batch.positions.shape
        n_emsemble = self.model.n_agents
        to_ub = math.sqrt(n_emsemble / (n_emsemble - 1))
        i2, j2, s2 = self.neighbor_list(batch)
        positions = Variable(batch.positions)
        elements = batch.elements
        cells = Variable(batch.cells)
        valid = batch.valid  # (batch, atoms)
        assert valid.shape == (n_batch, n_atoms)
        number_of_atoms = xp.sum(valid, 1)
        assert number_of_atoms.shape == (n_batch, )
        atomic_energies = self.model(
            cells, elements, positions, valid, i2, j2, s2)
        assert atomic_energies.shape == (n_emsemble, n_batch, n_atoms)
        molecular_energies = F.sum(atomic_energies, 2)  # (ensemble, batch)
        assert molecular_energies.shape == (n_emsemble, n_batch)
        mean_molecular_energies = F.mean(molecular_energies, 0)  # (batch)
        var_molecular_energies = (
            F.mean(molecular_energies ** 2, 0)
            - mean_molecular_energies)
        std_molecular_energies = F.sqrt(var_molecular_energies) * to_ub
        assert mean_molecular_energies.shape == (n_batch, )
        assert var_molecular_energies.shape == (n_batch, )

        atomic_std = atomic_energies.data.std(0) * to_ub

        batch.potential_energies = mean_molecular_energies.data
        batch.error = std_molecular_energies.data / xp.sqrt(number_of_atoms)
        batch.atomic_error = atomic_std

        grads, = grad([-mean_molecular_energies], [positions])
        batch.forces = grad_to_force(grads.data, cells.data)



class ANI1(Chain):
    """ANI-1 energy calculator."""

    def __init__(self, num_elements, aev_params, nn_params,
                 cutoff, pbc, n_agents, order):
        """Initializer."""
        super().__init__()
        with self.init_scope():
            # recieve flatten form and return flatten form.
            self.aev = ANI1AEV(num_elements, **aev_params)
            self.energies = ChainList(*[AtomWiseParamNN(**nn_params)
                                        for _ in range(n_agents)])  # reciece flatten form.
            self.shift = EnergyShifter(num_elements)
        self.add_persistent('pbc', pbc)
        self.cutoff = cutoff
        self.n_agents = n_agents
        self.order = order

    def forward(self, cells, elements, positions, valid, i2, j2, s2,
                cartesian_positions=None):
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
        if cartesian_positions is None:
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
            energies = en(aev, ei)  # (n_batch * n_atoms, n_quantities)
            atomic_novalid = F.squeeze(energies, axis=1) + shift
            # (n_batch * n_atoms)

            atomic_valid = F.where(v1,
                                   atomic_novalid,
                                   xp.zeros_like(atomic_novalid.data))
            return F.reshape(atomic_valid, (1, n_batch, n_atoms))

        # (n_parallel, n_batch, n_atoms)
        atomic_energies = F.concat([calculate_atomic_valid(e)
                                    for e in self.energies], axis=0)
        return atomic_energies


def atomic_energies_to_molecular_energies(atomic_energies):
    """Sum up atomic energies.

    Parameters
    ----------
    atomic_energies: (n_agents, n_batch, n_atoms)

    """
    return F.sum(atomic_energies, axis=2)


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

    def forward(self, cells, elements, positions, valid,
                i2, j2, s2, energies, forces):
        """Please be aware that positions is direct coordinate.
        However, forces is eV / Angstrome.
        It is vasprun.xml 's setting. And it is convenient for calculate loss.

        Parameters
        ----------
        cells: float[n_batch, n_dim, n_dim] Angsgrome
        elements: int[n_batch, n_atoms]
        positions: float[n_batch, n_atoms, n_dim] direct coordinate.
        valid: bool[n_batch, n_atoms]
        i2: int[n_pairs]
        j2: int[n_pairs]
        s2: int[n_pairs, n_dim]
        energies: float[n_batch] eV
        forces: float[n_batch, n_atoms, n_dim] eV / Angstrome

        """
        ri_direct = Variable(positions - positions // 1)
        ri_cartesian = direct_to_cartesian_chainer(cells, ri_direct)
        # n_agents x n_batch
        en_predict = atomic_energies_to_molecular_energies(
            self.predictor(cells, elements, positions,
                           valid, i2, j2, s2,
                           cartesian_positions=ri_cartesian)
        )
        assert en_predict.ndim == 2
        loss_e = 0.0
        loss_f = 0.0
        for i in range(self.predictor.n_agents):
            fi, = grad([-en_predict[i, :]], [ri_cartesian],
                       enable_double_backprop=True)
            loss_e += F.mean_squared_error(en_predict[i, :], energies)
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

    def forward(self, cells, elements, positions, valid,
                i2, j2, s2, energies, forces):
        """Loss.

        Parameters
        ----------
        target: Chain.
        ri: positions.
        e: Energy (ground truth.)
        f: Force (ground truth.)

        """
        ri_direct = Variable(positions - positions // 1)
        ri_cartesian = direct_to_cartesian_chainer(cells, ri_direct)
        # n_agents x n_batch
        en_predict = atomic_energies_to_molecular_energies(
            self.predictor(cells, elements, positions,
                           valid, i2, j2, s2,
                           cartesian_positions=ri_cartesian))
        assert en_predict.ndim == 2
        mean = F.mean(en_predict, axis=0)
        force, = grad([-mean], [ri_cartesian])
        loss_e = (mean - energies) ** 2
        loss_f = F.mean(F.mean((force - forces) ** 2, axis=-1), axis=-1)
        return self.ce * loss_e.data + self.cf * loss_f.data
