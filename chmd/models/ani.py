"""New ANI-1."""
from abc import ABC, abstractproperty, abstractmethod
import numpy as np
import chainer
from chainer import Chain, functions as F, report, Variable, grad
from chainer.datasets import open_pickle_dataset, open_pickle_dataset_writer
from chainer.dataset.convert import to_device
from chainer.backend import get_array_module
from chmd.utils.batchform import parallel_form, series_form
from chmd.functions.neighbors import (neighbor_duos,
                                      number_repeats,
                                      compute_shifts,
                                      concat_neighbors_flatten_form
                                      )
from chmd.links.shifter import EnergyShifter
from chmd.links.linear import AtomWiseParamNN
from chmd.links.ani import ANI1AEV
from chmd.preprocess import symbols_to_elements, Preprocessor
from chmd.links.energy import EnergyNet
from chmd.links.ensemble import EnsemblePredictor
from chmd.dynamics.batch import AbstractBatch


class ANI1Preprocessor(Preprocessor):
    def __init__(self, params, forcecut, mode='aev'):
        """Preprocessor for ANI-1.

        Parameters
        ----------
        mode: 'aev' or 'neighbors'

        """
        self.aev_calc = ANI1AEV(params['num_elements'], **params['aev_params'])
        self.order = np.array(params['order'])
        self.pbc = np.array(params['pbc'])
        self.cutoff = params['cutoff']
        self.mode = mode
        self.forcecut = forcecut

    def process(self, datas, device):
        self.drain_big_force(datas, self.forcecut, device)
        if self.mode == 'aev':
            self.add_elements(datas, self.order)
            self.add_neighbors(datas, self.cutoff, self.pbc, device)
            self.add_aev(datas, self.aev_calc, device)
            for data in datas:
                del data['i2']
                del data['j2']
                del data['s2']
        elif self.mode == 'neighbors':
            self.add_elements(datas, self.order)
            self.add_neighbors(datas, self.cutoff, self.pbc, device)
        else:
            raise NotImplementedError(self.mode)

    def classify(self, data):
        return tuple(number_repeats(
            data['cell'], self.pbc, self.cutoff).astype(np.int64).tolist())

    @staticmethod
    def drain_max_force(datas, forcecut, device):
        forces = [data['forces'] for data in datas]
        [fo], _ = parallel_form.from_list([forces], [0.0])
        fo = to_device(device, fo)
        xp = get_array_module(fo)
        norm = xp.linalg.norm(fo, axis=2)
        max_norm = xp.max(norm)
        filt = max_norm < forcecut
        for fi, data in zip(filt, datas):
            if fi:
                data['status'] = 'train'
            else:
                data['status'] = 'drain'

    @staticmethod
    def add_elements(datas, order):
        symbols = [data['symbols'] for data in datas]
        [sym], valid = parallel_form.from_list([symbols], [''])
        elem = symbols_to_elements(sym, order)
        for i, data in enumerate(datas):
            data['elements'] = elem[i][valid[i]]

    @staticmethod
    def add_neighbors(datas, cutoff, pbc, device):
        direct = [data['positions'] for data in datas]
        cells = np.array([data['cell'] for data in datas])
        [direct], valid = parallel_form.from_list([direct], [0.0])
        cells = to_device(device, cells)
        direct = to_device(device, direct)
        pbc = to_device(device, pbc)
        valid = to_device(device, valid)
        xp = get_array_module(direct)
        repeat = xp.min(number_repeats(cells, pbc, cutoff), axis=0)
        shifts = compute_shifts(repeat)
        n2, i2, j2, s2 = neighbor_duos(cells, direct, cutoff, shifts, valid)
        for i, data in enumerate(datas):
            selecter = n2 == i
            data['i2'] = i2[selecter]
            data['j2'] = j2[selecter]
            data['s2'] = s2[selecter]

    @staticmethod
    def add_aev(datas, aev_calc, device):
        direct = [data['positions'] for data in datas]
        cells = np.array([data['cell'] for data in datas])
        elements = [data['elements'] for data in datas]
        [direct, elements], valid = parallel_form.from_list(
            [direct, elements], [0.0, -1])
        cells = to_device(device, cells)
        direct = to_device(device, direct)
        elements = to_device(device, elements)
        valid = to_device(device, valid)
        xp = get_array_module(direct)
        cartesian = xp.sum(direct[:, :, :, None] * cells[:, None, :, :], axis=-2)
        n_batch, n_atoms, n_dim = cartesian.shape
        ri = cartesian.reshape((n_batch * n_atoms, n_dim))
        ei = elements.reshape((n_batch * n_atoms, ))
        i2_seed = [data['i2'] for data in datas]
        j2_seed = [data['j2'] for data in datas]
        s2_seed = [data['s2'] for data in datas]
        i2, j2, s2 = concat_neighbors_flatten_form(
            n_batch, n_atoms, i2_seed, j2_seed, s2_seed)
        aev = aev_calc(ri, ei, i2, j2, s2)  # (batch * atoms, features)
        n_feature = aev.shape[-1]
        aev_parallel = aev.reshape((n_batch, n_atoms, n_feature))
        aev_cpu = to_device(-1, aev_parallel.data)
        valid = to_device(-1, valid)
        for i, data in enumerate(datas):
            data['aev'] = aev_cpu[i][valid[i]]


class ANI1AEV2EnergyWithShifter(Chain):
    def __init__(self, nn_params, shift_params):
        super().__init__()
        with self.init_scope():
            self.nn = AtomWiseParamNN(**nn_params)
            self.shifter = EnergyShifter(**shift_params)

    def forward(self, aevs, elements, valid):
        """Inputs are assumed to be parallel form."""
        xp = get_array_module(aevs)
        n_batch, n_atoms, n_features = aevs.shape
        assert elements.shape == (n_batch, n_atoms)
        assert valid.shape == (n_batch, n_atoms)
        flatten_aevs = F.reshape(aevs, (n_batch * n_atoms, n_features))
        flatten_elms = xp.reshape(elements, (n_batch * n_atoms,))
        flatten_aev_energies = F.squeeze(self.nn(flatten_aevs, flatten_elms), -1)
        flatten_elm_energies = self.shifter(flatten_elms)
        flatten_energies = flatten_aev_energies + flatten_elm_energies
        energies = F.reshape(flatten_energies, (n_batch, n_atoms))
        energies = F.where(valid, energies, xp.zeros_like(energies.data))
        return energies


class ANI1AEV2EnergyNet(Chain):
    def __init__(self, nn_params, energy_params):
        super().__init__()
        with self.init_scope():
            self.nn = AtomWiseParamNN(**nn_params)
            self.en = EnergyNet(**energy_params)
    
    def forward(self, aevs, elements, valid):
        """Input are assumed to be parallel form."""
        xp = get_array_module(aevs)
        n_batch, n_atoms, n_features = aevs.shape
        assert elements.shape == (n_batch, n_atoms)
        assert valid.shape == (n_batch, n_atoms)
        flatten_aevs = F.reshape(aevs, (n_batch * n_atoms, n_features))
        flatten_elms = xp.reshape(elements, (n_batch * n_atoms,))
        flatten_features = self.nn(flatten_aevs, flatten_elms)
        energies = self.en(flatten_features, flatten_elms)
        energies = F.reshape(energies, (n_batch, n_atoms))
        assert xp.all(energies.data[~valid] == 0.0), (energies, valid, elements)
        return energies


class ANI1(Chain):
    def __init__(self, energy_model, num_elements, aev_params, energy_params,
                 cutoff, pbc, n_ensemble, order):
        super().__init__()
        with self.init_scope():
            self.aev = ANI1AEV(num_elements, **aev_params)
            self.energies = EnsemblePredictor(n_ensemble,
                                              lambda: energy_model(
                                                  **energy_params))
        self.add_persistent('pbc', pbc)
        self.cutoff = cutoff
        self.order = order
    
    def forward(self, elements, positions, valid, i2, j2, s2):
        """
        elements: int[batch, atoms]
        positions: float[batch, atoms] cartesian
        valid: bool[batch, atoms]
        i2: int[pair]
        j2: int[pair]
        s2: float[pair, dim] cartesian
        """
        n_batch, n_atoms, n_dim = positions.shape
        ri = F.reshape(positions, (n_batch * n_atoms, n_dim))
        ei = elements.reshape((n_batch * n_atoms))
        aev = self.aev(ri, ei, i2, j2, s2)
        n_feature = aev.shape[1]
        aev = F.reshape(aev, (n_batch, n_atoms, n_feature))
        energies = self.energies(aev, elements, valid)
        return energies


class ANI1WeightedEnergyLoss(Chain):
    def __init__(self, model, temperature):
        super().__init__()
        with self.init_scope():
            self.model = model
        self.temperature = temperature
    
    def forward(self, aevs, elements, energies, valid):
        xp = self.xp
        n_ensemble = len(self.model)
        n_batch, n_atoms, n_features = aevs.shape
        predict_atomwise = self.model(aevs, elements, valid)
        for i in range(n_ensemble):
            assert xp.all(predict_atomwise.data[i][~valid] == 0.0)
        predict = F.sum(predict_atomwise, axis=2)
        assert predict.shape == (n_ensemble, n_batch), "{} != ({}, {})".format(predict.shape, n_ensemble, n_batch)
        diff = predict - energies[xp.newaxis, :]  # (ensemble, batch)
        power_diff = diff * diff
        number_atoms = xp.sum(valid, axis=1)

        base = xp.min(energies / number_atoms)
        weights = xp.exp((base - energies / number_atoms) / self.temperature)  # (batch,)
        loss = F.sum(power_diff * weights / number_atoms / number_atoms) / n_batch / xp.sum(weights)
        report({'loss': loss.data}, self)
        return loss


class ANI1EnergyLoss(Chain):
    """ANI1 (2017)."""
    def __init__(self, model, tau=0.5):
        super().__init__()
        with self.init_scope():
            self.model = model
        self.tau = tau
    
    def forward(self, aevs, elements, energies, valid):
        xp = self.xp
        n_ensemble = len(self.model)
        n_batch, n_atoms, n_features = aevs.shape
        predict_atomwise = self.model(aevs, elements, valid)
        for i in range(n_ensemble):
            assert xp.all(predict_atomwise.data[i][~valid] == 0.0)
        predict = F.sum(predict_atomwise, axis=2)
        assert predict.shape == (n_ensemble, n_batch), "{} != ({}, {})".format(predict.shape, n_ensemble, n_batch)
        diff = predict - energies[xp.newaxis, :]  # (ensemble, batch)
        cost = self.tau * F.exp(F.sum(diff ** 2, axis=1) / self.tau)
        loss = F.sum(cost)
        report({'loss': loss.data}, self)
        return loss

class ANI1SimpleEnergyLoss(Chain):
    """ANI1 Energy loss from aevs."""

    def __init__(self, model):
        """Initialize."""
        super().__init__()
        with self.init_scope():
            self.model = model

    def forward(self, aevs, elements, energies, valid):
        """Calculate energy loss from aev. Input is assumed to be parallel form.

        Parameters
        ----------
        aevs: float(n_batch, n_atoms, n_features)
        elements: int(n_batch, n_atoms)
        energies: float(n_batch,)
        valids: bool(n_batch, n_atoms)

        """
        xp = self.xp
        n_ensemble = len(self.model)
        n_batch, n_atoms, n_features = aevs.shape
        predict_atomwise = self.model(aevs, elements, valid)
        for i in range(n_ensemble):
            assert xp.all(predict_atomwise.data[i][~valid] == 0.0)
        predict = F.sum(predict_atomwise, axis=2)
        assert predict.shape == (n_ensemble, n_batch), "{} != ({}, {})".format(predict.shape, n_ensemble, n_batch)
        diff = predict - energies[xp.newaxis, :]  # (ensemble, batch)
        power_diff = diff * diff
        mean_power_diff = F.mean(power_diff, 1)  # (ensemble,)
        assert mean_power_diff.shape == (n_ensemble,)
        loss = F.sum(mean_power_diff)
        report({'loss': loss.data}, self)
        return loss


@chainer.dataset.converter()
def concat_aev(batch, device):
    """Not tested yet."""
    lst_keys = ['aev', 'elements']
    paddings = [0.0, -1]
    lst = [[atoms[key] for atoms in batch] for key in lst_keys]
    parallels, valid = parallel_form.from_list(lst, paddings)
    aevs = parallels[0]
    elements = parallels[1]
    energies = np.array([atoms['energy'] for atoms in batch])
    dtype = chainer.config.dtype
    return_dict = {
        'aevs': aevs.astype(dtype),
        'elements': elements,
        'valid': valid,
        'energies': energies.astype(dtype),
    }
    for key in return_dict:
        return_dict[key] = to_device(device, return_dict[key])
    return return_dict

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
    def is_atom(self):
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

    @abstractproperty
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
    return - xp.sum(G[:, None, :, :] * grads[:, :, None, :], axis=-1)


class ANI1ForceField(object):
    def __init__(self, model: ANI1, neighbor_list, name='ANI1'):
        self.model: ANI1 = model
        self.neighbor_list = neighbor_list
        self.name = name

    def __call__(self, batch: ANI1Batch):
        xp = batch.xp
        n_batch, n_atoms, n_dim = batch.positions.shape
        n_emsemble = len(self.model.energies)
        to_ub = float(np.sqrt(n_emsemble / (n_emsemble - 1)))
        i2, j2, s2 = self.neighbor_list(batch)
        direct = Variable(batch.positions)
        elements = batch.elements
        cells = Variable(batch.cells)
        valid = batch.is_atom  # (batch, atoms)
        assert valid.shape == (n_batch, n_atoms)
        number_of_atoms = xp.sum(valid, 1)
        assert number_of_atoms.shape == (n_batch, )
        cart = F.sum(direct[:, :, :, None] * cells[:, None, :, :], axis=-2)
        atomic_energies = self.model(elements, cart, valid, i2, j2, s2)
        assert atomic_energies.shape == (n_emsemble, n_batch, n_atoms)
        molecular_energies = F.sum(atomic_energies, 2)  # (ensemble, batch)
        assert molecular_energies.shape == (n_emsemble, n_batch)
        mean_molecular_energies = F.mean(molecular_energies, 0)  # (batch)
        var_molecular_energies = (
            F.mean(molecular_energies ** 2, 0)
            - mean_molecular_energies ** 2)
        std_molecular_energies = F.sqrt(var_molecular_energies) * to_ub
        assert mean_molecular_energies.shape == (n_batch, )
        assert var_molecular_energies.shape == (n_batch, )

        atomic_std = atomic_energies.data.std(0) * to_ub

        batch.potential_energies = mean_molecular_energies.data
        batch.error = std_molecular_energies.data / xp.sqrt(number_of_atoms)
        batch.atomic_error = atomic_std

        grads, = grad([mean_molecular_energies], [direct])
        batch.forces = grad_to_force(grads.data, cells.data)
