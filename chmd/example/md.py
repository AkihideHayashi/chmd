import json
import numpy as np
from ase.units import kB
import chainer
from chainer import report, get_current_reporter
from chainer.dataset.convert import to_device
from chmd.dynamics.dynamics import VelocityScaling, MolecularDynamicsBatch
from chmd.models.ani import ANI1ForceField
from chmd.dynamics.ani import BasicNeighborList, ANI1MolecularDynamicsBatch
from chmd.dynamics.dynamics import Extension
from chmd.functions.activations import gaussian
from chmd.math.lattice import direct_to_cartesian
from chmd.loop.randomgeneration import random_coordinates, random_symbols


class MDReporter(Extension):
    def __init__(self, order):
        self.order = np.array(order)

    def setup(self, dynamics):
        dynamics.reporter.add_observer('quantities', self)

    def __call__(self, batch: MolecularDynamicsBatch):
        to_report = {
            'potential_energies': batch.potential_energies,
            'symbols': self.order[to_device(-1, batch.elements)],
            'positions': batch.positions,
            'cells': batch.cells,
            'valid': batch.valid,
            'velocities': batch.velocities,
            'times': batch.times,
        }
        report(to_report, self)


class ANI1Reporter(MDReporter):
    def __call__(self, batch: MolecularDynamicsBatch):
        super().__call__(batch)
        to_report = {
            'error': batch.error,
            'atomic_error': batch.atomic_error
        }
        report(to_report, self)


class XYZDumper(Extension):
    def __init__(self, prefix):
        self.prefix = prefix

    def setup(self, dynamics):
        pass

    def __call__(self, batch):
        observation = get_current_reporter().observation
        cells = observation['quantities/cells']
        cartesian_positions = observation['quantities/positions']
        positions = direct_to_cartesian(cells, cartesian_positions)
        symbols = observation['quantities/symbols']
        error = observation['quantities/error']
        valid = observation['quantities/valid']
        for i, (sym, pos, err, val) in enumerate(zip(symbols, positions, error, valid)):
            with open(f'{self.prefix}_{i}.xyz', 'a') as f:
                natoms = np.sum(val)
                f.write(f'{natoms}\n')
                f.write(f'{err}\n')
                for j, (s, p) in enumerate(zip(sym, pos)):
                    if valid[i, j]:
                        f.write(f'{s} {p[0]} {p[1]} {p[2]}\n')


class PrintReport(Extension):
    def __init__(self, keys):
        assert isinstance(keys, list)
        self.keys = keys
        self.step = 0

    def setup(self, dynamics):
        title = ' '.join(self.keys)
        print(title, flush=True)

    def __call__(self, batch):
        reporter = get_current_reporter()
        out = str(self.step) + \
            ' '.join(str(reporter.observation[key]) for key in self.keys)
        print(out, flush=True)
        self.step += 1


class JsonDumper(Extension):
    def __init__(self, path):
        self.path = path

    def __call__(self, batch):
        reporter = get_current_reporter()
        observation = reporter.observation
        torecord = {}
        for key in observation:
            torecord[key] = to_device(-1, observation[key]).tolist()
        append_json(torecord, self.path)

    def setup(self, dynamics):
        pass


def append_json(data: dict, path: str):
    with open(path, 'ab+') as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(json.dumps([data]).encode())
        else:
            f.seek(-1, 2)
            f.truncate()
            f.write(' , '.encode())
            f.write(json.dumps(data).encode())
            f.write(']'.encode())


def main():
    params = {
        "num_elements": 3,
        "aev_params": {
            "radial": {
                "cutoff": 9.0,
                "head": 0.7,
                "tail": 9.0,
                "step": 0.25,
                "sigma": 0.25
            },
            "angular": {
                "cutoff": 3.5,
                "head": 0.7,
                "tail": 3.5,
                "step": 0.4,
                "sigma": 0.4,
                "ndiv": 9,
                "zeta": 32.0
            }
        },
        "nn_params": {
            "n_layers": [[128, 128, 1], [128, 128, 1], [128, 128, 1]],
            "act": gaussian
        },
        "cutoff": 9.0,
        "pbc": [True, True, True],
        "n_agents": 4,
        "order": ["H", "C", "Pt"]
    }
    dtype = chainer.config.dtype
    cell = np.eye(3) * 10.0
    min_distance = 2.7
    n_atoms = 10
    max_cycle = 20
    n_batch = 3
    device_id = -1
    ratio = np.array([1.0, 1.0, 1.0])
    batch = random_batch(cell, min_distance, n_atoms, max_cycle, ratio, n_batch,
                         params, 'result/best_model', masses=np.array([10.0, 10.0, 10.0]))
    efv = ANI1ForceField(params, 'result/best_model', BasicNeighborList(9.0))
    batch.to_device(device_id)
    efv.model.to_device(device_id)
    kbt = np.ones(n_batch).astype(dtype) * 600 * kB
    dt = np.ones(n_batch).astype(dtype) * 0.2
    md = VelocityScaling(batch, efv, dt, kbt)
    md.extend(ANI1Reporter(params['order']))
    md.extend(XYZDumper('traj/md'))
    md.extend(PrintReport(['quantities/times']))
    md.extend(JsonDumper('out.json'))
    md.run(10000)


def random_batch(cell, min_distance, n_atoms,
                 max_cycle, ratio, n_batch, params, path, masses=None):
    order = np.array(params['order'])
    positions = [random_coordinates(cell, min_distance, n_atoms, max_cycle)
                 for _ in range(n_batch)]
    symbols = [random_symbols(order, ratio, len(p)) for p in positions]
    cells = np.array([cell for _ in range(n_batch)])
    velocities = [np.random.random(p.shape) * 2 - 1 for p in positions]
    t0 = np.zeros(n_batch, dtype=np.int32)
    return ANI1MolecularDynamicsBatch(symbols, cells, positions, velocities, t0, params, path, masses)


main()
