import numpy as np
import matplotlib.pyplot as plt
from chmd.dynamics.dynamics import NoseHooverChain, BasicNoseHooverBatch, MolecularDynamicsBatch, nose_hoover_conserve
from chmd.dynamics.nosehoover import setup_nose_hoover_chain_parallel
from chmd.dynamics.analyze import calculate_kinetic_energies, calculate_temperature
from chmd.preprocess import symbols_to_elements
from chmd.dynamics.dynamics import Extension
from chainer import report
from chainer.dataset import to_device
from chainer.datasets import PickleDatasetWriter, open_pickle_dataset
from chainer import get_current_reporter


class SimpleBatch(BasicNoseHooverBatch):
    def __init__(self, symbols, positions, velocities, masses, t0,
                 numbers, targets, timeconst, kbt, order):
        (sym, pos, vel, mas, nu, ta, kbt, isa, ist) = setup_nose_hoover_chain_parallel(
            symbols, positions, velocities, masses,
            numbers, targets, timeconst, kbt
        )
        ele = symbols_to_elements(sym, np.array(order))
        n_batch, _, n_dim = pos.shape
        cells = np.array([np.eye(n_dim) for _ in range(n_batch)])
        super().__init__(ele, cells, pos, vel, mas, t0,
                         kbt, nu, ta, isa, ist)

class CF(object):
    def __init__(self):
        self.name = 'cf'
    def __call__(self, batch: SimpleBatch):
        positions = batch.positions
        vec = np.linalg.norm(positions, axis=2)
        potential = 0.5 * np.sum(vec * vec, axis=1)
        forces = - positions
        batch.potential_energies = potential
        batch.forces = forces

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
            'is_atom': batch.is_atom,
            'velocities': batch.velocities,
            'times': batch.times,
            'masses': batch.masses
        }
        report(to_report)

class PickleDumper(Extension):
    def __init__(self, path):
        self.path = path
        self.writer = open(path, 'wb')
        self.pickle = PickleDatasetWriter(self.writer)
    
    def setup(self, dynamics):
        pass

    def __call__(self, batch):
        reporter = get_current_reporter()
        observation = reporter.observation
        torecord = {}
        for key in observation:
            torecord[key] = to_device(-1, observation[key]).tolist()
        self.pickle.write(torecord)

class NoseHooverExtension(Extension):
    def __init__(self):
        pass
    
    def setup(self, dynamics):
        pass

    def __call__(self, batch: SimpleBatch):
        xp = batch.xp
        positions = batch.positions
        i1 = xp.broadcast_to(xp.arange(positions.shape[0])[:, None, None], positions.shape).flatten()
        conserve = nose_hoover_conserve(
            batch.positions.flatten(),
            batch.velocities.flatten(),
            batch.masses.flatten(),
            batch.numbers,
            batch.targets,
            batch.kbt.flatten(),
            batch.potential_energies,
            i1
        )
        kinetic = calculate_kinetic_energies(batch.cells, batch.velocities, batch.masses, batch.is_atom)
        dof = xp.sum(batch.is_atom, axis=1) * batch.positions.shape[2]
        temperature = calculate_temperature(kinetic, dof)
        report({
            'conserve': conserve,
            'kinetic': kinetic,
            'energies': kinetic + batch.potential_energies,
            'temperature': temperature
        })

def main():
    positions = [
        [[0.0, 1.0],
         [0.0, 1.0]],

        [[0.0, 1.0],
         [0.0, 1.0]]
    ]

    velocities = [
        [[1.0, 0.0],
         [1.0, 0.0]],

        [[1.0, 0.0],
         [1.0, 0.0]]
    ]

    symbols = [
        ['H', 'H'],
        ['H', 'H']
    ]
    masses = [
        [1.0, 3.0],
        [1.0, 1.0]
    ]
    t0 = np.zeros(2)
    numbers = [
        [-1, -1],
        [-1, -1]
    ]
    targets = [
        [0, 1],
        [0, 1]
    ]
    timeconst = [
        [10.0],
        [10.0]
    ]
    kbt = [
        [10.0],
        [10.0]
    ]
    dt = np.array([0.1, 0.1])
    order = ['H']
    batch = SimpleBatch(symbols, positions, velocities, masses, t0,
                        numbers, targets, timeconst, kbt, order)
    ff = CF()
    md = NoseHooverChain(batch, ff, dt)
    md.extend(MDReporter(order))
    md.extend(NoseHooverExtension())
    md.extend(PickleDumper('test.pkl'))
    md.run(1000)

    with open_pickle_dataset('test.pkl') as f:
        positions = np.array([data['positions'] for data in f])
        potential_energies = np.array([data['potential_energies'] for data in f])
        conserve = np.array([data['conserve'] for data in f])
        energies = np.array([data['energies'] for data in f])
        temperature = np.array([data['temperature'] for data in f])
        times = np.array([data['times'] for data in f])

    plt.plot(times[:, 0], conserve[:, 0])
    plt.plot(times[:, 0], energies[:, 0])
    plt.plot(times[:, 0], temperature[:, 0])
    plt.show()


                

main()
