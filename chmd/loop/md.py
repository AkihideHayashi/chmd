import numpy as np
from chainer import Variable, report
from chmd.functions.neighbors import number_repeats, compute_shifts
from chmd.math.xp import repeat_interleave, cumsum_from_zero
from chmd.dynamics.dynamics import VelocityScaling
from chmd.dynamics.batch import Batch
from chmd.models.ani import ANI1, EnergyForceVar
from chmd.preprocess import symbols_to_elements


def random_select_generator(ratio):
    assert ratio.ndim == 1
    while True:
        rand = np.random.random(len(ratio))
        yield np.argmax(rand * ratio)


def random_generation(
        unique_symbols,
        ratio,
        cell,
        minimum_distance,
        n_atoms,
        max_cycle
):
    selector = random_select_generator(ratio)
    symbols = []
    numbers = []
    positions = np.zeros((0, 3))
    shifts = compute_shifts(
        number_repeats(
            cell, np.array([True, True, True]), minimum_distance
        )
    )
    real_shifts = shifts @ cell
    for i, n in enumerate(selector):
        p = cell.T @ np.random.random(3)
        in_min_dist = np.any(
            np.linalg.norm(
                (positions[:, np.newaxis, :]
                 - p[np.newaxis, np.newaxis, :]
                 - real_shifts[np.newaxis, :, :]), axis=2)
            < minimum_distance)
        if in_min_dist:
            continue
        else:
            positions = positions.tolist()
            positions.append(p)
            positions = np.array(positions)
            numbers.append(n)
            if len(numbers) >= n_atoms:
                break
        if i > max_cycle:
            break
    numbers = np.array(numbers)
    sort = np.argsort(numbers)
    symbols = unique_symbols[numbers[sort]]
    positions = positions[sort]
    return symbols, positions


def gen():
    cell_size = 10.0
    unique_symbols = np.array(["H", "C", "Pt"])
    ratio = np.array([1.0, 1.0, 1.0])
    cell = np.eye(3) * cell_size
    minimum_distance = 1.2
    n_atoms = 10
    max_cycle = 20
    s, p = random_generation(unique_symbols,
                             ratio, cell, minimum_distance, n_atoms, max_cycle)
    return cell, symbols_to_elements(s, unique_symbols), p


def gens():
    cells = []
    elements = []
    positions = []
    for _ in range(40):
        c, e, p = gen()
        cells.append(c)
        elements.append(e)
        positions.append(p)
    n_atoms = np.array([len(e) for e in elements])
    i1 = repeat_interleave(n_atoms)
    cells = np.concatenate([c[None, :, :] for c in cells])
    positions = np.concatenate(positions)
    elements = np.concatenate(elements)
    batch = Batch(affiliations=i1,
                  cells=cells,
                  positions=positions,
                  elements=elements,
                  )

class Evaluator(object):
    def __init__(self, model):
        self.model = model
    
    def __call__(self, batch: Batch):
        positions = batch.positions
        elements = batch.elements
        cells = batch.cells
        e, f, v = self.model(positions=positions, elements=elements, cells=cells)
        batch.forces = f
        batch.potential_energies = e
        report({'var', v}, self)

class StoreAll(Extension):
    def __init__(self):
        self.dict = {}

    def setup(self, dynamics):
        pass

    def __call__(self, observation):
        for key in observation:
            if key not in self.dict:
                self.dict[key] = []
            self.dict[key].append(observation[key])

from ase.units import kB
def main():
    batch = gens()
    print(batch)
    model = ANI1()
    efv = EnergyForceVar(model)
    evalator = Evaluator(efv)
    kbt = 6000 * kB
    dt = np.ones(40) * 1.0
    md = VelocityScaling(batch, evalator, dt, kbt)
    reporter = StoreAll()
    md.extend(reporter)
    md.run(10)


main()
