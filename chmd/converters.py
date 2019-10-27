import sqlite3
import numpy as np
from chmd.neighbors import cumsum_from_zero
from chmd.preprocess import save, load


class SQLConverter(object):
    def __init__(self, path):
        self.path = path
        self.conn = None

    def all_ids(self):
        self.conn.row_factory = None
        ids = np.array([i[0] for i in self.conn.execute('SELECT id FROM atoms')])
        self.conn.row_factory = lambda *x: dict(sqlite3.Row(*x))
        return ids

    def __enter__(self):
        sqlite3.register_adapter(np.ndarray, save)
        sqlite3.register_converter('NDARRAY', load)
        self.conn = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = lambda *x: dict(sqlite3.Row(*x))
        return self

    def __exit__(self, typ, value, traceback):
        self.conn.close()

    def __call__(self, ids):
        ids = [int(i) for i in ids]
        mols = list(self.conn.execute(
                'SELECT elements, cell, positions, i2, j2, s2, energy, forces FROM atoms WHERE id IN ({})'.format(','.join('?' for _ in ids)),
                ids))
        i1 = np.concatenate([i * np.ones_like(atoms['elements']) for i, atoms in enumerate(mols)])
        head = cumsum_from_zero(np.array([len(atoms['elements']) for atoms in mols]))
        n2 = np.concatenate([i * np.ones_like(atoms['i2']) for i, atoms in enumerate(mols)])
        i2 = np.concatenate([atoms['i2'] for atoms in mols]) + head[n2]
        j2 = np.concatenate([atoms['j2'] for atoms in mols]) + head[n2]
        s2 = np.concatenate([atoms['s2'] for atoms in mols])
        positions = np.concatenate([atoms['positions'] for atoms in mols])
        cells = np.concatenate([atoms['cell'][None, :, :] for atoms in mols])
        elements = np.concatenate([atoms['elements'] for atoms in mols])
        return {'elements': elements,
                'cells': cells,
                'positions': positions,
                'i1': i1,
                'i2': i2,
                'j2': j2,
                's2': s2}
