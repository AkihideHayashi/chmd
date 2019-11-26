"""Parser of vasprun.xml."""
import xml.etree.ElementTree as ET
import numpy as np
import os


def find_attrib(tree, key, val):
    """Find section which have the attribute."""
    for child in tree:
        if key in child.attrib and child.attrib[key] == val:
            return child


def read_varray(varray):
    """Read varray section."""
    v = []
    for child in varray.findall('v'):
        v.append(child.text.split())
    return np.array(v).astype(np.float64)


def read_symbols(root):
    """Read symbols from root."""
    symbols = []
    s = find_attrib(root.find('atominfo'), 'name', 'atoms').find('set')
    for child in s:
        for i, c in enumerate(child):
            if i == 0:
                symbols.append(c.text.strip())
    return np.array(symbols)
    # return np.array(symbols, dtype=h5py.special_dtype(vlen=str))


def read_calculation(nelm: int, x):
    """Read calculation.

    Parameters
    ----------
    nelm: nelm tag (max iter).
    x: calculation section.

    Returns
    -------
    dict(cell, positions, energy, forces), converged

    """
    n = len(x.findall('scstep'))
    structure = x.find('structure')
    cell = read_varray(
        find_attrib(
            structure.find('crystal'), 'name', 'basis'))
    positions = read_varray(
        find_attrib(structure, 'name', 'positions'))
    forces = read_varray(find_attrib(x, 'name', 'forces'))
    energy = np.array(float(find_attrib(
        x.find('energy'), 'name', "e_wo_entrp").text.strip()))
    return {'cell': cell,
            'positions': positions,
            'energy': energy,
            'forces': forces}, nelm > n


def read_trajectory(path=None, generation=None, string=None):
    """Read vasprun.xml.

    Parameters
    ----------
    path: The path for vasprun.xml.

    Returns
    -------
    List[Dict]
    [{symbols, vasprun, generation, cell, positions, energy, forces}]

    symbols: str[n_atoms]
    vasprun: str
    generation: str
    cell: eV / Angstrome float[3, 3]
    positions: direct coordinate float[n_atoms, 3]
    energy: eV float
    forces: eV / Angstrome float[n_atoms, 3]

    """
    assert generation is not None
    if path:
        tree = ET.parse(path)
    elif string:
        tree = ET.fromstring(string)
    else:
        raise TypeError('read_trajectory must recieve path or string.')
    root = tree.getroot()
    nelm = int(find_attrib(root.find('incar'), 'name', 'NELM').text)
    symbols = read_symbols(root)
    trajectory = []
    for child in root.findall('calculation'):
        values, converged = read_calculation(nelm, child)
        if converged:
            values['symbols'] = symbols
            values['vasprun'] = os.path.basename(path)
            values['generation'] = generation
            trajectory.append(values)
    return trajectory
