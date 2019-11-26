"""Future version of vasprun.py."""
from typing import Dict
from xml.etree import ElementTree
import numpy as np


@np.vectorize
def to_type(x, typename):
    """Transform type."""
    if typename == 'logical':
        if x == 'T':
            return True
        elif x == 'F':
            return False
        else:
            raise KeyError()
    elif typename == 'int':
        return int(x.strip())
    elif typename == 'string':
        return x
    elif typename == 'float':
        return float(x.strip())
    else:
        raise NotImplementedError()


def read_v(v: ElementTree.Element):
    """Read v tag."""
    if v.text is None:
        raise RuntimeError()
    array = np.array(v.text.split())
    if 'type' in v.attrib:
        if v.attrib['type'] == 'logical':
            @np.vectorize
            def read_logical(x):
                if x == 'T':
                    return True
                elif x == 'F':
                    return False
                else:
                    raise KeyError()
            return read_logical(array)
        else:
            raise NotImplementedError()
    else:
        return array.astype(np.float64)
    return array


def read_i(i: ElementTree.Element):
    """Read i tag."""
    if i.text is None:
        raise RuntimeError()
    array = np.array(i.text.strip())
    if 'type' in i.attrib:
        if i.attrib['type'] == 'logical':
            @np.vectorize
            def read_logical(x):
                if x == 'T':
                    return True
                elif x == 'F':
                    return False
                else:
                    raise KeyError()
            return read_logical(array)
        elif i.attrib['type'] == 'int':
            return array.astype(np.int64)
        else:
            raise NotImplementedError()
    else:
        return array.astype(np.float64)
    return array


def read_varray(varray):
    """Read varray tag."""
    return np.array([read_v(c) for c in varray])


def read_rc(rc):
    """Read rc c tag as array of str."""
    return tuple([c.text.strip() for c in rc])


def read_r(r):
    """Read r tag as array of str."""
    return r.text.split()


def read_set(set_):
    """Read set tag as array of str."""
    ret = []
    for child in set_:
        if child.tag == 'rc':
            ret.append(read_rc(child))
        elif child.tag == 'r':
            ret.append(read_r(child))
        elif child.tag == 'set':
            ret.append(read_set(child))
        else:
            raise NotImplementedError(child.tag)
    return np.array(ret)


def read_dimensions(dimensions):
    """Read dimension tags in array tag."""
    dims = [int(d.attrib['dim']) for d in dimensions]
    assert dims == list(range(1, len(dims) + 1))
    dimension_tags = [d.text for d in dimensions]
    return dimension_tags


def read_fields(fields):
    """Read fields tags in array tag."""
    types = [f.attrib['type'] if 'type' in f.attrib else 'float'
             for f in fields]
    field_names = [f.text for f in fields]
    return {n: t for t, n in zip(types, field_names)}


def read_array(array: ElementTree.Element):
    """Read an array tag.

    Parameters
    ----------
    array: array tag.

    Returns
    -------
    dimensions: names of each dimension.
    array: structured array which contains elements.

    """
    dimensions = read_dimensions(array.findall('dimension'))
    fields = read_fields(array.findall('field'))
    set_ = read_set(array.find('set'))
    tmp = {name: to_type(np.take(set_, i, axis=-1), fields[name])
           for i, name in enumerate(fields)}
    return list(reversed(dimensions)), dict_to_structured_array(tmp)


def dict_to_structured_array(dic: Dict[str, np.ndarray]):
    """Transform dict of array to structured array."""
    dtype = [(t, v.dtype) for t, v in dic.items()]
    shape = next(iter(dic.values())).shape
    tmp = np.zeros(shape, dtype)
    for key, val in dic.items():
        tmp[key] = val
    return tmp


def read_vasprun_symbols(vasprun):
    """Get symbol list."""
    dim_name, array = read_array(vasprun.find('atominfo/array[@name="atoms"]'))
    assert dim_name == ['ion']
    return array['element']


def read_vasprun_selective(vasprun):
    """Get Selective Dynamics information."""
    varray = read_varray(vasprun.find(
        'structure[@name="initialpos"]/varray[@name="selective"]'))
    return varray

