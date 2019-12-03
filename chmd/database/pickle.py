"""Read vasprun and dump to pickle."""
import os
import logging
import json
import tarfile
from xml.etree import ElementTree
import numpy as np
from chainer.backend import get_array_module
from chainer.datasets import open_pickle_dataset_writer, open_pickle_dataset
from chmd.database.vasprun import read_symbols, read_calculation, read_nelm, read_kpoints
from chmd.preprocess import symbols_to_elements
from chmd.functions.neighbors import number_repeats, neighbor_duos
from chmd.database.util import good_kpoints


def merge_pickle(inps, out):
    with open_pickle_dataset_writer(out) as fo:
        for inp in inps:
            with open_pickle_dataset(inp) as fi:
                for data in fi:
                    fo.write(data)


def tarfile_to_pickle(tarpath, picklepath):
    """Make pickle file from tar vaspruns."""
    print("tarfile_to_pickle", flush=True)
    with tarfile.open(tarpath) as tar:
        n = sum(1 for member in tar if member.isreg())
        with open_pickle_dataset_writer(picklepath) as f:
            for i, tarinfo in enumerate(tar):
                print("{}/{}".format(i, n), flush=True)
                name = tarinfo.name
                xml = tar.extractfile(tarinfo).read()
                root = ElementTree.fromstring(xml)
                nelm = read_nelm(root)
                kpt = tuple(read_kpoints(root).tolist())
                symbols = read_symbols(root)
                calculations = root.findall('calculation')
                for calc in calculations:
                    data = read_calculation(calc, nelm)
                    kpt_good = good_kpoints(data['cell'])
                    if kpt_good != (1, 1, 1) and kpt == (1, 1, 1):
                        break
                    if not data:
                        continue
                    data['status'] = 'train'
                    data['symbols'] = symbols
                    data['generation'] = 'mdminimize'
                    data['vasprun'] = name
                    f.write(data)