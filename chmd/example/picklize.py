from xml.etree import ElementTree
import tarfile
from chainer.datasets import open_pickle_dataset_writer
from chmd.database.vasprun import read_calculation, read_symbols, read_nelm
from chmd.models.ani import preprocess, add_aev, AddAEV
from chmd.functions.activations import gaussian
from chmd.links.ani import ANI1AEV

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
        "n_layers": [[128, 128, 1], [128, 128, 1], [256, 256, 1]],
        "act": gaussian
    },
    "cutoff": 9.0,
    "pbc": [True, True, True],
    "n_agents": 4,
    "order": ["H", "C", "Pt"]
}

def tarfile_to_pickle(tarpath, picklepath):
    with tarfile.open(tarpath) as tar:
        n = sum(1 for member in tar if member.isreg())
        with open_pickle_dataset_writer(picklepath) as f:
            for i, tarinfo in enumerate(tar):
                print("{}/{}".format(i, n))
                name = tarinfo.name
                xml = tar.extractfile(tarinfo).read()
                root = ElementTree.fromstring(xml)
                nelm = read_nelm(root)
                symbols = read_symbols(root)
                calculations = root.findall('calculation')
                for calc in calculations:
                    data = read_calculation(calc, nelm)
                    if not data:
                        continue
                    if data['energy'] < 0.0:
                        data['status'] = 'train'
                    else:
                        data['status'] = 'drain'
                    data['symbols'] = symbols
                    data['generation'] = 'mdminimize'
                    data['vasprun'] = name
                    f.write(data)
import numpy as np
aev_calc = ANI1AEV(params['num_elements'], **params['aev_params'])

def trans(datas, device):
    add_elements_aev(aev_calc, datas, np.array(params['order']), device, params['cutoff'], params['pbc'])

def trans2(datas, device):
    add_elements(datas, params['order'])
    add_neighbors(datas, params['cutoff'], params['pbc'], device)
    add_aev(datas, aev_calc, device)
    for data in datas:
        del data['i2']
        del data['j2']
        del data['s2']


def main():
    tarfile_path = '../../../note/vaspruns.tar'
    pickle_path = '../../../note/tmp.pkl'
    out_path = '../../../note/processed3.pkl'
    batch_size = 200
    # tarfile_to_pickle(tarfile_path, pickle_path)
    preprocess(pickle_path, out_path, batch_size, params['pbc'], params['cutoff'], -1, AddAEV(params))


if __name__ == "__main__":
    main()