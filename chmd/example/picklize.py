from xml.etree import ElementTree
import tarfile
from chainer.datasets import open_pickle_dataset_writer
from chmd.database.vasprun import read_calculation, read_symbols, read_nelm
from chmd.models.ani import ANI1Preprocessor
from chmd.preprocess import preprocess
from chmd.functions.activations import gaussian

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
                    data['status'] = 'train'
                    data['symbols'] = symbols
                    data['generation'] = 'mdminimize'
                    data['vasprun'] = name
                    f.write(data)


def main():
    tarfile_path = '../../../note/vaspruns.tar'
    pickle_path = '../../../note/tmp.pkl'
    out_path = '../../../note/processed.pkl'
    batch_size = 200
    tarfile_to_pickle(tarfile_path, pickle_path)
    preprocess(pickle_path, out_path, batch_size, -1, ANI1Preprocessor(params))


if __name__ == "__main__":
    main()