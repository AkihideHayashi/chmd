"""Common preprocess."""
from abc import ABC, abstractmethod
import numpy as np
from chainer.datasets import open_pickle_dataset_writer, open_pickle_dataset

def symbols_to_elements(symbols: np.ndarray,
                        order: np.ndarray) -> np.ndarray:
    """Convert symbols to uniqued elements number.

    Parameters
    ----------
    symbols         : symbols that consists molecular
    order_of_symbols: unique symbols

    Returns
    -------
    elements: where order_of_symbols[elements] == symbols

    """
    shape = symbols.shape
    s = symbols.flatten()
    condlist = s[None, :] == order[:, None]
    # assert np.all(np.any(condlist, axis=0))
    choicelist = np.arange(len(order))
    elements = np.select(condlist, choicelist, default=-1).reshape(shape)
    valid = symbols != ''
    assert np.all(order[elements[valid]] == symbols[valid])
    return elements


class Preprocessor(ABC):
    """Abstract class for preprocessor."""

    @abstractmethod
    def process(self, datas, device):
        """Carry out preprocess.

        Parameters
        ----------
        datas: dict
        device: int

        """

    @abstractmethod
    def classify(self, data):
        """Calculate classify key."""


def preprocess(inp_path, out_path, batch_size, device, trans: Preprocessor):
    """Preprocess data."""
    print("preprocess", flush=True)
    datasets = {}
    with open_pickle_dataset(inp_path) as fi:
        with open_pickle_dataset_writer(out_path) as fo:
            for i, data in enumerate(fi):
                repeats = trans.classify(data)
                if repeats not in datasets:
                    datasets[repeats] = [data]
                else:
                    datasets[repeats].append(data)
                    if len(datasets[repeats]) > batch_size:
                        datas = datasets.pop(repeats)
                        print('{} process for {}'.format(i, repeats), flush=True)
                        trans.process(datas, device)
                        for d in datas:
                            fo.write(d)
            keys = list(datasets.keys())
            for repeats in keys:
                datas = datasets.pop(repeats)
                print('process for {}'.format(repeats), flush=True)
                trans.process(datas, device)
                for d in datas:
                    fo.write(d)