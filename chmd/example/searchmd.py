import numpy as np
from chainer import Chain, Variable
from chmd.models.ani import Coordinates2Energy
from chmd.links.multi import MeanVariance
from chmd.dynamics.batch import Batch

    


class ForceField(Chain):
    def __init__(self, params, n):
        super().__init__()
        with self.init_scope():
            self.mean_var = MeanVariance([Coordinates2Energy(**params) for _ in range(n)])
    
    def forward(self, batch):



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
        }
    }
    mv = MeanVariance([Coordinates2Energy(**params) for _ in range(4)])

