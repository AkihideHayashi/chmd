from chainer import ChainList, functions as F


def concat0(array):
    """Concat by new axis 0."""
    return F.concat([F.expand_dims(a, 0) for a in array], axis=0)


class EnsemblePredictor(ChainList):

    def __init__(self, n, initializer):
        models = [initializer() for _ in range(n)]
        super().__init__(*models)
    
    def forward(self, *args, **kwargs):
        return concat0([model(*args, **kwargs) for model in self])