"""Multiagent for calculate variance."""
from chainer import ChainList, Chain


class MeanVariance(Chain):
    def __init__(self, agents):
        super().__init__()
        with self.init_scope():
            self.agents = agents
    
    def forward(self, *args, **kwargs):
        n = len(self.agents)
        predicts = [l(*args, **kwargs) for l in self.agents]
        mean = sum(predicts) / n
        var = sum(p ** 2 for p in predicts) / n - mean * mean
        return mean, var
