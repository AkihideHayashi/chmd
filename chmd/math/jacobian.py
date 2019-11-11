import chainer
from chainer import Variable, grad, functions as F

def jacobian(fx, x):
    xp = chainer.backend.get_array_module(x)
    def inner(i):
        return Variable(xp.eye(len(fx))[i])
    j = F.concat([grad([fx], [x], [inner(i)])[0][None, :] for i in range(len(fx))], axis=0)
    return j.data