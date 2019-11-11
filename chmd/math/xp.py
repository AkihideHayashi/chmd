"""Unified interface for numpy and cupy."""
import numpy as np
import chainer
from chainer.backend import get_array_module


def scatter_add(a, indices, b) -> None:
    """Scatter add operation for numpy or cupy.

    numpy.add.at and cupy.scatter_add are same functions.
    However, you can't use both only by switching xp.
    So you can use it for both of them.
    """
    xp = chainer.backend.get_array_module(indices)
    if xp is np:
        return xp.add.at(a, indices, b)
    else:
        return xp.scatter_add(a, indices, b)


def scatter_add_to_zero(shape, indices, b):
    """Scatter add to zeros and return it."""
    xp = chainer.backend.get_array_module(indices)
    a = xp.zeros(shape, dtype=b.dtype)
    scatter_add(a, indices, b)
    return a


def repeat_interleave(n: np.ndarray):
    """Repeat arange.

    >>> n = np.array([1, 1, 2, 2, 3, 4])
    >>> center, number = np.unique(n, return_counts=True)
    >>> number
    ... [2 2 1 1]
    >>> repeat_interleave(number)
    ... [0 0 1 1 2 3]
    """
    xp = get_array_module(n)
    n0 = len(n)
    n1 = int(xp.max(n))
    arange1, arange2, count = xp.broadcast_arrays(
        xp.arange(n1)[xp.newaxis, :],
        xp.arange(n0)[:, xp.newaxis],
        n[:, xp.newaxis]
    )
    mask = count > arange1
    ret = arange2[mask]
    return ret


def cumsum_from_zero(input_: np.ndarray):
    """Like xp.cumsum. But start from 0.

    >>> n = np.array([1, 1, 2, 2, 3, 4])
    >>> center, number = np.unique(n, return_counts=True)
    >>> number
    ... [2 2 1 1]
    >>> cumsum_from_zero(number)
    ... [0 2 4 5]
    """
    xp = get_array_module(input_)
    cumsum = xp.cumsum(input_, axis=0)
    cumsum = xp.roll(cumsum, 1)
    cumsum[0] = 0
    return cumsum


def cartesian_product(*args) -> np.ndarray:
    """Cartesian product of arrays.

    Examples
    --------
    >>> x = np.array([10, 20])
    >>> y = np.array([1, 2, 3])
    >>> cartesian_product(x, y)
    ... [[10  1]
         [10  2]
         [10  3]
         [20  1]
         [20  2]
         [20  3]]

    """
    xp = get_array_module(args[0])
    n = len(args)
    shapes = [tuple(len(x) if i == j else 1
                    for i in range(n)) for j, x in enumerate(args)]
    # x = tuple(a.reshape(shape) for a, shape in zip(args, shapes))
    broad = xp.broadcast_arrays(*tuple(a.reshape(shape)
                                       for a, shape in zip(args, shapes)))
    cat = xp.concatenate([xp.expand_dims(b, -1) for b in broad], axis=-1)
    return cat.reshape((-1, n))
