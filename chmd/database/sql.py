"""Save and load using sql."""
import io
import numpy as np


def save(**kwargs):
    """Compressed byte repression."""
    tmp = io.BytesIO()
    np.savez_compressed(tmp, **kwargs)
    return tmp.getvalue()


def load(value: bytes):
    """Load bytes to np.ndarray."""
    tmp = io.BytesIO(value)
    return np.load(tmp)


# def save(array):
#     """Save np.ndarray to byte."""
#     tmp = io.BytesIO()
#     np.save(tmp, array, allow_pickle=False)
#     return tmp.getvalue()
# 
# 
# def load(byte):
#     """Load bytes to np.ndarray."""
#     return np.load(io.BytesIO(byte), allow_pickle=False)
# 
# 
# def savetxt(array):
#     """Save np.ndarray to byte."""
#     tmp = io.StringIO()
#     np.savetxt(tmp, array)
#     return tmp.getvalue()
# 
# 
# def loadtxt(byte):
#     """Load bytes to np.ndarray."""
#     return np.loadtxt(io.StringIO(byte))
# 