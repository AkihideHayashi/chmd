"""About path generation."""
import string
import random
from typing import List
import numpy as np


def random_string(num: int):
    """Generate random string."""
    dat = string.digits + string.ascii_lowercase + string.ascii_uppercase
    return ''.join([random.choice(dat) for i in range(num)])


def random_path(exclude: List[str], ext: str):
    """Generate random path which is not in exclude."""
    n = 1
    while True:
        s = random_string(n) + ext
        if s not in exclude:
            return s
        else:
            n += 1


def define_kpoints(cell):
    """Define required kpoints from the fact that bulk Pt requires 12."""
    inv = np.linalg.inv(cell)
    b = np.linalg.norm(inv, axis=1)
    pt_12 = 0.03682081
    return tuple(np.ceil(b / pt_12).astype(np.int64).tolist())
