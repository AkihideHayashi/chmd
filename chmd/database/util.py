"""About path generation."""
import string
import random
from typing import List


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
