import os
from pathlib import Path
from typing import List
import tarfile
from chmd.database.util import random_path


def add_to_tarfile(paths: List[Path], addto: Path):
    """Add and delte."""
    with tarfile.open(str(addto), 'a') as tar:
        names = [tarinfo.name for tarinfo in tar]
        for path in paths:
            new_name = random_path(names, path.suffix)
            names.append(new_name)
            tar.add(str(path), new_name)
            os.remove(str(path))
