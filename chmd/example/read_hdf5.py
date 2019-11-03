"""Read hdf5."""
import argparse
from chmd.database.hdf5 import HDF5Recorder


def main():
    """Read hdf5."""
    parser = argparse.ArgumentParser(description='Read hdf5.')
    parser.add_argument('path', help='path.')
    parser.add_argument('key', help='key.')
    args = parser.parse_args()
    path = args.path
    key = args.key
    with HDF5Recorder(path, 'r') as recorder:
        print(dict(recorder.attrs))
        for k, v in recorder[key].items():
            print(k)
            print(v)


if __name__ == '__main__':
    main()
