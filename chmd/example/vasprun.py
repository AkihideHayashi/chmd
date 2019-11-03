"""Parse vasprun.xml and dump to pickle."""
import argparse
import logging
from chmd.database.pickle import make_dataset


def main():
    """Main."""
    parser = argparse.ArgumentParser(description='Convert vasprun.xml to hdf.')
    parser.add_argument('--vaspruns', required=True,
                        help='Directory where vasprun.xml s are exist.')
    parser.add_argument('--pklout', required=True,
                        help='Output pickle file.')
    parser.add_argument('--pklinp', required=False,
                        help='Pickle file that already exists')
    parser.add_argument('--maninp', required=True,
                        help='manager json file')
    parser.add_argument('--manout', required=True,
                        help='manager json file')
    args = parser.parse_args()
    vaspruns_dir = args.vaspruns
    pklout = args.pklout
    pklinp = args.pklinp
    maninp = args.maninp
    manout = args.manout
    logging.basicConfig(level=logging.INFO)
    make_dataset(vaspruns_dir=vaspruns_dir,
                 pklinp=pklinp,
                 pklout=pklout,
                 maninp=maninp,
                 manout=manout,
                 )


if __name__ == '__main__':
    main()
