"""Conditioner for original appearance plus a separately sized causal boundary image."""
import argparse
from pathlib import Path
import runpy
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundary-edge', type=int, choices=(128, 768), required=True)
    args, rest = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model.boundary_reference import configure_boundary_reference
    configure_boundary_reference(args.boundary_edge)
    entry = root / 'script/conditioner.py'
    sys.argv = [str(entry), *rest]
    runpy.run_path(str(entry), run_name='__main__')
