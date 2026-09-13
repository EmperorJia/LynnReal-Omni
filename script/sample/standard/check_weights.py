"""Fast preflight shared by all Standard shell launchers; no CUDA imports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from model.weights import standard_transformer


def main():
    args = sys.argv[1:]
    if '--help' in args or '-h' in args:
        return
    weights = ROOT / 'weight/standard'
    for i, value in enumerate(args):
        if value == '--weights' and i + 1 < len(args):
            weights = Path(args[i + 1])
        elif value.startswith('--weights='):
            weights = Path(value.split('=', 1)[1])
    try:
        standard_transformer(weights)
    except (OSError, ValueError, KeyError, IndexError) as error:
        raise SystemExit(f'Standard transformer preflight failed: {error}')


if __name__ == '__main__':
    main()
