"""Compatibility entry point for the fixed-anchor continuation ablation."""
import sys
from dense_stream import main
if __name__ == '__main__':
    sys.argv.append('--retain-anchor')
    main()
