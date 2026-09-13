"""Run the background editor with exact source-head latent preservation."""
from pathlib import Path
import runpy
import sys


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model import background_flow45_edit
    from model.boundary_flow_edit import configure_boundary_flow
    background_flow45_edit.configure_background_flow = configure_boundary_flow
    entry = root / 'script/background_chunk_editor.py'
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name='__main__')
