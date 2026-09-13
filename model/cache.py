"""Publish complete conditioning caches safely across concurrent samplers."""
from pathlib import Path
import tempfile
import torch


def save_cache(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
