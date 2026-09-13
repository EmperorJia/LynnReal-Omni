"""Content-addressed inference source snapshots for reproducible sample records."""
import hashlib
import importlib.metadata
import uuid
from pathlib import Path
import sys


def snapshot_sources(cache):
    root = Path(__file__).resolve().parents[1]
    paths = {str(p.relative_to(root)): p for folder in ("model", "script", "tool")
             for p in (root / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".sh"}}
    for name, module in tuple(sys.modules.items()):
        if name.startswith("diffusers.") and "minimax_h3" in name and getattr(module, "__file__", None):
            paths[name] = Path(module.__file__)
    destination = Path(cache) / "source_snapshots"
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, path in sorted(paths.items()):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        saved = destination / (digest + path.suffix)
        if not saved.exists():
            temporary = saved.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(data)
            temporary.replace(saved)
        files[name] = {"path": str(path), "sha256": digest, "snapshot": str(saved)}
    versions = {}
    for name in ("torch", "diffusers", "transformers", "safetensors", "triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"files": files, "versions": versions}


def changed_sources(record):
    return [name for name, item in record["files"].items()
            if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]]
