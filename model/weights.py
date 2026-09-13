"""Local model locations and artifact checksums."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1] / "weight"

def local_weights(path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Model weights must be inside the release weight directory")
    return path

def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def standard_transformer(weights):
    """Validate the one physical Standard DiT directory without loading tensors."""
    weights = local_weights(weights)
    config = json.loads((weights / 'inference_config.json').read_text())
    if config.get('variant') != 'standard' or config.get('steps') != 4:
        raise ValueError('Standard launchers require a four-step standard bundle')
    directory = weights / 'transformer'
    index_path = directory / 'diffusion_pytorch_model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    shards = set(index['weight_map'].values())
    if not shards or any(Path(s).name != s or not (directory / s).is_file() for s in shards):
        raise ValueError('Standard transformer index contains missing or invalid shard paths')
    json.loads((directory / 'config.json').read_text())
    components = json.loads((weights / 'modular_model_index.json').read_text())
    for name in ('transformer', 'transformer_ref'):
        spec = components[name][2]
        if spec.get('subfolder') != 'transformer' or spec.get('pretrained_model_name_or_path') != '.':
            raise ValueError(f'{name} must load the shared Standard transformer/ directory')
    return {'directory': str(directory.resolve()), 'shards': len(shards),
            'index_sha256': sha256(index_path), 'config_sha256': sha256(directory / 'config.json')}

