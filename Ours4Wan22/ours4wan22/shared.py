"""Load repository implementations under private names, without copying them."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = PROJECT.parent
WORKSPACE = OFFICIAL.parents[1]
MODELS = WORKSPACE / 'models'
EXP = Path('/all/yiran07-disk3/huteng_data/exp')


def load(name, relative):
    name = '_ours22_' + name
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, OFFICIAL / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def under(path, root):
    logical = Path(os.path.abspath(Path(path).expanduser()))
    # The workspace models registry intentionally contains links to disk-backed weights.
    model_alias = Path(root) == MODELS and logical != MODELS and logical.is_relative_to(MODELS)
    path, root = logical.resolve(), Path(root).resolve()
    if model_alias:
        return path
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'path must be below {root}: {path}')
    return path


def result_dir(path, description):
    path = under(path, EXP)
    link = PROJECT / 'experiment_results' / path.name
    if path.exists() or link.exists() or link.is_symlink():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(path, target_is_directory=True)
    (path / 'README.md').write_text(description + '\n')
    return path


def verify_sources():
    lock = read(PROJECT / 'source_lock.json')
    for relative, expected in lock['files'].items():
        if sha256(OFFICIAL / relative) != expected:
            raise ValueError(f'reference implementation changed: {relative}')
    return lock


def implementation_hashes():
    return {str(p.relative_to(PROJECT)):sha256(p)
            for p in sorted(PROJECT.rglob('*.py'))
            if 'experiment_results' not in p.parts}
