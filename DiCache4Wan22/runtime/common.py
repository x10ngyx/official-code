"""Fixed model protocol, output layout, and provenance for DiCache4Wan22."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
WORKSPACE = REPOSITORY.parents[1]
EXP_ROOT = Path("/all/yiran07-disk3/huteng_data/exp").resolve()
MODEL_ROOT = WORKSPACE / "models"
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def check_environment():
    if Path(sys.prefix).name != "wan2.2":
        raise RuntimeError("use the conda wan2.2 Python interpreter")
    if any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise RuntimeError("set all four BLAS/NumPy thread variables to 1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("the fixed protocol uses one GPU without FSDP/SP")


def external_output(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(EXP_ROOT) or path == EXP_ROOT:
        raise ValueError(f"result must be below {EXP_ROOT}")
    return path


def checkpoint_path(path):
    path = Path(path).expanduser().absolute()
    # Both the model root and its registered model directories may be symlinks.
    registered = [MODEL_ROOT.resolve()] + [p.resolve() for p in MODEL_ROOT.iterdir() if p.is_dir()]
    if not any(path.resolve().is_relative_to(root) for root in registered):
        raise ValueError(f"checkpoint must be stored under {MODEL_ROOT}")
    for part in ("high_noise_model", "low_noise_model"):
        if not (path / part / "config.json").is_file():
            raise ValueError(f"missing Wan2.2 A14B checkpoint component: {part}")
    return path.resolve()


def validate_source(source):
    source = Path(source).expanduser().resolve()
    lock = read_json(PROJECT / "upstream_lock.json")
    if not lock.get("package_file_sha256"):
        raise ValueError("package source lock must not be empty")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["wan22"]["commit"]:
        raise ValueError("Wan2.2 commit mismatch")
    # Require all tracked source bytes to equal the commit, including scheduler
    # and attention files outside the compact lock. No cached-method patches.
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=True)
    untracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"], text=True)
    if any(Path(name).suffix == ".py" for name in untracked.splitlines()):
        raise ValueError("pristine Wan2.2 source contains untracked Python files")
    for relative, digest in lock["wan22"]["original_file_sha256"].items():
        if sha256(source / relative) != digest:
            raise ValueError(f"Wan2.2 file mismatch: {relative}")
    for relative, digest in lock["package_file_sha256"].items():
        if sha256(PROJECT / relative) != digest:
            raise ValueError(f"DiCache package file mismatch: {relative}")
    return dict(schema="dicache4wan22_prepared_v1", status="pass", mode="prepared",
                source=str(source), wan22_commit=commit,
                dicache_commit=lock["dicache"]["commit"],
                package_file_sha256=lock["package_file_sha256"])


def index_result(path):
    path = external_output(path)
    # Index each top-level experiment once, including nested per-video/profile outputs.
    path = EXP_ROOT / path.relative_to(EXP_ROOT).parts[0]
    link = PROJECT / "experiment_results" / path.name
    if link.is_symlink() and link.resolve() == path:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"result index collision: {link}")
    link.symlink_to(path, target_is_directory=True)


def validate_config(cfg):
    values = dict(sample_shift=12.0, boundary=.875, sample_guide_scale=(3.0, 4.0),
                  sample_fps=16, num_train_timesteps=1000, dim=5120, num_layers=40)
    for name, expected in values.items():
        value = getattr(cfg, name)
        if name == "sample_guide_scale":
            value = tuple(value)
        if value != expected:
            raise ValueError(f"Wan2.2 config mismatch: {name}={value}")
    if str(cfg.param_dtype) != "torch.bfloat16":
        raise ValueError("DiT dtype must be BF16")


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))
