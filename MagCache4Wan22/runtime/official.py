"""Execute the pinned upstream functions, without rewriting their algorithm."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "vendor/magcache_generate.py"
SOURCE_SHA256 = "9455858d01a7ed55f6e6ba97358354f34ba9f0f5a8d03a967d9abb42d0c3d10d"
FUNCTIONS = {
    "nearest_interp", "save_json", "get_timesteps", "magcache_calibration",
    "magcache_forward", "init_magcache", "init_magcache_calibration",
}
STATE_NAMES = (
    "forward", "cnt", "num_steps", "split_step", "mode", "magcache_thresh",
    "K", "accumulated_err", "accumulated_steps", "accumulated_ratio",
    "retention_ratio", "residual_cache", "mag_ratios", "norm_ratio",
    "norm_std", "cos_dis",
)
MISSING = object()


def load_official(sinusoidal_embedding=None):
    """Compile original function ASTs; omit upstream CLI imports and main()."""
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("vendored MagCache source SHA256 mismatch")
    tree = ast.parse(data, filename=str(SOURCE))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    if {n.name for n in nodes} != FUNCTIONS:
        raise ValueError("incomplete upstream function set")
    if sinusoidal_embedding is None:
        from wan.modules.model import sinusoidal_embedding_1d
        sinusoidal_embedding = sinusoidal_embedding_1d
    namespace = dict(np=np, torch=torch, F=F, json=json, Optional=Optional,
                     Union=Union, List=List, Tuple=Tuple,
                     sinusoidal_embedding_1d=sinusoidal_embedding)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    generate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    t2v = next(n for n in generate.body if isinstance(n, ast.If)
               and ast.unparse(n.test) == "'t2v' in args.task")
    assignments = [n for n in ast.walk(t2v.body[2]) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "mag_ratios" for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError("cannot identify the official T2V magnitude table")
    ratios = ast.literal_eval(assignments[0].value)
    if len(ratios) != 78:
        raise ValueError("unexpected official T2V table length")
    return SimpleNamespace(**{n: namespace[n] for n in FUNCTIONS},
                           ratios=ratios, namespace=namespace)


def method_args(threshold=.06, k=2, retention_ratio=.4, sample_steps=50):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("K must be a nonnegative integer")
    # Exact per-stage initialization coverage is checked by official_forward.
    if not math.isfinite(retention_ratio) or not 0 < retention_ratio <= 1:
        raise ValueError("retention_ratio must be in (0, 1] to initialize the official cache")
    if isinstance(sample_steps, bool) or not isinstance(sample_steps, int) or sample_steps < 1:
        raise ValueError("sample_steps must be positive")
    return SimpleNamespace(magcache_thresh=threshold, magcache_K=k,
                           retention_ratio=retention_ratio, sample_steps=sample_steps)


@contextmanager
def official_forward(models, *, core, args, split_steps, calibration_dir=None):
    """One fresh official CLI lifecycle, restored even after an exception.

    Inside this scope the original shared-class counter/cache/accumulators and
    forced-full behavior are untouched. Reinitializing BETWEEN videos matches
    invoking the upstream CLI separately, and prevents its multi-video leak.
    """
    models = list(models)
    if not models or any(type(m) is not type(models[0]) for m in models):
        raise ValueError("official A14B requires the same unwrapped WanModel class")
    if (calibration_dir is None and args.magcache_thresh > 0 and args.magcache_K > 0
            and int(split_steps * 2 * args.retention_ratio) < 2):
        raise ValueError("retention must protect both initial CFG calls before official cache reuse")
    cls = type(models[0])
    if getattr(cls, "_magcache4wan22_active", False):
        raise RuntimeError("concurrent/nested MagCache sessions on one class are unsupported")
    class_state = {n: cls.__dict__.get(n, MISSING) for n in STATE_NAMES}
    instance_state = [{n: m.__dict__.get(n, MISSING) for n in STATE_NAMES} for m in models]
    cls._magcache4wan22_active = True
    try:
        if calibration_dir is None:
            core.init_magcache(models[0], list(core.ratios), args, split_steps)
        else:
            destination = Path(calibration_dir)
            destination.mkdir(parents=True, exist_ok=False)
            def save_json(filename, values):
                (destination / (filename + ".json")).write_text(json.dumps(values) + "\n")
            core.namespace["save_json"] = save_json
            core.init_magcache_calibration(models[0], args)
        # A previous observer may have left an instance-bound native forward.
        # Expose the newly installed original class forward for this video.
        for model in models:
            for name in STATE_NAMES:
                if name in model.__dict__:
                    delattr(model, name)
        yield
    finally:
        for model, saved in zip(models, instance_state):
            for name, original in saved.items():
                if original is MISSING:
                    if name in model.__dict__:
                        delattr(model, name)
                else:
                    setattr(model, name, original)
        for name, original in class_state.items():
            if original is MISSING:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, original)
        delattr(cls, "_magcache4wan22_active")


@contextmanager
def record_calls(models, records, *, calibration=False):
    """Observe actual block execution and scalar state; never choose actions."""
    handles = []
    original_forwards = []
    active = [None]
    try:
        for stage, model in zip(("high", "low"), models):
            original_forwards.append(model.forward)
            def count_block(_module, _inputs):
                if active[0] is not None:
                    active[0]["blocks_executed"] += 1
            for block in model.blocks:
                handles.append(block.register_forward_pre_hook(count_block))
            original = model.forward
            @wraps(original)
            def observed(*inputs, _model=model, _stage=stage, _original=original, **kwargs):
                counter = int(_model.cnt)
                branch = counter % 2
                row = dict(call_index=counter, step_index=counter // 2,
                           model_stage=_stage, cfg_branch=("cond", "uncond")[branch],
                           blocks_executed=0)
                if not calibration:
                    row.update(magnitude_ratio=float(_model.mag_ratios[counter]),
                               accumulated_ratio_before=float(_model.accumulated_ratio[branch]),
                               accumulated_error_before=float(_model.accumulated_err[branch]),
                               accumulated_steps_before=int(_model.accumulated_steps[branch]))
                active[0] = row
                try:
                    result = _original(*inputs, **kwargs)
                finally:
                    active[0] = None
                row["action"] = "reuse" if row["blocks_executed"] == 0 else "recompute"
                if not calibration:
                    row.update(accumulated_ratio_after=float(_model.accumulated_ratio[branch]),
                               accumulated_error_after=float(_model.accumulated_err[branch]),
                               accumulated_steps_after=int(_model.accumulated_steps[branch]))
                records.append(row)
                return result
            model.forward = observed
        yield
    finally:
        for handle in handles:
            handle.remove()
        for model, original in zip(models, original_forwards):
            model.forward = original
