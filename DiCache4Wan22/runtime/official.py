"""Unmodified official DiCache block algorithm in the native Wan2.2 forward.

Only embedding/head interfaces and the two-expert lifecycle are adapted.
The pinned upstream gate, DCTA, window updates and tensor operations are
compiled directly from the vendor AST, including its original edge cases.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import math
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace, MethodType

import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "vendor/run_wan_dicache.py"
SOURCE_SHA256 = "49edfa4033d1af2983a6db07354d87b142cacdf232566bb37a2c2b7b7853ed04"
NATIVE_SHA256 = "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432"
STATE_NAMES = (
    "cnt", "num_steps", "probe_depth", "rel_l1_thresh", "ret_ratio",
    "accumulated_rel_l1_distance", "residual_cache", "probe_residual_cache",
    "residual_window", "probe_residual_window", "previous_internal_states",
    "previous_input", "previous_output", "resume_flag",
)
MISSING = object()


def load_official():
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("official DiCache source SHA256 mismatch")
    tree = ast.parse(data, filename=str(SOURCE))
    forward = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dicache_forward")
    start = next(i for i, n in enumerate(forward.body) if ast.unparse(n) == "skip_forward = False")
    stop = next(i for i, n in enumerate(forward.body) if ast.unparse(n) == "x = self.head(x, e)")
    algorithm = copy.deepcopy(forward.body[start:stop])
    fn = ast.parse("def dicache_blocks(self, x, kwargs):\n    pass\n").body[0]
    # Return diagnostics after the original statements. No expression or branch
    # in the algorithm is changed, including x += ..., division and clip.
    fn.body = algorithm + ast.parse(
        "return x, {'skip': skip_forward, 'gamma': locals().get('gamma'), "
        "'delta_y': locals().get('delta_y')}"
    ).body
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(blocks=namespace["dicache_blocks"],
                           algorithm_ast=ast.dump(ast.Module(body=algorithm, type_ignores=[]), include_attributes=False))


def method_args(threshold=.08, retention_ratio=.2, sample_steps=50):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if isinstance(sample_steps, bool) or not isinstance(sample_steps, int) or sample_steps < 1:
        raise ValueError("sample_steps must be a positive integer")
    if (not math.isfinite(retention_ratio) or not 0 < retention_ratio <= 1
            or int(sample_steps * 2 * retention_ratio) < 2):
        raise ValueError("retention_ratio must protect both initial CFG calls and be <= 1")
    return SimpleNamespace(rel_l1_thresh=threshold, ret_ratio=retention_ratio,
                           sample_steps=sample_steps, probe_depth=1)


def initialize(model, args):
    model.cnt, model.num_steps = 0, args.sample_steps * 2
    model.probe_depth, model.rel_l1_thresh, model.ret_ratio = 1, args.rel_l1_thresh, args.ret_ratio
    clear_history(model)


def clear_history(model):
    model.accumulated_rel_l1_distance = [0.0, 0.0]
    model.residual_cache = [None, None]
    model.probe_residual_cache = [None, None]
    model.residual_window = [[], []]
    model.probe_residual_window = [[], []]
    model.previous_internal_states = [None, None]
    model.previous_input = [None, None]
    model.previous_output = [None, None]
    model.resume_flag = [False, False]


def compose_forward(native_forward, block_call, *, verify=True):
    """Replace ONLY the native full-block loop; preserve Wan2.2 prefix/suffix."""
    native_forward = getattr(native_forward, "__func__", native_forward)
    if verify:
        path = Path(inspect.getfile(native_forward))
        if hashlib.sha256(path.read_bytes()).hexdigest() != NATIVE_SHA256:
            raise ValueError("compose_forward requires the pinned native Wan2.2 model.py")
    fn = ast.parse(textwrap.dedent(inspect.getsource(native_forward))).body[0]
    fn.decorator_list = []
    matches = [i for i, n in enumerate(fn.body) if isinstance(n, ast.For)
               and ast.unparse(n) == "for block in self.blocks:\n    x = block(x, **kwargs)"]
    if len(matches) != 1:
        raise ValueError("cannot identify unique native WanModel full-block loop")
    fn.body[matches[0]:matches[0]+1] = ast.parse("x = _dicache_block_call(self, x, kwargs)").body
    env = dict(native_forward.__globals__, _dicache_block_call=block_call)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 inspect.getfile(native_forward), "exec"), env)
    return env[fn.name]


def scalar(value):
    if value is None:
        return None
    result = float(value)
    # Do not alter the official numerical computation on NaN/Inf. Diagnostics
    # are JSON-safe and explicitly mark the exceptional value.
    return result if math.isfinite(result) else str(result)


@contextmanager
def official_forward(models, *, core, args, records=None, verify_native=True, pipeline=None):
    """Global retention + one compulsory initialization pair per new expert.

    Model calls retain the native sampler order. Each expert owns two CFG
    states. On its first pair only, cnt=0/1 selects the official initialization
    path; all later calls use the global counter and unchanged global retention.
    No low-stage percentage warmup, final-step full, epsilon or memory rewrite
    is introduced. Release completed experts before their weights are offloaded.
    """
    models = list(models)
    if len(models) not in (1, 2) or len({id(m) for m in models}) != len(models):
        raise ValueError("provide one model for parity or high/low models in sampler order")
    if any(getattr(m, "_dicache_active", False) for m in models):
        raise RuntimeError("nested/concurrent DiCache scopes on the same model are unsupported")
    for model in models:
        if len(model.blocks) <= args.probe_depth:
            raise ValueError("DiCache requires more blocks than the official probe depth")
    names = ("forward",) + STATE_NAMES
    saved = [{k: m.__dict__.get(k, MISSING) for k in names} for m in models]
    current = {"count": 0, "expert": 0, "active": None}
    local_counts = [0] * len(models)
    handles = []
    stages = ("high", "low") if len(models) == 2 else ("high",)
    saved_prepare = MISSING

    def block_call(model, x, kwargs):
        row = current["active"]
        branch = row["call_index"] % 2
        # Capture diagnostics after executing the untouched upstream fragment.
        before = scalar(model.accumulated_rel_l1_distance[branch])
        x, info = core.blocks(model, x, kwargs)
        row.update(action="reuse" if info["skip"] else "recompute",
                   accumulated_error_before=before,
                   accumulated_error_after=scalar(model.accumulated_rel_l1_distance[branch]),
                   probe_relative_l1=scalar(info["delta_y"]), gamma=scalar(info["gamma"]),
                   residual_window_length=len(model.residual_window[branch]),
                   block_output_dtype=str(x.dtype))
        return x

    try:
        if pipeline is not None:
            saved_prepare = pipeline.__dict__.get("_prepare_model_for_timestep", MISSING)
            native_prepare = pipeline._prepare_model_for_timestep
            def prepare(t, boundary, offload_model):
                if t.item() < boundary and current["expert"] == 0:
                    if current["count"] % 2:
                        raise RuntimeError("expert transfer between CFG branches")
                    clear_history(models[0])
                return native_prepare(t, boundary, offload_model)
            pipeline._prepare_model_for_timestep = prepare
        for expert, model in enumerate(models):
            model._dicache_active = True
            initialize(model, args)
            native = compose_forward(model.forward, block_call, verify=verify_native)
            def count_block(_module, _inputs):
                if current["active"] is not None:
                    current["active"]["blocks_executed"] += 1
            for block in model.blocks:
                handles.append(block.register_forward_pre_hook(count_block))

            def forward(self, *inputs, _expert=expert, _native=native, **kwargs):
                i = current["count"]
                if i >= args.sample_steps * 2:
                    raise RuntimeError("extra model call after the video finished")
                if _expert != current["expert"]:
                    if _expert != current["expert"] + 1 or i % 2:
                        raise RuntimeError("expert changes must follow high-to-low paired CFG order")
                    clear_history(models[current["expert"]])
                    current["expert"] = _expert
                initial = local_counts[_expert] < 2
                self.cnt = i % 2 if initial else i
                row = dict(call_index=i, step_index=i // 2, model_stage=stages[_expert],
                           cfg_branch=("cond", "uncond")[i % 2],
                           expert_call_index=local_counts[_expert], official_counter=self.cnt,
                           expert_initialization=initial, probe_depth=1,
                           global_retention=(i < int(args.sample_steps * 2 * args.ret_ratio)),
                           blocks_executed=0)
                current["active"] = row
                try:
                    result = _native(self, *inputs, **kwargs)
                finally:
                    current["active"] = None
                expected = self.probe_depth if row["action"] == "reuse" else len(self.blocks)
                if row["blocks_executed"] != expected:
                    raise RuntimeError("official action disagrees with actual block execution")
                if records is not None:
                    records.append(row)
                local_counts[_expert] += 1
                current["count"] += 1
                # Like the original final uncond reset: release before VAE decode.
                if current["count"] == args.sample_steps * 2:
                    for item in models:
                        clear_history(item)
                return result
            model.forward = MethodType(forward, model)
        yield
    finally:
        if pipeline is not None:
            if saved_prepare is MISSING:
                pipeline.__dict__.pop("_prepare_model_for_timestep", None)
            else:
                pipeline._prepare_model_for_timestep = saved_prepare
        for handle in handles:
            handle.remove()
        for model, state in zip(models, saved):
            for name, original in state.items():
                if original is MISSING:
                    model.__dict__.pop(name, None)
                else:
                    setattr(model, name, original)
            model.__dict__.pop("_dicache_active", None)
