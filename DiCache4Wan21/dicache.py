"""Execute the byte-locked official DiCache block algorithm without rewrites."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parent / 'vendor/run_wan_dicache.py'
SOURCE_SHA256 = '49edfa4033d1af2983a6db07354d87b142cacdf232566bb37a2c2b7b7853ed04'
BRANCHES = ('cond', 'uncond')


def load_official():
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('official DiCache source SHA256 mismatch')
    forward = next(n for n in ast.parse(data).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'dicache_forward')
    start = next(i for i,n in enumerate(forward.body) if ast.unparse(n) == 'skip_forward = False')
    stop = next(i for i,n in enumerate(forward.body) if ast.unparse(n) == 'x = self.head(x, e)')
    algorithm = copy.deepcopy(forward.body[start:stop])
    fn = ast.parse('def blocks(self, x, kwargs):\n    pass').body[0]
    fn.body = algorithm + ast.parse(
        "return x, skip_forward, locals().get('delta_y'), locals().get('gamma')"
    ).body
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(SOURCE), 'exec'), namespace)
    return namespace['blocks']


@dataclass(frozen=True)
class DiCacheConfig:
    threshold: float
    probe_depth: int = 1
    retention_ratio: float = .2
    dcta: bool = True
    trace_path: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError('threshold must be finite and nonnegative')
        if self.probe_depth != 1 or not self.dcta:
            raise ValueError('official protocol requires probe1 and DCTA')
        if not math.isfinite(self.retention_ratio) or not .02 <= self.retention_ratio < 1:
            raise ValueError('retention must initialize both CFG branches: .02 <= ratio < 1')


def diagnostic(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else str(value)


class DiCacheController:
    def __init__(self, config: DiCacheConfig):
        self.config = config
        self.core = load_official()
        self.reset()

    def reset(self):
        self.decisions = []
        self.state = SimpleNamespace(
            cnt=0, num_steps=100, probe_depth=1,
            rel_l1_thresh=self.config.threshold, ret_ratio=self.config.retention_ratio,
            accumulated_rel_l1_distance=[0.,0.], residual_cache=[None,None],
            probe_residual_cache=[None,None], residual_window=[[],[]],
            probe_residual_window=[[],[]], previous_internal_states=[None,None],
            previous_input=[None,None], previous_output=[None,None], resume_flag=[False,False],
        )

    def execute(self, *, blocks, x, kwargs, branch, step_index, num_steps):
        i = len(self.decisions)
        if num_steps != 50 or i >= 100 or (step_index,branch) != (i//2,BRANCHES[i%2]):
            raise ValueError('expected 50 steps with alternating cond/uncond calls; reset per video')
        if len(blocks) <= 1:
            raise ValueError('DiCache requires at least two blocks')
        state = self.state
        state.blocks, state.cnt = blocks, i
        # Conversion is for trace only; never feed it into the official math.
        before = diagnostic(state.accumulated_rel_l1_distance[i%2])
        output, skip, delta, gamma = self.core(state, x, kwargs)
        self.decisions.append(dict(
            call_index=i, step_index=step_index, branch=branch,
            action='probe_reuse' if skip else 'full_compute',
            probe_blocks_executed=1, deep_blocks_executed=0 if skip else len(blocks)-1,
            probe_relative_change=diagnostic(delta), dcta_gamma=diagnostic(gamma),
            accumulated_error_before=before,
            accumulated_error_after_gate=diagnostic(state.accumulated_rel_l1_distance[i%2]),
            residual_window_length=len(state.residual_window[i%2]),
            block_output_dtype=str(output.dtype), execution='probe_reuse' if skip else 'full_compute',
        ))
        state.cnt = i+1
        if state.cnt == 100:
            # Official end-of-video cleanup before VAE; keep only scalar trace.
            decisions = self.decisions
            self.reset()
            self.decisions = decisions
        return output

    def summary(self):
        return dict(schema='dicache4wan21_official_trace_v2', threshold=self.config.threshold,
                    probe_depth=1, retention_ratio=self.config.retention_ratio, dcta=True,
                    source_sha256=SOURCE_SHA256, total_steps=50, total_branch_calls=len(self.decisions),
                    full_compute=sum(r['action']=='full_compute' for r in self.decisions),
                    probe_reuse=sum(r['action']=='probe_reuse' for r in self.decisions),
                    decisions=self.decisions)

    def write_trace(self, extra=None):
        if not self.config.trace_path:
            return None
        payload = self.summary()
        if extra:
            if payload.keys() & extra.keys():
                raise ValueError('trace fields overlap')
            payload.update(extra)
        path = Path(self.config.trace_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
        temp.replace(path)
        return path
