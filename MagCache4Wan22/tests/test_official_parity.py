"""Compare the adapter to a full import of the unchanged official script."""
from __future__ import annotations
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))
from official import SOURCE, load_official, method_args, official_forward, record_calls


def sinusoidal(dim, t):
    return torch.sin(t[:, None].float() / (1 + torch.arange(dim).float())[None])


def import_complete_upstream():
    stubs = {}
    for name in ("wan", "wan.configs", "wan.distributed", "wan.distributed.util",
                 "wan.utils", "wan.utils.prompt_extend", "wan.utils.utils", "wan.modules", "wan.modules.model"):
        stubs[name] = types.ModuleType(name)
    for name in ("MAX_AREA_CONFIGS", "SIZE_CONFIGS", "SUPPORTED_SIZES", "WAN_CONFIGS"):
        setattr(stubs["wan.configs"], name, {})
    stubs["wan.distributed.util"].init_distributed_group = lambda: None
    stubs["wan.utils.prompt_extend"].DashScopePromptExpander = object
    stubs["wan.utils.prompt_extend"].QwenPromptExpander = object
    stubs["wan.utils.utils"].save_video = lambda **kwargs: None
    stubs["wan.utils.utils"].str2bool = bool
    stubs["wan.modules.model"].sinusoidal_embedding_1d = sinusoidal
    spec = importlib.util.spec_from_file_location("official_reference_full", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs), warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


class Block(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.executions = 0
    def forward(self, x, *, e, context, **kwargs):
        self.executions += 1
        return x + self.factor * (1 + .01 * e[:, :, 0] + .03 * context.mean(1, keepdim=True))


class Head(nn.Module):
    def forward(self, x, e):
        return (x + .02 * e)[..., :2]


class ToyModel(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.model_type = "t2v"
        self.dim = self.freq_dim = 4
        self.text_len = 3
        self.freqs = torch.zeros(1)
        self.patch_embedding = nn.Conv3d(2, 4, 1, bias=False)
        self.time_embedding = nn.Linear(4, 4)
        self.time_projection = nn.Linear(4, 24)
        self.text_embedding = nn.Linear(5, 4)
        self.blocks = nn.ModuleList([Block(scale), Block(scale * 2)])
        self.head = Head()
    def unpatchify(self, x, grid_sizes):
        return [u.transpose(0, 1).reshape(2, *grid.tolist()) for u, grid in zip(x, grid_sizes)]
    def forward(self, value):
        return value + 99  # Native sentinel proves that cleanup restores baseline.


def models():
    # Match official shared-class behavior without contaminating another case.
    cls = type("IsolatedWanModel", (ToyModel,), {})
    torch.manual_seed(42)
    return [cls(.2), cls(.7)]


def inputs(steps=50):
    generator = torch.Generator().manual_seed(710)
    return [([torch.randn(2, 1, 1, 2, generator=generator)],
             dict(t=torch.tensor([1000. - i * 9]),
                  context=[torch.randn(2, 5, generator=generator)], seq_len=2))
            for i in range(2 * steps)]


def run_forward(pair, values, split):
    outputs, counts = [], []
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (latent, kwargs) in enumerate(values):
            model = pair[0 if i < 2 * split else 1]
            before = sum(b.executions for b in model.blocks)
            outputs.append(model(latent, **kwargs)[0].clone())
            counts.append(sum(b.executions for b in model.blocks) - before)
    return outputs, counts


class OfficialParityTests(unittest.TestCase):
    def test_outputs_and_100_decisions_match_full_official_import(self):
        reference = import_complete_upstream()
        for threshold, k, retention, steps, split in ((.06, 2, .4, 50, 32), (.02, 1, .1, 50, 32),
                                                       (.1, 4, .2, 40, 26), (100., 100, .4, 50, 32)):
            with self.subTest(threshold=threshold, k=k, steps=steps):
                core = load_official(sinusoidal)
                args = method_args(threshold, k, retention, steps)
                direct = models()
                reference.init_magcache(direct[0], core.ratios, args, split)
                expected, expected_counts = run_forward(direct, inputs(steps), split)
                actual_models = models()
                trace = []
                with official_forward(actual_models, core=core, args=args, split_steps=split), record_calls(actual_models, trace):
                    actual, actual_counts = run_forward(actual_models, inputs(steps), split)
                self.assertEqual(actual_counts, expected_counts)
                self.assertEqual([r["blocks_executed"] for r in trace], expected_counts)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
                self.assertEqual([r["call_index"] for r in trace], list(range(2 * steps)))
                if threshold == .06:
                    self.assertEqual(actual_counts[24:26], [2, 0])  # Official CFG branches differ.
                if threshold == 100:
                    self.assertEqual(actual_counts[-2:], [0, 0])  # No invented final full step.

    def test_independent_scalar_schedule_preserves_forced_full_accumulators(self):
        core = load_official(sinusoidal)
        pair, trace = models(), []
        args = method_args(.06, 2, .4)
        padded = np.array([1., 1.] + core.ratios)
        indices = np.rint(np.linspace(0, 39, 50)).astype(int)
        ratios = np.stack([padded[::2][indices], padded[1::2][indices]], axis=-1).reshape(-1)
        accum, error, count = [1., 1.], [0., 0.], [0, 0]
        expected = []
        for call, ratio in enumerate(ratios):
            b = call % 2
            protected = call < int(64 * .4) or 64 <= call <= 64 + 36 * .4
            full = True
            if not protected:
                accum[b] *= ratio
                error[b] += abs(1 - accum[b])
                count[b] += 1
                full = not (error[b] < .06 and count[b] <= 2)
                if full:
                    accum[b], error[b], count[b] = 1., 0., 0
            expected.append(2 if full else 0)
        with official_forward(pair, core=core, args=args, split_steps=32), record_calls(pair, trace):
            _, observed = run_forward(pair, inputs(), 32)
        self.assertEqual(observed, expected)
        self.assertEqual(trace[64]["accumulated_error_before"], trace[64]["accumulated_error_after"])
        self.assertGreater(trace[64]["accumulated_error_before"], 0)

    def test_repeated_video_and_failure_restore_original_class_and_instances(self):
        pair = models()
        core = load_official(sinusoidal)
        before_class = dict(type(pair[0]).__dict__)
        before_attrs = [set(m.__dict__) for m in pair]
        outputs = []
        for _ in range(2):
            with official_forward(pair, core=core, args=method_args(), split_steps=32):
                outputs.append(run_forward(pair, inputs(), 32)[0])
            self.assertEqual(set(type(pair[0]).__dict__), set(before_class))
            self.assertEqual([set(m.__dict__) for m in pair], before_attrs)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*outputs)))
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with official_forward(pair, core=core, args=method_args(), split_steps=32):
                run_forward(pair, inputs()[:3], 32)
                raise RuntimeError("deliberate failure")
        self.assertEqual(pair[0](torch.tensor(1)).item(), 100)
        self.assertEqual(set(type(pair[0]).__dict__), set(before_class))

    def test_calibration_matches_official_measurements(self):
        reference = import_complete_upstream()
        args = method_args()
        direct, wrapped = models(), models()
        reference.init_magcache_calibration(direct[0], args)
        expected = {}
        reference.save_json = lambda name, values: expected.update({name: list(values)})
        with contextlib.redirect_stdout(io.StringIO()):
            outputs, _ = run_forward(direct, inputs(), 32)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "calibration"
            with official_forward(wrapped, core=load_official(sinusoidal), args=args, split_steps=32,
                                  calibration_dir=directory), contextlib.redirect_stdout(io.StringIO()):
                actual, counts = run_forward(wrapped, inputs(), 32)
            self.assertEqual(counts, [2] * 100)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(outputs, actual)))
            for name, values in expected.items():
                self.assertEqual(len(values), 98)
                self.assertEqual(json.loads((directory / (name + ".json")).read_text()), values)

    def test_builtin_table_and_official_split(self):
        core = load_official(sinusoidal)
        self.assertEqual(len(core.ratios), 78)
        self.assertEqual(int((core.get_timesteps(shift=12, num_inference_steps=50) >= 875).sum()), 32)

    def test_invalid_parameters_and_nested_session_rejected(self):
        for kwargs in ({"threshold": float("nan")}, {"k": -1}, {"retention_ratio": 0}, {"retention_ratio": 1.1}):
            with self.assertRaises(ValueError):
                method_args(**kwargs)
        pair, core = models(), load_official(sinusoidal)
        with official_forward(pair, core=core, args=method_args(), split_steps=32):
            with self.assertRaises(RuntimeError):
                with official_forward(pair, core=core, args=method_args(), split_steps=32):
                    pass

    def test_timing_observer_preserves_official_outputs_and_trace(self):
        from inference_timing import _PipelineProfiler
        from test_inference_timing import FakeTextEncoder, FakeVAE
        class Pipeline:
            def __init__(self):
                self.high_noise_model, self.low_noise_model = models()
                self.device = torch.device("cpu")
                self.text_encoder, self.vae = FakeTextEncoder(), FakeVAE()
            def generate(self):
                self.text_encoder(torch.zeros(1))
                self.text_encoder(torch.zeros(1))
                result, _ = run_forward([self.high_noise_model, self.low_noise_model], inputs(), 32)
                return self.vae.decode(torch.stack(result))
        pipeline = Pipeline()
        pair = [pipeline.high_noise_model, pipeline.low_noise_model]
        core, args, trace = load_official(sinusoidal), method_args(), []
        with official_forward(pair, core=core, args=args, split_steps=32):
            expected = pipeline.generate()
        with tempfile.TemporaryDirectory() as temp:
            timing_path = Path(temp) / "timing.json"
            with official_forward(pair, core=core, args=args, split_steps=32), record_calls(pair, trace):
                profiler = _PipelineProfiler(pipeline, init_wall_seconds=0, output_path=timing_path,
                                             implementation="magcache")
                profiler.install()
                observed = pipeline.generate()
            timing = json.loads(timing_path.read_text())
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(len(timing["calls"]), 100)
        self.assertEqual([r["blocks_executed"] for r in timing["calls"]], [r["blocks_executed"] for r in trace])
        self.assertEqual(timing["component_latency"]["t5"]["call_count"], 2)
        self.assertEqual(timing["component_latency"]["vae_decode"]["call_count"], 1)

    def test_real_pinned_wan_blocks_full_forward_matches_native(self):
        """Real small WanModel on CPU; replace only the CUDA attention kernel."""
        import hashlib
        path = PROJECT / "build/Wan2.2-42bf4cf/wan/modules/model.py"
        if not path.exists():
            self.skipTest("prepare the pinned Wan2.2 checkout for the real-block CPU check")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432")
        package = types.ModuleType("magcache_test_native")
        package.__path__ = []
        attention = types.ModuleType("magcache_test_native.attention")
        def sdpa(q, k, v, **kwargs):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        attention.flash_attention = sdpa
        spec = importlib.util.spec_from_file_location("magcache_test_native.model", path)
        native = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"magcache_test_native": package,
                                     "magcache_test_native.attention": attention,
                                     "magcache_test_native.model": native}):
            spec.loader.exec_module(native)
            torch.manual_seed(80)
            pair = [native.WanModel(model_type="t2v", patch_size=(1, 1, 1), text_len=3,
                                    in_dim=2, dim=12, ffn_dim=24, freq_dim=4, text_dim=5,
                                    out_dim=2, num_heads=2, num_layers=2) for _ in range(2)]
        for model in pair:
            nn.init.normal_(model.head.head.weight, std=.1)
            for block in model.blocks:
                block.executions = 0
                def track(module, _args):
                    module.executions += 1
                block.register_forward_pre_hook(track)
        core = load_official(native.sinusoidal_embedding_1d)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected, _ = run_forward(pair, inputs(), 32)
            with official_forward(pair, core=core, args=method_args(k=0), split_steps=32):
                actual, _ = run_forward(pair, inputs(), 32)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))

if __name__ == "__main__":
    unittest.main()
