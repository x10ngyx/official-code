"""Independent full official-script oracle, plus native Wan2.2 block checks."""
from __future__ import annotations
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
# Load external dependencies before temporarily stubbing only Wan modules;
# unittest.patch.dict otherwise removes newly imported dispatcher modules.
import torch._dynamo
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
import matplotlib.pyplot

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))
from official import SOURCE, STATE_NAMES, load_official, method_args, official_forward


def load_native():
    path = PROJECT / "build/Wan2.2-42bf4cf/wan/modules/model.py"
    if not path.exists():
        raise RuntimeError("prepare pinned Wan2.2 before parity tests")
    package = types.ModuleType("dicache_test_native")
    package.__path__ = []
    attention = types.ModuleType("dicache_test_native.attention")
    def sdpa(q, k, v, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    attention.flash_attention = sdpa
    spec = importlib.util.spec_from_file_location("dicache_test_native.model", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"dicache_test_native": package,
                                 "dicache_test_native.attention": attention,
                                 "dicache_test_native.model": module}):
        spec.loader.exec_module(module)
    return module


NATIVE = load_native()


def import_full_official():
    stubs = {}
    for name in ("wan", "wan.configs", "wan.utils", "wan.utils.prompt_extend", "wan.utils.utils", "wan.modules", "wan.modules.model"):
        stubs[name] = types.ModuleType(name)
    for name in ("MAX_AREA_CONFIGS", "SIZE_CONFIGS", "SUPPORTED_SIZES", "WAN_CONFIGS"):
        setattr(stubs["wan.configs"], name, {})
    for name in ("DashScopePromptExpander", "QwenPromptExpander"):
        setattr(stubs["wan.utils.prompt_extend"], name, object)
    for name in ("cache_image", "cache_video", "str2bool"):
        setattr(stubs["wan.utils.utils"], name, lambda *a, **k: None)
    stubs["wan.modules.model"].sinusoidal_embedding_1d = NATIVE.sinusoidal_embedding_1d
    spec = importlib.util.spec_from_file_location("dicache_full_official_reference", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs), warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


class Projection(nn.Module):
    def forward(self, x):
        return x.repeat(*([1] * (x.ndim - 1)), 6)


class PatchEmbedding(nn.Conv3d):
    def __init__(self, bf16):
        super().__init__(2, 4, 1, bias=False)
        self.bf16 = bf16
    def forward(self, x):
        x = super().forward(x)
        return x.bfloat16() if self.bf16 else x


class Block(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.executions = 0
    def forward(self, x, *, e, context, **kwargs):
        self.executions += 1
        if e.ndim == 3:
            e = e.unsqueeze(1)
        return x.float() + self.scale * torch.tanh(x.float() + .03 * e[:, :, 0] + .04 * context.mean(1, keepdim=True))


class Head(nn.Module):
    def forward(self, x, e):
        if e.ndim == 2:
            e = e.unsqueeze(1)
        return (x.float() + .02 * e)[..., :2]


class ToyModel(nn.Module):
    forward = NATIVE.WanModel.forward
    def __init__(self, scale=.2, bf16=False):
        super().__init__()
        self.model_type = "t2v"
        self.dim = self.freq_dim = 4
        self.text_len = 3
        self.freqs = torch.zeros(1)
        self.patch_embedding = PatchEmbedding(bf16)
        self.time_embedding = nn.Identity()
        self.time_projection = Projection()
        self.text_embedding = nn.Linear(5, 4)
        self.blocks = nn.ModuleList([Block(scale*(j+1)) for j in range(4)])
        self.head = Head()
    def unpatchify(self, x, grid_sizes):
        return [u.transpose(0, 1).reshape(2, *grid.tolist()) for u, grid in zip(x, grid_sizes)]


def models(bf16=False):
    torch.manual_seed(42)
    return [ToyModel(.2, bf16), ToyModel(.8, bf16)]


def inputs(steps=50):
    rng = torch.Generator().manual_seed(710)
    latent = torch.randn(2, 1, 1, 3, generator=rng) + 2
    contexts = [torch.randn(2, 5, generator=rng) for _ in range(2)]
    return [([latent + (i//2)*(.004 if i%2 == 0 else .05)],
             dict(t=torch.tensor([1000. - (i//2)*.1]), context=[contexts[i%2]], seq_len=3))
            for i in range(steps*2)]


def init_reference(model, options):
    # The unchanged CLI's assignments, instantiated separately for each expert.
    model.cnt = 0
    model.probe_depth = 1
    model.num_steps = options.sample_steps*2
    model.rel_l1_thresh = options.rel_l1_thresh
    model.ret_ratio = options.ret_ratio
    model.accumulated_rel_l1_distance = [0., 0.]
    model.residual_cache = [None, None]
    model.probe_residual_cache = [None, None]
    model.residual_window = [[], []]
    model.probe_residual_window = [[], []]
    model.previous_internal_states = [None, None]
    model.previous_input = [None, None]
    model.previous_output = [None, None]
    model.resume_flag = [False, False]


def run(pair, values, split=32):
    outputs, counts = [], []
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (x, kw) in enumerate(values):
            model = pair[0 if i < split*2 else 1]
            before = sum(b.executions for b in model.blocks)
            outputs.append(model(x, **kw)[0].clone())
            counts.append(sum(b.executions for b in model.blocks)-before)
    return outputs, counts


class OfficialParityTests(unittest.TestCase):
    def test_strict_threshold_and_original_zero_denominator(self):
        class IdentityBlock(nn.Module):
            def forward(self, x, **kwargs):
                return x.float()
        core = load_official()
        x = torch.full((1, 2, 4), 1.125)
        for threshold, skip in ((.125, False), (.125001, True)):
            m = models()[0]
            init_reference(m, method_args(threshold))
            m.blocks = nn.ModuleList([IdentityBlock() for _ in range(4)])
            m.cnt = 20
            m.previous_input[0] = torch.ones_like(x)
            m.previous_internal_states[0] = torch.ones_like(x)
            m.residual_cache[0] = torch.full_like(x, .25)
            m.residual_window[0] = [m.residual_cache[0]]
            m.probe_residual_window[0] = [torch.zeros_like(x)]
            value, info = core.blocks(m, x.clone(), {})
            self.assertEqual(info["skip"], skip)
            self.assertTrue(torch.equal(value, x+.25 if skip else x))
        m = models()[0]
        init_reference(m, method_args())
        m.blocks = nn.ModuleList([IdentityBlock() for _ in range(4)])
        m.cnt = 20
        m.previous_input[0] = m.previous_internal_states[0] = x.clone()
        m.residual_cache[0] = torch.ones_like(x)
        m.residual_window[0] = [torch.zeros_like(x), torch.ones_like(x)]
        m.probe_residual_window[0] = [torch.zeros_like(x), torch.zeros_like(x)]
        value, info = core.blocks(m, x.clone(), {})
        self.assertTrue(info["skip"])
        self.assertTrue(torch.isnan(info["gamma"]).item())
        self.assertTrue(torch.isnan(value).all().item())

    def test_full_official_oracle_outputs_and_decisions(self):
        reference = import_full_official()
        for bf16 in (False, True):
            for threshold, retention in ((0., .2), (.08, .2), (.2, .2), (100., .2), (100., .02), (.08, .25)):
                with self.subTest(bf16=bf16, threshold=threshold, retention=retention):
                    opts = method_args(threshold, retention)
                    pair = models(bf16)
                    expected, expected_counts = [], []
                    for m in pair:
                        init_reference(m, opts)
                    with torch.no_grad(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for i, (x, kw) in enumerate(inputs()):
                            m = pair[0 if i < 64 else 1]
                            # Only the new expert's first pair needs mandatory
                            # initialization; all subsequent retention is global.
                            m.cnt = i%2 if i in (64, 65) else i
                            before = sum(b.executions for b in m.blocks)
                            expected.append(reference.dicache_forward(m, x, **kw)[0].clone())
                            expected_counts.append(sum(b.executions for b in m.blocks)-before)
                    actual_pair, trace = models(bf16), []
                    with official_forward(actual_pair, core=load_official(), args=opts, records=trace):
                        actual, counts = run(actual_pair, inputs())
                        self.assertTrue(all(m.residual_cache == [None, None] for m in actual_pair))
                    self.assertEqual(counts, expected_counts)
                    self.assertEqual([r["blocks_executed"] for r in trace], expected_counts)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
                    if threshold == 100 and retention == .2:
                        self.assertEqual(counts[:20], [4]*20)
                        self.assertEqual(counts[64:68], [4, 4, 1, 1])
                        self.assertEqual(counts[-2:], [1, 1])
                        self.assertEqual(trace[20]["residual_window_length"], 3)
                        self.assertIsNone(trace[66]["gamma"])
                        if bf16:
                            self.assertEqual(trace[20]["block_output_dtype"], "torch.bfloat16")
                            self.assertEqual(trace[66]["block_output_dtype"], "torch.float32")

    def test_single_expert_matches_unadapted_official_lifecycle(self):
        reference = import_full_official()
        direct = models()[0]
        opts = method_args()
        init_reference(direct, opts)
        expected = []
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, kw in inputs():
                expected.append(reference.dicache_forward(direct, x, **kw)[0])
        actual = models()[0]
        trace = []
        with official_forward([actual], core=load_official(), args=opts, records=trace):
            observed, counts = run([actual], inputs(), split=50)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(observed, expected)))
        self.assertTrue(any(counts[i] != counts[i+1] for i in range(0, 100, 2)))

    def test_repeated_videos_and_exception_cleanup(self):
        pair = models()
        originals = [m.forward for m in pair]
        keys = [set(m.__dict__) for m in pair]
        class_keys = set(ToyModel.__dict__)
        results = []
        for _ in range(2):
            with official_forward(pair, core=load_official(), args=method_args()):
                results.append(run(pair, inputs())[0])
            self.assertEqual([set(m.__dict__) for m in pair], keys)
            self.assertEqual([m.forward for m in pair], originals)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*results)))
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with official_forward(pair, core=load_official(), args=method_args()):
                run(pair, inputs()[:24])
                raise RuntimeError("deliberate failure")
        self.assertEqual([set(m.__dict__) for m in pair], keys)
        self.assertEqual(set(ToyModel.__dict__), class_keys)
        self.assertTrue(all(not b._forward_pre_hooks for m in pair for b in m.blocks))

    def test_invalid_parameters_and_nested_scope(self):
        for kw in ({"threshold": float("nan")}, {"threshold": -1}, {"retention_ratio": 0}, {"retention_ratio": .001}, {"retention_ratio": 1.1}):
            with self.assertRaises(ValueError):
                method_args(**kw)
        pair = models()
        with official_forward(pair, core=load_official(), args=method_args()):
            with self.assertRaises(RuntimeError):
                with official_forward(pair, core=load_official(), args=method_args()):
                    pass

    def test_timing_probe_and_state_cleanup_before_transfer_and_vae(self):
        from inference_timing import _PipelineProfiler
        from test_inference_timing import FakeTextEncoder, FakeVAE
        class Pipeline:
            def __init__(self):
                self.high_noise_model, self.low_noise_model = models()
                self.device = torch.device("cpu")
                self.text_encoder, self.vae = FakeTextEncoder(), FakeVAE()
            def _prepare_model_for_timestep(self, t, boundary, offload_model):
                if t.item() < boundary:
                    assert self.high_noise_model.residual_cache == [None, None]
                return self.high_noise_model if t.item() >= boundary else self.low_noise_model
            def generate(self):
                self.text_encoder(torch.zeros(1)); self.text_encoder(torch.zeros(1))
                out = []
                for i, (x, kw) in enumerate(inputs()):
                    model = self._prepare_model_for_timestep(torch.tensor(900 if i < 64 else 100), 875, True)
                    out.append(model(x, **kw)[0])
                assert self.high_noise_model.residual_cache == [None, None]
                assert self.low_noise_model.residual_cache == [None, None]
                return self.vae.decode(torch.stack(out))
        pipeline = Pipeline()
        pair = [pipeline.high_noise_model, pipeline.low_noise_model]
        trace = []
        with tempfile.TemporaryDirectory() as temp, torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = Path(temp)/"timing.json"
            with official_forward(pair, core=load_official(), args=method_args(), records=trace, pipeline=pipeline):
                profiler = _PipelineProfiler(pipeline, init_wall_seconds=0, output_path=path, implementation="dicache")
                profiler.install()
                pipeline.generate()
            timing = json.loads(path.read_text())
        self.assertEqual([r["blocks_executed"] for r in timing["calls"]], [r["blocks_executed"] for r in trace])
        self.assertEqual(timing["reuse_forward_calls"], sum(r["action"] == "reuse" for r in trace))
        self.assertEqual(timing["component_latency"]["t5"]["call_count"], 2)
        self.assertEqual(timing["component_latency"]["vae_decode"]["call_count"], 1)
        self.assertNotIn("_prepare_model_for_timestep", pipeline.__dict__)

    def test_real_wan22_full_path_exact_native_outputs(self):
        torch.manual_seed(80)
        pair = [NATIVE.WanModel(model_type="t2v", patch_size=(1, 1, 1), text_len=3,
                               in_dim=2, dim=12, ffn_dim=24, freq_dim=4, text_dim=5,
                               out_dim=2, num_heads=2, num_layers=4) for _ in range(2)]
        for m in pair:
            nn.init.normal_(m.head.head.weight, std=.1)
            for block in m.blocks:
                block.executions = 0
                block.register_forward_pre_hook(lambda m, _: setattr(m, "executions", m.executions+1))
        expected, _ = run(pair, inputs())
        with official_forward(pair, core=load_official(), args=method_args(threshold=0)):
            observed, counts = run(pair, inputs())
        self.assertEqual(counts, [4]*100)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, observed)))


if __name__ == "__main__":
    unittest.main()
