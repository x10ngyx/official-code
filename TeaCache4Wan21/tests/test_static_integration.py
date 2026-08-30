from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
import types
import unittest
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_batch_module():
    path = PROJECT_DIR / "experiments" / "vbench200_t2v" / "generate_vbench200.py"
    spec = importlib.util.spec_from_file_location("vbench200_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_entrypoint_module():
    path = PROJECT_DIR / "generate.py"
    spec = importlib.util.spec_from_file_location("teacache_wan21_entrypoint", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StaticIntegrationTests(unittest.TestCase):
    def test_official_source_hash(self):
        source = PROJECT_DIR / "upstream" / "teacache_generate.py"
        self.assertEqual(
            hashlib.sha256(source.read_bytes()).hexdigest(),
            "97af76136337869152f3d6fe9e049cadc2c480740c492749fcd5efa80d9bf7ee",
        )

    def test_baseline_calls_original_before_lazy_teacache_import(self):
        source = (PROJECT_DIR / "generate.py").read_text(encoding="utf-8")
        baseline = source.index(
            "if not wrapper_args.enable_teacache and wrapper_args.timing_json is None:"
        )
        direct_call = source.index("original.generate(args)", baseline)
        lazy_import = source.index("from teacache import patch_pipeline_construction")
        self.assertLess(baseline, direct_call)
        self.assertLess(direct_call, lazy_import)

    def test_installer_imports_official_functions(self):
        tree = ast.parse((PROJECT_DIR / "teacache.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "upstream.teacache_generate":
                imported.update(alias.name for alias in node.names)
        self.assertEqual(
            imported,
            {"t2v_generate", "i2v_generate", "teacache_forward"},
        )

    def test_batch_uses_one_entrypoint_for_both_paths(self):
        module = load_batch_module()
        common = dict(
            wan21_root=Path("/source/Wan2.1"),
            ckpt_dir=Path("/models/Wan2.1-T2V-1.3B"),
            task="t2v-1.3B",
            size="832*480",
            frame_num=81,
            sample_steps=50,
            sample_shift=5.0,
            guide_scale=5.0,
            sample_solver="unipc",
            offload_model=False,
            t5_cpu=False,
            use_ret_steps=False,
        )
        sample = {"sample_id": "vbench200_001", "prompt_en": "test prompt"}
        baseline = Namespace(
            **common,
            implementation="wan21",
            teacache_thresh=None,
        )
        candidate = Namespace(
            **common,
            implementation="teacache",
            teacache_thresh=0.08,
        )
        baseline_command = module.build_command(
            baseline, sample, 42, Path("/results/reference.mp4")
        )
        candidate_command = module.build_command(
            candidate,
            sample,
            42,
            Path("/results/candidate.mp4"),
            Path("/results/candidate.timing.json"),
        )
        self.assertEqual(baseline_command[1], candidate_command[1])
        self.assertNotIn("--enable_teacache", baseline_command)
        self.assertIn("--enable_teacache", candidate_command)
        timing_index = candidate_command.index("--timing_json")
        self.assertEqual(
            candidate_command[timing_index + 1], "/results/candidate.timing.json"
        )

    def test_runtime_baseline_does_not_import_teacache(self):
        module = load_entrypoint_module()
        calls = []
        teacache_before = sys.modules.get("teacache")
        original = types.SimpleNamespace(
            wan=object(),
            _parse_args=lambda: Namespace(
                task="t2v-1.3B",
                ckpt_dir="/models/Wan2.1-T2V-1.3B",
                size="832*480",
                frame_num=81,
                sample_steps=50,
                sample_solver="unipc",
                sample_shift=5.0,
                sample_guide_scale=5.0,
                base_seed=42,
                offload_model=False,
                t5_cpu=False,
                t5_fsdp=False,
                dit_fsdp=False,
                ulysses_size=1,
                ring_size=1,
                use_prompt_extend=False,
            ),
            generate=lambda args: calls.append(args),
        )
        with mock.patch.object(module, "resolve_wan21_root", return_value=Path("/source")), \
             mock.patch.object(module, "load_original_generate", return_value=original), \
             mock.patch.object(sys, "argv", ["generate.py", "--task", "t2v-1.3B"]):
            module.main()
        self.assertEqual(len(calls), 1)
        self.assertIs(sys.modules.get("teacache"), teacache_before)

    def test_runtime_teacache_is_explicit_and_wraps_original(self):
        module = load_entrypoint_module()
        calls = []
        patch_calls = []
        original = types.SimpleNamespace(
            wan=object(),
            _parse_args=lambda: Namespace(
                task="t2v-1.3B",
                ckpt_dir="/models/Wan2.1-T2V-1.3B",
                size="832*480",
                frame_num=81,
                sample_steps=50,
                sample_solver="unipc",
                sample_shift=5.0,
                sample_guide_scale=5.0,
                base_seed=42,
                offload_model=False,
                t5_cpu=False,
                t5_fsdp=False,
                dit_fsdp=False,
                ulysses_size=1,
                ring_size=1,
                use_prompt_extend=False,
            ),
            generate=lambda args: calls.append(args),
        )

        @contextmanager
        def fake_patch(*args, **kwargs):
            patch_calls.append((args, kwargs))
            yield

        fake_teacache = types.ModuleType("teacache")
        fake_teacache.patch_pipeline_construction = fake_patch
        argv = [
            "generate.py",
            "--enable_teacache",
            "--teacache_thresh",
            "0.08",
            "--task",
            "t2v-1.3B",
        ]
        with mock.patch.object(module, "resolve_wan21_root", return_value=Path("/source")), \
             mock.patch.object(module, "load_original_generate", return_value=original), \
             mock.patch.dict(sys.modules, {"teacache": fake_teacache}), \
             mock.patch.object(sys, "argv", argv):
            module.main()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(patch_calls), 1)
        self.assertEqual(patch_calls[0][1]["threshold"], 0.08)

    def test_threshold_zero_is_an_explicit_valid_diagnostic(self):
        module = load_entrypoint_module()
        wrapper_args, wan_args = module.parse_wrapper_args(
            ["--enable_teacache", "--teacache_thresh", "0", "--task", "t2v-1.3B"]
        )
        self.assertEqual(wrapper_args.teacache_thresh, 0.0)
        self.assertEqual(wan_args, ["--task", "t2v-1.3B"])

    def test_timing_output_is_a_wrapper_only_option(self):
        module = load_entrypoint_module()
        wrapper_args, wan_args = module.parse_wrapper_args(
            [
                "--timing_json",
                "/results/timing.json",
                "--task",
                "t2v-1.3B",
            ]
        )
        self.assertEqual(wrapper_args.timing_json, Path("/results/timing.json"))
        self.assertEqual(wan_args, ["--task", "t2v-1.3B"])

    def test_threshold_zero_runs_official_forward_without_residual_reuse(self):
        fake_upstream = types.ModuleType("upstream.teacache_generate")
        fake_upstream.t2v_generate = lambda *args, **kwargs: None
        fake_upstream.i2v_generate = lambda *args, **kwargs: None
        fake_upstream.teacache_forward = lambda *args, **kwargs: None
        spec = importlib.util.spec_from_file_location(
            "teacache_zero_test", PROJECT_DIR / "teacache.py"
        )
        assert spec is not None and spec.loader is not None
        teacache = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules, {"upstream.teacache_generate": fake_upstream}
        ):
            spec.loader.exec_module(teacache)

        class FakeModel:
            pass

        class FakeWanT2V:
            def __init__(self):
                self.model = FakeModel()

        wan_module = types.SimpleNamespace(WanT2V=FakeWanT2V)
        original_init = FakeWanT2V.__init__
        with teacache.patch_pipeline_construction(
            wan_module,
            task="t2v-1.3B",
            checkpoint_dir="/models/Wan2.1-T2V-1.3B",
            sample_steps=4,
            threshold=0.0,
            use_ret_steps=False,
        ):
            self.assertIsNot(FakeWanT2V.__init__, original_init)
            pipeline = FakeWanT2V()
            self.assertEqual(pipeline.model.teacache_configured_thresh, 0.0)
            self.assertEqual(pipeline.model.teacache_thresh, float("-inf"))
            self.assertIs(
                pipeline.model.forward.__func__, fake_upstream.teacache_forward
            )
        self.assertIs(FakeWanT2V.__init__, original_init)

    def test_negative_threshold_is_rejected(self):
        module = load_entrypoint_module()
        with self.assertRaises(SystemExit):
            module.parse_wrapper_args(
                ["--enable_teacache", "--teacache_thresh", "-0.01"]
            )


if __name__ == "__main__":
    unittest.main()
