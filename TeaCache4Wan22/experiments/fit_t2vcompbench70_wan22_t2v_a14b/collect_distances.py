#!/usr/bin/env python3
"""Collect full-compute TeaCache calibration distances for one manifest shard."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_WAN_REPO = Path(os.environ["WAN22_SOURCE"]) if os.environ.get("WAN22_SOURCE") else None
DEFAULT_CKPT_DIR = Path(os.environ["WAN22_CKPT"]) if os.environ.get("WAN22_CKPT") else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, default=DEFAULT_WAN_REPO)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--skip-residual-diagnostic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def log_context(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    old_handlers = list(logger.handlers)
    old_level = logger.level
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    with path.open("a", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)):
            with contextlib.redirect_stderr(Tee(sys.stderr, stream)):
                try:
                    yield
                finally:
                    handler.flush()
                    handler.close()
                    logger.handlers = old_handlers
                    logger.setLevel(old_level)


class BlockDistanceObserver:
    """Observe e, block input Z, and post-block H without changing Wan forward."""

    STAGE_LAYOUT = {
        "high": {"offset": 0, "steps": 32},
        "low": {"offset": 32, "steps": 18},
    }

    def __init__(self, torch_module: Any, chunk_tokens: int, collect_residual: bool):
        self.torch = torch_module
        self.chunk_tokens = chunk_tokens
        self.collect_residual = collect_residual
        self.handles: list[Any] = []
        self.prompt: dict[str, Any] | None = None
        self.calls: dict[str, int] = {}
        self.previous: dict[tuple[str, str], dict[str, Any]] = {}
        self.active: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.stat_elapsed_seconds = 0.0

    def attach(self, pipeline: Any) -> None:
        for stage, model in (
            ("high", pipeline.high_noise_model),
            ("low", pipeline.low_noise_model),
        ):
            self.handles.append(
                model.register_forward_pre_hook(self._make_model_pre_hook(stage), with_kwargs=True)
            )
            self.handles.append(model.time_embedding.register_forward_hook(self._time_embedding_hook))
            self.handles.append(model.blocks[0].register_forward_pre_hook(self._first_block_pre_hook))
            self.handles.append(model.blocks[-1].register_forward_hook(self._last_block_hook))

    def close(self) -> None:
        self.clear_prompt()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start_prompt(self, prompt: dict[str, Any]) -> None:
        self.clear_prompt()
        self.prompt = dict(prompt)
        self.calls = {"high": 0, "low": 0}
        self.records = []
        self.stat_elapsed_seconds = 0.0

    def clear_prompt(self) -> None:
        self.prompt = None
        self.calls = {}
        self.previous.clear()
        self.active = None
        self.records = []
        gc.collect()

    def finish_prompt(self) -> tuple[list[dict[str, Any]], float]:
        expected_calls = {stage: layout["steps"] * 2 for stage, layout in self.STAGE_LAYOUT.items()}
        if self.calls != expected_calls:
            raise AssertionError(f"Unexpected model call counts: observed={self.calls}, expected={expected_calls}")
        if len(self.records) != 100:
            raise AssertionError(f"Expected 100 CFG branch records, observed {len(self.records)}")
        paired = [row for row in self.records if row["h_rel_l1"] is not None]
        eligible = [row for row in paired if row["fit_eligible"]]
        if len(paired) != 96 or len(eligible) != 94:
            raise AssertionError(
                f"Expected 96 within-stage transitions and 94 gate-eligible transitions; got {len(paired)} and {len(eligible)}"
            )
        records = self.records
        stat_elapsed = self.stat_elapsed_seconds
        self.prompt = None
        self.previous.clear()
        self.active = None
        self.records = []
        return records, stat_elapsed

    def _make_model_pre_hook(self, stage: str):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if self.prompt is None:
                raise RuntimeError("Observer received a model call outside an active prompt")
            if self.active is not None:
                raise RuntimeError("Observer model calls overlapped unexpectedly")
            call_index = self.calls[stage]
            layout = self.STAGE_LAYOUT[stage]
            if call_index >= layout["steps"] * 2:
                raise AssertionError(f"Too many {stage} model calls")
            stage_step = call_index // 2
            branch = "cond" if call_index % 2 == 0 else "uncond"
            global_step = layout["offset"] + stage_step
            timestep_tensor = kwargs.get("t")
            if timestep_tensor is None:
                raise AssertionError("WanModel forward did not receive keyword t")
            timestep = float(timestep_tensor.detach().reshape(-1)[0].item())
            self.active = {
                "stage": stage,
                "branch": branch,
                "stage_step": stage_step,
                "global_step": global_step,
                "timestep": timestep,
            }
            self.calls[stage] += 1

        return hook

    def _time_embedding_hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if self.active is None:
            raise RuntimeError("time_embedding hook has no active model call")
        if output.ndim != 3 or output.shape[0] != 1:
            raise AssertionError(f"Unexpected time embedding shape: {tuple(output.shape)}")
        # clone() is essential: a view would retain the full token-wise FP32 tensor.
        self.active["e"] = output[:, 0, :].detach().clone()

    def _first_block_pre_hook(self, module: Any, inputs: tuple[Any, ...]) -> None:
        if self.active is None:
            raise RuntimeError("first-block hook has no active model call")
        if not inputs:
            raise AssertionError("First Wan block did not receive positional x")
        self.active["z"] = inputs[0].detach()

    def _last_block_hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if self.active is None:
            raise RuntimeError("last-block hook has no active model call")
        if not self.torch.is_tensor(output):
            raise AssertionError("Expected tensor output from the final Wan block in full-compute mode")
        active = self.active
        e = active.get("e")
        z = active.get("z")
        if e is None or z is None:
            raise AssertionError("Observer did not capture e and Z before final-block output")
        h = output.detach()
        if h.shape != z.shape:
            raise AssertionError(f"H/Z shape mismatch: {tuple(h.shape)} vs {tuple(z.shape)}")

        key = (active["stage"], active["branch"])
        previous = self.previous.get(key)
        x_rel_l1 = None
        h_rel_l1 = None
        residual_rel_l1 = None
        if previous is not None:
            started = time.perf_counter()
            x_rel_l1 = self._relative_l1((e,), (previous["e"],), (1.0,))
            h_rel_l1 = self._relative_l1((h,), (previous["h"],), (1.0,))
            if self.collect_residual:
                residual_rel_l1 = self._relative_l1(
                    (h, z),
                    (previous["h"], previous["z"]),
                    (1.0, -1.0),
                )
            self.stat_elapsed_seconds += time.perf_counter() - started

        global_step = int(active["global_step"])
        paired_within_stage = previous is not None
        fit_eligible = paired_within_stage and global_step != 49
        row = {
            "sample_id": self.prompt["sample_id"],
            "category": self.prompt["category"],
            "source_line_1based": self.prompt["source_line_1based"],
            "stage": active["stage"],
            "branch": active["branch"],
            "global_step": global_step,
            "stage_step": int(active["stage_step"]),
            "timestep": active["timestep"],
            "e_rel_l1": x_rel_l1,
            "h_rel_l1": h_rel_l1,
            "block_residual_rel_l1": residual_rel_l1,
            "paired_within_stage": paired_within_stage,
            "fit_eligible": fit_eligible,
            "fit_exclusion": (
                None
                if fit_eligible
                else "stage_first_no_previous"
                if not paired_within_stage
                else "global_final_forced_recompute"
            ),
        }
        self.records.append(row)
        # H and Z are detached references to existing BF16 activations; no full-size clone.
        self.previous[key] = {"e": e, "h": h, "z": z}
        self.active = None

    def _relative_l1(
        self,
        current_terms: tuple[Any, ...],
        previous_terms: tuple[Any, ...],
        signs: tuple[float, ...],
    ) -> float:
        shape = current_terms[0].shape
        if len(shape) == 2:
            current_terms = tuple(term.unsqueeze(1) for term in current_terms)
            previous_terms = tuple(term.unsqueeze(1) for term in previous_terms)
            shape = current_terms[0].shape
        if len(shape) != 3:
            raise AssertionError(
                f"Expected [B, hidden] or [B, sequence, hidden] tensor, got {tuple(shape)}"
            )
        if any(term.shape != shape for term in (*current_terms, *previous_terms)):
            raise AssertionError("relative-L1 terms have inconsistent shapes")
        numerator = self.torch.zeros((), device=current_terms[0].device, dtype=self.torch.float32)
        denominator = self.torch.zeros_like(numerator)
        for start in range(0, shape[1], self.chunk_tokens):
            stop = min(start + self.chunk_tokens, shape[1])
            current = current_terms[0][:, start:stop].float() * signs[0]
            previous = previous_terms[0][:, start:stop].float() * signs[0]
            for current_term, previous_term, sign in zip(
                current_terms[1:], previous_terms[1:], signs[1:]
            ):
                current = current + current_term[:, start:stop].float() * sign
                previous = previous + previous_term[:, start:stop].float() * sign
            numerator = numerator + (current - previous).abs().sum()
            denominator = denominator + previous.abs().sum()
        return float((numerator / denominator.clamp_min(1.0e-16)).item())


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 70:
        raise ValueError(f"Expected a 70-prompt manifest, found {len(rows)} rows in {path}")
    if [row["ordinal"] for row in rows] != list(range(70)):
        raise ValueError("Manifest ordinals must be exactly 0..69")
    if len({row["sample_id"] for row in rows}) != 70:
        raise ValueError("Manifest sample_id values are not unique")
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def completed_sample(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete" and len(payload.get("records", [])) == 100


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_root / "samples"
    failure_dir = args.output_root / "failures"
    sample_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    if args.wan_repo is None or args.ckpt_dir is None:
        raise ValueError(
            "Set WAN22_SOURCE and WAN22_CKPT or pass --wan-repo and --ckpt-dir."
        )
    wan_repo = args.wan_repo.resolve()
    ckpt_dir = args.ckpt_dir.resolve()
    if not (wan_repo / "wan" / "text2video.py").is_file():
        raise FileNotFoundError(f"Invalid Wan2.2 repository: {wan_repo}")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Invalid checkpoint directory: {ckpt_dir}")
    validation_mode = (
        "prepared"
        if (wan_repo / ".teacache4wan22_prepared.json").is_file()
        else "upstream"
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_prepared_tree.py"),
            "--source",
            str(wan_repo),
            "--mode",
            validation_mode,
        ],
        check=True,
    )
    sys.path.insert(0, str(wan_repo))

    import torch
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

    manifest = load_manifest(args.manifest)
    shard_rows = [row for row in manifest if row["ordinal"] % args.num_shards == args.shard_index]
    if args.limit is not None:
        shard_rows = shard_rows[: args.limit]
    if not shard_rows:
        raise ValueError(f"Shard {args.shard_index} has no selected prompts")

    config = WAN_CONFIGS["t2v-A14B"]
    run_config = {
        "task": "t2v-A14B",
        "size": "832*480",
        "size_wh": list(SIZE_CONFIGS["832*480"]),
        "frame_num": 45,
        "sampling_steps": 50,
        "sample_solver": "dpm++",
        "shift": 12.0,
        "guide_scale_low_high": [3.0, 4.0],
        "boundary": float(config.boundary),
        "seed": args.seed,
        "fps": int(config.sample_fps),
        "param_dtype": str(config.param_dtype),
        "offload_model": True,
        "t5_cpu": False,
        "convert_model_dtype": True,
        "use_ret_steps": False,
        "cache_enabled_during_collection": False,
        "vae_decode": "skipped_after_denoising",
        "distance_reduction_dtype": "float32",
        "fit_x": "adjacent relative-L1 of first token of time_embedding output e; tokens are identical",
        "fit_y": "adjacent relative-L1 of post-all-blocks/pre-head hidden state H",
        "diagnostic": (
            "disabled"
            if args.skip_residual_diagnostic
            else "adjacent relative-L1 of block residual H-Z"
        ),
    }
    shard_metadata = {
        "status": "running",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_sample_ids": [row["sample_id"] for row in shard_rows],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "wan_repo": str(wan_repo),
        "wan_model_py_sha256": file_sha256(wan_repo / "wan" / "modules" / "model.py"),
        "wan_text2video_py_sha256": file_sha256(wan_repo / "wan" / "text2video.py"),
        "ckpt_dir": str(ckpt_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "run_config": run_config,
        "started_unix": time.time(),
    }
    metadata_path = args.output_root / "shards" / f"shard_{args.shard_index}.json"
    atomic_json(metadata_path, shard_metadata)

    logging.info("Creating Wan2.2 T2V-A14B pipeline for shard %d", args.shard_index)
    pipeline = wan.WanT2V(
        config=config,
        checkpoint_dir=str(ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=True,
    )
    # Polynomial fitting only needs the denoising trajectory, not pixels.
    pipeline.vae.decode = lambda latents: [torch.empty(0, dtype=torch.float32)]
    observer = BlockDistanceObserver(
        torch,
        chunk_tokens=args.chunk_tokens,
        collect_residual=not args.skip_residual_diagnostic,
    )
    observer.attach(pipeline)

    completed = 0
    skipped = 0
    try:
        for prompt_row in shard_rows:
            output_path = sample_dir / f"{prompt_row['ordinal']:02d}_{prompt_row['sample_id']}.json"
            if not args.overwrite and completed_sample(output_path):
                logging.info("Skipping completed sample %s", prompt_row["sample_id"])
                skipped += 1
                continue
            prompt_log = args.output_root / "logs" / f"{prompt_row['ordinal']:02d}_{prompt_row['sample_id']}.log"
            with log_context(prompt_log):
                logging.info("Starting ordinal=%s sample_id=%s category=%s", prompt_row["ordinal"], prompt_row["sample_id"], prompt_row["category"])
                logging.info("Prompt: %s", prompt_row["prompt"])
                observer.start_prompt(prompt_row)
                started = time.perf_counter()
                try:
                    pipeline.generate(
                        prompt_row["prompt"],
                        size=SIZE_CONFIGS["832*480"],
                        frame_num=45,
                        shift=12.0,
                        sample_solver="dpm++",
                        sampling_steps=50,
                        guide_scale=(3.0, 4.0),
                        seed=args.seed,
                        offload_model=True,
                    )
                    wall_seconds = time.perf_counter() - started
                    records, stat_elapsed = observer.finish_prompt()
                    payload = {
                        "status": "complete",
                        "prompt": prompt_row,
                        "run_config": run_config,
                        "wall_seconds_including_observer": wall_seconds,
                        "observer_stat_seconds": stat_elapsed,
                        "record_count": len(records),
                        "records": records,
                    }
                    atomic_json(output_path, payload)
                    stale_failure = failure_dir / f"{prompt_row['ordinal']:02d}_{prompt_row['sample_id']}.json"
                    stale_failure.unlink(missing_ok=True)
                    completed += 1
                    logging.info(
                        "Completed %s: records=%d wall_seconds=%.3f observer_stat_seconds=%.3f",
                        prompt_row["sample_id"],
                        len(records),
                        wall_seconds,
                        stat_elapsed,
                    )
                except Exception as exc:
                    failure = {
                        "status": "failed",
                        "prompt": prompt_row,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    atomic_json(failure_dir / f"{prompt_row['ordinal']:02d}_{prompt_row['sample_id']}.json", failure)
                    observer.clear_prompt()
                    raise
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        observer.close()

    shard_metadata.update(
        {
            "status": "complete",
            "completed_this_invocation": completed,
            "skipped_existing": skipped,
            "finished_unix": time.time(),
        }
    )
    atomic_json(metadata_path, shard_metadata)
    logging.info("Shard %d complete: completed=%d skipped=%d", args.shard_index, completed, skipped)


if __name__ == "__main__":
    main()
