# MagCache4Wan22

Wan2.2-T2V-A14B integration of the **unchanged official MagCache algorithm**,
with the workspace's fixed inference and measurement protocol.

## Official implementation

`vendor/magcache_generate.py` is a byte-identical snapshot of
[Zehong-Ma/MagCache@df81cb181776c2c61477c08e1d21f87fda1cd938](https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4Wan2.2/magcache_generate.py).
The adapter checks its SHA256 and compiles its original function ASTs without
editing any function. Omitting the upstream CLI imports keeps configuration
and output handling in this package; it does not substitute a new controller.

- Original T2V table: 78 ratios plus two initial 1.0 values (40 CFG-paired steps).
  The official per-branch nearest interpolation expands it to 50 steps.
- Original recurrence: multiply ratios, accumulate `abs(1 - accumulated_ratio)`,
  reuse only under the strict error threshold and inclusive consecutive-skip K.
- Original cached tensor: all-block output minus block input. Reuse adds the
  unchanged cached residual; it does not rescale it by the magnitude ratio.
- Original independent cond/uncond decisions and shared high/low class state.
- Original high/low retention conditions, integer rounding, and inclusive low
  boundary. Forced-full retention calls update residuals **without resetting
  accumulated ratio/error/count**. No extra stage reset or final-step full call
  is added. The split uses the official simplified timestep helper.
- One fresh initialization per video, matching separate original CLI invocations.
  Class and instance attributes are restored after success or failure so the
  next video and native baseline cannot inherit mutated counters or forwards.

The default method settings follow the official A14B example:
`threshold=0.06`, `K=2`, `retention_ratio=0.4`. These differ from the upstream
parser defaults `.04/2/.2`. All three parameters are exposed. Nonfinite and
negative values are rejected; retention must be in `(0,1]` because upstream's
zero-retention path can reuse a missing initial cache. `K=0` is allowed.

## Fixed inference protocol

Wan2.2-T2V-A14B, batch 1, `832×480`, 45 frames, 16 fps, 50-step DPM++, shift 12,
low/high CFG `(3,4)`, boundary `.875`, seed 42, BF16 DiT, one GPU,
`offload_model=True`, `t5_cpu=False`, no FSDP/SP or prompt extension.

The official README's example uses upstream defaults (40-step UniPC, 81 frames,
T5 CPU). We preserve **its algorithm** under the project's required model
protocol; this is not a claim to reproduce the README's historical latency.
The built-in 40-step table is interpolated exactly as upstream specifies,
without a new fit or recalibration. The package intentionally targets A14B T2V;
the vendor snapshot also contains the upstream I2V/5B code, which is not exposed
by this package's fixed-protocol runner.

## Files

- `vendor/`: unchanged official script, README and Apache-2.0 license.
- `runtime/`: original-function loader, scoped installation, observational trace,
  source/protocol checks, and component timing.
- `scripts/`: pristine Wan2.2 preparation, validation and single-video runner.
- `configs/`: frozen inference protocol.
- `experiments/performance_t2v_a14b/`: real-shape high/low DiT and T5/VAE Calflops
  profiling, plus measured-call-weighted aggregation.
- `experiments/paired_benchmark/`: paired video generation, performance,
  PSNR/SSIM/LPIPS, and both baseline/candidate VBench scores.
- `experiments/smoke_official_core/`: CUDA/BF16 parity on small random-weight
  WanModel instances; no A14B weights or quality benchmark.
- `tests/`: CPU comparison against a full import of the unchanged official
  script, scalar schedule cross-checks, lifecycle and reporting contracts.
- `experiment_results/`: symlinks into the external experiment disk.
- `PROGRESS.md`, `logs/`: local status and handoff notes, excluded from Git.

## Setup and single-video usage

Use conda `wan2.2`; no installation or modification of the shared environment
is needed. Model directories must be registered under the workspace `models/`
root (existing symlink registrations are supported).

```bash
conda activate wan2.2
export WAN22_PYTHON="$CONDA_PREFIX/bin/python"
bash scripts/prepare_wan22.sh "$PWD/build/Wan2.2-42bf4cf"

bash scripts/run_t2v_a14b.sh \
  --source "$PWD/build/Wan2.2-42bf4cf" \
  --checkpoint /path/to/workspace/models/Wan2.2-T2V-A14B \
  --output-dir /all/yiran07-disk3/huteng_data/exp/magcache_wan22_example \
  --prompt 'A man is talking to a woman in the office room.' \
  --mode magcache --threshold 0.06 --magcache-k 2 --retention-ratio 0.4
```

`--mode baseline` leaves the native Wan2.2 forward intact; this is distinct from
MagCache with threshold zero, which still runs the official replacement
forward. `--mode calibrate` calls the original calibration function and saves
its ratio/std/cosine JSON files under the result's `calibration/` directory.
Their legacy `wan2_1_*` names are retained from upstream. Calibration does not
overwrite the official built-in table or implicitly change future runs.

Add `--dry-run` to validate the locked source, model location, method parameters,
and plan without loading weights or creating a result. Source preparation
checks out `Wan-Video/Wan2.2@42bf4cfaa384bc21833865abc2f9e6c0e67233dc`
and applies **no patches**. Existing destinations/results are never overwritten.
`WAN22_REPOSITORY` can point to a local Git clone for offline source preparation.

## Baseline 主干与批量实验

本项目准备的 `build/Wan2.2-42bf4cf/generate.py` 与上游逐字节一致，原始主干 SHA
也与 SeaCache4Wan22、TeaCache4Wan22 锁定记录一致。单视频和 batch 共用
`runtime/generation.py` 的原生 `WanT2V.generate` 调用；baseline 不安装 MagCache forward。
SeaCache 关闭缓存后的采样器、WanModel 和 attention block 数值路径已通过 AST 等价核对。
各方法 CLI 的缓存参数和计时代码不同，不能声称 prepared `generate.py` 整文件相同。
完整 A14B 跨方法 baseline 视频尚未生成并逐位对比。

- [Vbench200 batch](experiments/vbench200_t2v/README.md)：每卡每进程一次模型加载，
  同卡配对、默认一次完整 warmup、逐视频重置缓存/计时器、严格断点恢复和质量评测。
- [参数组合扫描](experiments/parameter_scan/README.md)：threshold × K × retention，
  11 条覆盖 16 维的固定 prompt，共享 baseline，按实测完整推理时间选择约 1.8×/2.4×/3.0×。
- [三档 E 细扫](experiments/targeted_threshold_scan/README.md)：固定每档 R/K，
  E 起点 `.075/.386/.200`，步长 `.001`，组内选择实测最近值，默认相对 ±2%；尚未实测标定。
- [源码审计](scripts/audit_baseline.py)：记录共同上游 SHA、cache-disabled AST 等价性与依赖 SHA。

旧 `experiments/paired_benchmark/run_benchmark.py` 已转为持久 batch 兼容入口。
`--dry-run` 仅输出计划；`--resume` 校验现有结果后继续；`--defer-evaluation`
允许先完成计量，保留 quality pending，之后不重新加载模型即可继续评测。

## Timing, FLOPs and quality

Headline latency is CUDA-synchronized complete `WanT2V.generate()` wall time:
T5, denoising, cache, CFG/scheduler, in-generate model transfers and VAE decode.
Pipeline initialization, MP4 encoding, trace writes and evaluation are excluded.
T5, DiT and VAE CUDA and host times are retained separately. Timing and cache
traces independently observe actual block execution and must agree per call.

DiT headline TFLOPs use real-shape Calflops profiles of the original MagCache
forward, with separate full-stack and embeddings/head-only measurements and
manual dense FlashAttention-core compensation. Each measured cond/uncond call
selects the matching stage/branch cost. T5 and VAE are profiled separately.
Scalar controller, residual addition, CFG/scheduler and video export FLOPs are
excluded and explicitly disclosed. TFLOPs means operation count, not TFLOP/s.

The paired benchmark uses sibling `VideoMetrics` RGB PSNR/SSIM/spatial AlexNet
LPIPS and `VbenchEvaluation`. Standard mode requires exact Vbench200 prompts
covering all 16 dimensions. Its score is a Vbench200 subset score. Explicit
custom mode uses the existing ten-dimension custom-input raw mean and labels
it separately from the standard score. A benchmark is marked complete only
after performance and all required quality stages succeed.

## Validate

```bash
WAN22_PYTHON="$CONDA_PREFIX/bin/python" bash tests/run_tests.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/validate_source.py --source "$PWD/build/Wan2.2-42bf4cf"
```

CPU parity exercises the original forward on deterministic small tensor models,
both noise stages and independent CFG branches, including 40→50 interpolation,
retention boundaries, K, calibration, repeated videos and exception cleanup.
It verifies exact tensor equality with the official script; it does not establish
full A14B GPU video quality, latency or memory use.

The local implementation check also runs real pinned WanModel blocks on CPU
with a substituted SDPA attention kernel and compares native versus official
all-full outputs. The separate CUDA smoke exercises the actual CUDA attention
path on small BF16 models. Neither check is a full A14B video benchmark.
