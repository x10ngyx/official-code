# Wan2.1 random-threshold collection v1

This experiment freezes a 3,000-prompt OpenVid sample and three randomized
SeaCache threshold trajectories per prompt.  The four shards are deterministic:
750 baselines and 2,250 candidates per GPU.  A pipeline is loaded once per
shard and reused serially.

Set `PROMPT_LIMIT=1000` for the first release requested in September 2026.
That mode collects exactly 1,000 baselines and the first 3,000 candidate rows
(250/750 per GPU). The full 9,000-row manifest is still frozen: its first
1,000 prompts have an exact 800/100/100 split, while later extension fills out
the unchanged full 2,400/300/300 split. Prompt membership remains a uniform
sample without replacement, and candidate target-speedup/q/path seeds are
unchanged. The manifest summary contains the observed stage-vs-full drift
checks; finite-sample proportions are close, not bit-for-bit identical.

Run `launch_4gpu.sh <phase>`. Supported phases are:

- `preflight`: verify this project and its shared `offical-code/` resources.
- `plan`: freeze the 9,000-row pending plan; no GPU is used.
- `profile`: create the real-shape Calflops profile on one GPU.
- `baselines`: collect the 3,000 shared full-compute references on four GPUs.
- `materialize`: combine the pending plan with a separately fitted calibrated
  speedup-to-mean-threshold mapping.  The checked-in pending mapping is
  intentionally rejected.
- `candidates`: require the runnable manifest and every baseline, then collect
  9,000 random trajectories on four GPUs.
- `publish`: atomically publish the longest contiguous completion prefix.
- `finalize`: require all 9,000 rows, publish, and run the archive audit.

When `PROMPT_LIMIT` is set, `baselines`, `candidates`, VBench and `finalize`
operate on that prompt prefix. Stage finalization requires the corresponding
candidate prefix (`3 * PROMPT_LIMIT`) and writes count-namespaced VBench/audit
artifacts. Omit `PROMPT_LIMIT` later to extend and finalize the full archive.

Training-data releases may skip VBench without weakening publication or the
archive consistency audit. Place a non-empty scoped control at
`controls/skip_vbench_prefix_<PROMPT_LIMIT>.json`; `finalize` then records
`vbench_required=false`, leaves `vbench_summary=null`, and still requires and
audits the complete candidate prefix. The full-archive scope uses
`controls/skip_vbench_full.json`, so a prefix decision does not silently carry
over to a later full-archive extension. The explicit `vbench` phase always runs
VBench and ignores this finalize-only control.

Set `RUN_ID`, `WAN21_ROOT`, `CHECKPOINT_DIR`, and after fitting
`CALIBRATION_CONFIG` as needed. Set `EXP_BASE` when the remote result root is
not `/all/yiran07-disk3/huteng_data/exp`; the project result directory receives
only a symlink. Worker logs are part of that external result archive.

The bundled prompt pool is the default. PSNR/SSIM/LPIPS and manual FLOP
accounting use sibling `VideoMetrics/` and `CalflopsEvaluation/`. One LPIPS
AlexNet model is loaded per candidate worker and reused across that shard;
configure `METRICS_MODEL_CACHE`, `METRICS_DEVICE`, and `LPIPS_BATCH_SIZE` when
needed. Metric/decode time remains outside inference speedup. See
`REMOTE_DEPLOYMENT.md` for the external-input checklist.

The frozen source checker rejects a Wan2.1 tree whose four compatibility files
do not match commit `65386b2e03c490796eede31b0325a6a595cc684e`.

Candidate trace schema v2 records 100 ordered CFG branch calls. Each branch
row contains the SEA-filtered adjacent-step relative-L1 and accumulator
`before`, `with_current` (the threshold operand before reset), and `after`.
Published snapshots expose these rows in `tables/branch_transitions.jsonl` and
`tables/branch_transitions.csv`, in addition to the 50-row step view.

For an unattended staged run, `queue_after_calibration.sh` waits until
`CALIBRATION_CONFIG` exists and all four GPUs have remained idle for 60
seconds, then runs `candidates` followed by fail-closed `finalize`. It never
signals an existing process. The caller must export the same runtime variables
as the launcher, including `PROMPT_LIMIT=1000` and `VBENCH_PYTHON`.
