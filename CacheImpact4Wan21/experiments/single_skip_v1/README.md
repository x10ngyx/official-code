# single_skip_v1：远端运行手册

## 文件

`main.py/run.sh`公开入口；`common.py`计划/合同/原子完成；`runtime.py`干预与特征；`collect.py`持久GPU进程；`evaluate.py`独立评测/增量CSV；`dataset.py`完整latent存档/数据集manifest/三层表发布；本目录不存实验产物。

## 1. 准备代码、环境、模型、40个prompt

保留工作区布局`<workspace>/work/offical-code/`，将本项目及以下兄弟目录一起同步到远端：
`SeaCache4Wan21`、`Wan21Benchmark`、`ComponentMetrics`、`CalflopsEvaluation`、`DiCache4Wan21/upstream_lock.json`、`VideoMetrics`、`VbenchEvaluation`、`Vbench200`。
排除所有`experiment_results/`、`__pycache__/`、本地日志和大型产物。模型置于`<workspace>/models/`，不得复制进源码。

使用现有`wan2.2`环境，准备locked Wan2.1源码及其依赖、PyTorch/CUDA、imageio/imageio-ffmpeg、VideoMetrics所需numpy/OpenCV/lpips、Calflops 0.3.2与VBench依赖。无需新建其他环境。Wan源要求commit `65386b2e03c490796eede31b0325a6a595cc684e`，入口检查关键文件SHA。权重为`models/Wan2.1-T2V-1.3B`，支持该注册目录指向外部存储。

准备40行JSONL，每行形如：

```json
{"sample_id":"p001","prompt_en":"A dog running happily."}
```

也接受`prompt`字段。可提供`split`（train/val/test/unassigned），未提供时保存为unassigned，不擅自划分。必须40个不同ID及不同prompt文本；保持行顺序。没有擅自替用户选40条prompt。可从`Vbench200/prompts.jsonl`选定后冻结；不要只截前40条而误称均衡抽样。为了后续无泄漏比较，应核对它们是否出现在已有latent encoder训练集中。

从`<workspace>/work/offical-code`执行下列命令。请替换WAN_SRC和PROMPTS，OUT必须位于规定存储根。

```bash
conda activate wan2.2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME="$(pwd)/../../models/torch-cache"
export HF_HOME="$(pwd)/../../models/huggingface"
export XDG_CACHE_HOME="$(pwd)/../../models/xdg"
WAN_SRC=/absolute/path/to/locked/Wan2.1
PROMPTS=/absolute/path/to/prompts40.jsonl
OUT=/all/yiran07-disk3/huteng_data/exp/wan21_single_skip_40_v1
CKPT="$(pwd)/../../models/Wan2.1-T2V-1.3B"
ENTRY=CacheImpact4Wan21/experiments/single_skip_v1/run.sh
bash "$ENTRY" plan --prompts "$PROMPTS"
```

`plan`不加载模型、不访问GPU、不写结果。首轮按2/5/10/.../50执行，`--stage all`也保证首轮在先、遗漏step在后。

## 2. 一次性Calflops组件profile

不要与采集并占同一卡。profile包含DiT完整/无block路径、两次T5编码和一次VAE解码；源权重和配置需与采集匹配。

```bash
python Wan21Benchmark/experiments/component_profile/profile_calflops.py \
  --wan21-root "$WAN_SRC" --checkpoint-dir "$CKPT" \
  --output "$OUT/component_profile.json"
```

此脚本拒绝覆盖既有profile；已有正确profile无需重复跑。

## 3. 先跑首轮（默认）

```bash
bash "$ENTRY" run --prompts "$PROMPTS" --wan21-root "$WAN_SRC" \
  --checkpoint-dir "$CKPT" --profile "$OUT/component_profile.json" \
  --output "$OUT" --stage coarse
```

可以在激活好环境的tmux内运行。实际Python进程也会显式设置四项线程变量为1。仅一张48GB GPU、模型持久驻留；不会自动ssh到其他机器或发起远程任务。

`--max-new-cells 1`可在完成一条新轨迹后退出，便于远端验收。使用单prompt smoke时须另建结果目录、单行JSONL，显式`--expected-prompts 1 --max-new-cells 2`；两条新cell为baseline和第2步候选。末步路径由CPU全部49干预测试覆盖，完整首轮会实测第50步。

每次加载后先跑原生全算与instrumented全算，比较完整float32视频bitwise equality；不一致直接停止。每候选还逐步比对直到目标步的latent SHA，并检查目标步proxy/G1与baseline相同；trace与真实block调用必须为98次full+2次reuse（baseline100次full）。第2步/第50步均同步切换两个CFG分支。

每条结束立即写`metrics.json`主MSE并更新`summary.csv/by_step.csv`，不必等LPIPS/VBench即可初看趋势。baseline与候选均完整重新采样；没有前缀快照复用，也没有额外oracle全算目标步。

## 4. 评测首轮，并查看结果

采集进程退出释放模型后执行，可对已完成的部分cell评测；VBench只评完整40-prompt的step组。

```bash
bash "$ENTRY" evaluate --output "$OUT" --stage coarse --device cuda:0
bash "$ENTRY" vbench --output "$OUT" --stage coarse
bash "$ENTRY" summarize --output "$OUT"
bash "$ENTRY" audit --output "$OUT" --stage coarse
```

`evaluate`既在原始RGB上复算MSE及质量（`quality.json/quality_per_frame.csv`），也通过原pipeline的`VideoMetrics.evaluate_pairs/write_evaluation`在MP4解码后生成标准`video_metrics/per_frame.csv`、`per_video.csv`、`summary.json`和`metrics.json`。完成后写`completed/<trajectory_id>.json`（baseline写BASELINE_COMPLETE.json），含trajectory_row、50 step_rows、100 branch_rows。每cell可续评；本轮评测末尾自动publish连续完成前缀，也可手动`bash "$ENTRY" publish --output "$OUT"`。

`vbench`用相邻`run_custom_vbench.sh`处理任意40prompt，为baseline及每个step各计算10个custom维度，输出`vbench_score`。该值是明确标注的**本地custom原始分数均值，不是官方完整Total Score**；原始维度保留。模型缓存固定在`models/VBench`、`models/torch-cache`等目录。依赖/权重缺失时失败而不伪造或静默跳过分数。`audit`在缺少任何目标cell、质量结果或VBench组时返回非零。

## 5. 升序补齐剩余step

```bash
bash "$ENTRY" run --prompts "$PROMPTS" --wan21-root "$WAN_SRC" \
  --checkpoint-dir "$CKPT" --profile "$OUT/component_profile.json" \
  --output "$OUT" --stage remaining
bash "$ENTRY" evaluate --output "$OUT" --stage remaining --device cuda:0
bash "$ENTRY" vbench --output "$OUT" --stage remaining
bash "$ENTRY" audit --output "$OUT" --stage all
```

`remaining`要求首轮生成已完成；baseline不会重跑。若希望不中断自动补齐，可首次直接`--stage all`。用户希望尽快判断效果时默认coarse停在首轮边界。

## 续跑和输出合同

- `COMPLETE.json`仅在视频、RGB、trace、features、timing、主MSE都写完并哈希后出现。只有完整匹配的cell可跳过；损坏完整产物立即报错，禁止静默接受。
- 未完成cell在重试前移入`_incomplete/`，不会覆盖旧证据；重跑该cell。单个结果目录有排他worker锁。
- `run.json`绑定40prompt文本/顺序、完整模型文件SHA、共享与本项目源码SHA、profile SHA、GPU身份和Torch/CUDA版本。不同条件禁止混用同一结果目录。
- `latents/`始终保存全部50份CPU FP16输入tensor，名称与原数据集pipeline一致；`trace.json`含50 step_records、100 decisions，记录sigma/timestep及双方同step路径。保存包括干预后的全部状态，FP16转换和I/O在generate计时外；原始状态暂存在GPU，不offload模型。
- `features.pt`保存可因果获得的G1输入（baseline全部49步、候选干预步）和全程proxy；它是便捷派生层，后续可从完整latents重新提取其他特征。`latent_hashes`仍只用于原精度前缀一致性核验。
- `timing.json`含完整generate及T5/DiT/VAE计时；`metrics.json`含DiT TFLOPs和T5/VAE独立TFLOPs。完整generate包含诊断特征/哈希开销，**不是无诊断的生产推理latency**。FLOPs只计真实模型路径，诊断FFT/池化/哈希不混进DiT估计值。
- 原始RGB约0.362GiB/视频、全50步FP16 latent约0.195GiB/视频；完整合计约1114GiB（1.09TiB），首轮约267GiB，另计MP4与表。保留float32是为了可复算微小终局差异，删掉后不能仅凭MP4重建主标签。
- `COARSE_GENERATION_COMPLETE.json`与`ALL_GENERATION_COMPLETE.json`只表示生成完成；完整评测以`AUDIT_all.json`为准。

## 数据格式版本

当前保存协议为`single_skip_dataset_v2`。旧v1只保存目标特征，不能无损补出后缀原始latent，因此禁止用旧结果冒充v2；新采集使用新的OUT。旧交接包保留作历史，使用本次dataset_v2包。详见[DATA_FORMAT.md](DATA_FORMAT.md)。
