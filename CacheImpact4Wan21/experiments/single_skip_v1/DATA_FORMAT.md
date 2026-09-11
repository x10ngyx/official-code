# 单步干预数据集保存合同 v2

标准依据：`Ours4Wan21/data_collection/src/ours4wan21_data/runtime.py::RuntimeCapture.save_artifacts`、`collector.py`和`publisher.py`。本项目采用相同原始latent格式、shared baseline/candidate组织和trajectory/step/branch三级表；因动作规则及标签不同，使用独立schema，避免旧训练器误接收末步干预。

## 结果目录

```text
<OUT>/
  README.md
  run.json                           # 协议/模型/源码/40prompt/GPU合同
  manifests/
    baselines.jsonl                  # 40条、稳定ID与prompt文本
    candidates.jsonl                 # 1960条，step-major coarse-first release_index
  shared_baselines/<sample_id>/
    baseline.mp4 -> video.mp4
    latents/step_000_input.pt ... step_049_input.pt
    trace.json, timing.json, performance.json, ffprobe.json
    rgb_f32.npy, features.pt, metrics.json
    COMPLETE.json                    # 原始采集与主MSE完成
    quality.json, quality_per_frame.csv
    video_metrics/{per_frame.csv,per_video.csv,summary.json,metrics.json}
    BASELINE_COMPLETE.json           # 全指标数据集完成记录
  shards/shard_00/candidates/<sample_id>__skip_<step>/
    candidate.mp4 -> video.mp4
    latents/step_000_input.pt ... step_049_input.pt
    # 其余文件同baseline；单机单worker使用shard_00
  completed/<trajectory_id>.json      # canonical质量完成后才生成
  published/
    snapshots/<version>/
      tables/trajectory_summary.{jsonl,csv}
      tables/step_transitions.{jsonl,csv}
      tables/branch_transitions.{jsonl,csv}
      summary.json, SHA256SUMS.json, README.md
    current -> snapshots/<version>   # 原子切换，旧snapshot独立保留
  summary.csv, by_step.csv            # 可在生成半程查看的诊断汇总
  vbench/<baseline|step_NN>/          # 每组40视频的custom VBench原始分项/聚合
```

所有大体积结果位于规定exp根，项目`experiment_results/`只有symlink。

## 原始状态与trace

- 一条轨迹固定50个输入tensor，直接`torch.load(..., weights_only=True)`读出CPU `float16 [16,21,60,104]`。不存字典包装、不只保存pool后特征、不裁掉目标步之后的状态。运行计算仍使用原始精度；保存时转换FP16，与数据集pipeline一致。
- 每份latent对应执行该步模型前的输入；下一个文件是该轨迹实际求解器更新后的状态，candidate不能用baseline后缀补齐。
- `trace.json.step_records`50条：0-based step_index、step_fraction、实际timestep/sigma、model_stage、动作、两个CFG动作及distance/累积量、latent路径/形状/dtype/统计/SHA、baseline同step路径。
- `trace.json.decisions`100条：按cond/uncond顺序记录真实执行、cache来源步、raw/SEA relative-L1及累积量。所有步都真实计算proxy，不从FP16 latent反推SEA信号。
- threshold字段为null，累积量仅记录“真实动作下保留/重置”的诊断历史，不参与动作选择。只有指定step跳过，其上一步全部重算保证cache age=1。
- 原精度前缀latent SHA、干预步proxy及G1与baseline逐值核验仍保留。FP16存档不能替代这些原精度一致性证据，也不是包含全部UniPC内部历史的可恢复进程checkpoint。

## 两种质量口径

1. `metrics.json.terminal_rgb_mse`：pre-encode RGB float32终局MSE，作为单次缓存影响的主要标签；原始视频保存在rgb_f32.npy，可完整重算。
2. `quality.json`：原始RGB上的辅助PSNR/SSIM/LPIPS与MSE复核；汇总表中加`preencode_`前缀。
3. `video_metrics/`：与原pipeline一致，输入为MP4解码RGB，使用共享VideoMetrics的PSNR/SSIM/LPIPS、逐帧/逐视频统计和完整summary。三级表的mean_psnr/mean_ssim/mean_lpips及final_mean_*均来自这一canonical口径。

不要用canonical MP4 MSE替换敏感的pre-encode主标签。VBench分数为每个step组的custom raw mean，不是每条轨迹的独立分数或官方Total；发布snapshot的summary.json单列vbench_group_scores及路径/SHA。

## 三级表与完成语义

- trajectory_summary每候选一行：冻结ID/release_index/prompt_rank/split、prompt、baseline/候选路径、真实动作数量、组件时间/TFLOPs、canonical指标统计与terminal_rgb_mse。
- step_transitions每候选50行：完整逐步状态索引和动作、两个CFG信号、baseline latent配对、最终指标，以及is_intervention_step。
- branch_transitions每候选100行：完整CFG分支、距离/动作/执行和对应latent路径。
- 每step行携带的final_terminal_rgb_mse是整条episode结果，**不是这50步各自的边际标签**；本次被操纵的动作只由is_intervention_step标识。不能把同一条轨迹的终局损失当作50个不同干预的监督标签。
- split接受输入名单提供的train/val/test，缺省unassigned。不得在分析时随机拆step而让同prompt泄漏到测试集。
- `COMPLETE.json`只保证采集产物；`completed/`只有在原始质量和canonical质量文件通过校验后出现，内含trajectory_row/step_rows/branch_rows及产物哈希。
- `publish`只发布manifest中连续完成的候选前缀，提供CSV与保留嵌套字段的JSONL。空洞之后的候选不冒充连续完成。无变化时不重复生成snapshot；质量或VBench增加后发布新版。

## 读取示例

在激活wan2.2并设置四项线程变量为1后：

```python
import json
from pathlib import Path
import torch

root = Path('/all/yiran07-disk3/huteng_data/exp/YOUR_RUN')
tables = root / 'published/current/tables'
rows = [json.loads(line) for line in (tables / 'step_transitions.jsonl').read_text().splitlines()]
target = next(row for row in rows if row['is_intervention_step'])
candidate_input = torch.load(target['latent_path'], map_location='cpu', weights_only=True)
baseline_input = torch.load(target['baseline_latent_path'], map_location='cpu', weights_only=True)
label = target['final_terminal_rgb_mse']
```

路径沿用原pipeline的绝对archive路径约定；若整体迁移结果根，需显式重定位路径并重新发布，不可只移动tables。旧v1缺少50份原始latent，无法无损升级为v2；使用新OUT采集，保留原包和原结果。

## 资源

全量2000视频包含100000份FP16输入latent，约390.5GiB；原始RGB约723.0GiB；两者合计1113.6GiB，另计MP4和索引。首轮480视频约267.3GiB。latent暂存/诊断操作计入instrumented generate墙钟，转CPU FP16与文件I/O在generate返回后，不改模型驻GPU协议。
