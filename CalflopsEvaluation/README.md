# Calflops Evaluation

本项目提供缓存式扩散推理的 FLOPs/TFLOPs 评测资源。它把工作拆成两层：

1. 用 [Calflops](https://github.com/MrYxJ/calculate-flops.pytorch) 对真实 `torch.nn.Module.forward` 输入逐路径计数。
2. 按实际 cache step trace 累加路径成本，并分别计算 baseline 与方法的每视频总 TFLOPs 和 FLOPs speedup。

这里的 `TFLOPs` 是 `10^12` 次浮点运算的**数量**，不是 `TFLOP/s`。Calflops 的终端字符串会写成 `TFLOPS`，本项目统一保存原始数值 `flops` 和 `tflops = flops / 10^12`，避免把运算量误写成吞吐率。

## 目录结构

- `calflops_eval/`：profile adapter、Calflops 调用、手工算子补偿、trace 聚合与 CLI。
- `examples/`：可直接运行的 toy adapter、action mapping 和 trace 示例。
- `tests/`：CPU-only 回归测试。
- `upstream_lock.json`：Calflops 版本、官方仓库 commit、许可证和已知覆盖边界。
- `requirements.txt` / `pyproject.toml`：可复现安装入口。
- `PROGRESS.md` / `logs/`：本子项目状态和交接记录。

## 安装

Wan2.2 使用项目既有的 `wan2.2` 环境：

```bash
conda activate wan2.2
python -m pip install -e work/offical-code/CalflopsEvaluation
```

依赖锁定 `calflops==0.3.2`。模型权重继续使用根目录 `models/`，本项目不复制权重。

## 完整评测流程

### 1. 冻结计数范围

论文主结果建议使用 `DiT + cache path, forward-only`：

- 包含 high/low DiT、cond/uncond CFG、cache controller/probe、reuse/correction 路径。
- 不包含 text encoder、VAE、视频导出、文件 I/O、质量评估。
- 固定 batch size、分辨率、帧数、采样步数、context 长度、precision 和 attention backend。
- 固定 `1 MAC = 2 FLOPs`、十进制 `1 TFLOP = 10^12 FLOPs`。

若最终只覆盖 DiT 线性层而没有补齐 attention/controller 等自定义算子，结果必须标成 `DiT-forward partial TFLOPs`，不能称完整 TFLOPs。

### 2. 为每种真实执行路径构造 ProfileCase

adapter 必须用模型推理时的真实 tensor shape 和 `kwargs`。Wan2.2 `WanModel.forward` 的核心输入是：

```python
model(
    x=[latent],          # 每个元素 [C, F_latent, H_latent, W_latent]
    t=timestep,          # [batch]
    context=[embedding], # 每个元素 [text_tokens, 4096]
    seq_len=seq_len,
)
```

当前正式协议 `832×480 / 45 frames` 对应 latent shape `[16, 12, 60, 104]`、patch token 数/`seq_len=18720`。应分别建立至少：

- `high_full_cond`
- `high_full_uncond`
- `low_full_cond`
- `low_full_uncond`
- `cache_controller`
- `reuse_path`

high/low 或 cond/uncond 理论上可能同构，但正式计量仍各 profile 一次并检查数值是否一致，不直接假定。

adapter 接口示例见 `examples/toy_adapter.py`。它返回 `ProfileCase` 或 `ManualComponent`：

```python
from calflops_eval import ManualComponent, ProfileCase

def build_profile_items():
    return [
        ProfileCase(
            name="high_full_cond",
            model=high_model,
            kwargs={"x": [latent], "t": timestep, "context": [context], "seq_len": 18720},
        ),
        ManualComponent(
            name="custom_attention_correction",
            flops=attention_flops,
            formula="2 FLOPs per QK MAC + softmax + 2 FLOPs per AV MAC",
        ),
    ]
```

运行 profile：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
calflops-eval profile \
  --adapter path/to/wan22_adapter.py:build_profile_items \
  --output /path/to/cost_table.json \
  --print-detailed
```

`--print-detailed` 用于人工检查各子模块是否出现异常 `0 FLOPs`；正式批量时可关闭。

### 3. 补齐 Calflops 未覆盖的算子

Calflops 0.3.2 能统计 Linear/Conv 和若干 `torch.matmul/einsum/softmax`，但不会自动识别 Wan2.2 当前使用的 FlashAttention 或 `torch.nn.functional.scaled_dot_product_attention`，也不能保证覆盖 FFT、自定义 CUDA op 和 Python cache controller。

因此流程不是“运行一次 Calflops 就结束”，而是：

1. 查看 detailed profile，找出零计数或明显偏低的模块。
2. 对 attention core 使用 `calflops_eval.manual_ops.dense_attention_counts` 补偿；FlashAttention 只改变实现，不改变 dense attention 的理论运算量。
3. controller 若是 `nn.Module`，单独建立 `ProfileCase`；纯函数/FFT 用 `ManualComponent` 加入解析计数。
4. 把所有补偿公式和输入维度写进 component metadata。

### 4. 用 mapping 定义每个 cache action 执行哪些组件

`examples/toy_mapping.json` 展示了推荐结构。以 timestep cache 为例：

- `baseline`：每步执行 cond full + uncond full，不包含方法 controller。
- `recompute`：执行 controller + cond full + uncond full。
- `reuse`：执行 controller + reuse/correction。

mapping 按 high/low stage 分开，组件名必须来自 `cost_table.json`。

### 5. 从真实 trace 聚合每视频 TFLOPs

trace 支持 JSONL、JSON 数组，或包含 `step_records` 的项目 `trace.json`。默认读取：

- `sample_id`
- `step_index`
- `model_stage`
- `decision`

运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
calflops-eval aggregate \
  --cost-table /path/to/cost_table.json \
  --mapping /path/to/action_mapping.json \
  --trace /path/to/steps.jsonl \
  --output /path/to/flops_summary.json
```

聚合公式为：

```text
candidate_flops(video) = Σ_step Σ_component F(component | stage, actual_action)
baseline_flops(video)  = Σ_step Σ_component F(component | stage, baseline_action)
FLOPs speedup          = Σ_video baseline_flops / Σ_video candidate_flops
```

headline 使用总量比，不平均逐视频 speedup。输出同时保留 per-sample 计数，便于检查动态策略的 prompt 差异。

### 6. Sanity checks

正式发布前至少通过：

- no-cache trace 的 FLOPs speedup 应为 `1.0×`。
- baseline 总量应等于 50 个 step 的分阶段逐步和；Wan2.2 当前 stage 为 high `32`步、low `18`步。
- 25/15-step vanilla 主体 FLOPs 应近似按步数线性变化。
- high/low、cond/uncond 同构时，profile 数值应一致；不一致必须解释输入 shape 或执行分支差异。
- 随机抽一个小 shape，用解析 Transformer/attention 公式交叉核验。
- 输出中不得存在未说明的 zero-count attention、FFT、controller 或 reuse component。

## Toy smoke test

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
calflops-eval profile \
  --adapter examples/toy_adapter.py:build_profile_items \
  --output /tmp/toy_costs.json \
  --overwrite

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
calflops-eval aggregate \
  --cost-table /tmp/toy_costs.json \
  --mapping examples/toy_mapping.json \
  --trace examples/toy_trace.jsonl \
  --output /tmp/toy_summary.json \
  --overwrite
```

## 结果目录

正式模型评测结果应写到仓库外部的独立实验目录。不要把大型 profile、模型权重或生成结果复制进本源码目录。
