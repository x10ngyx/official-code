# TeaCache4Wan21

本目录以锁定的原始 Wan2.1 为唯一推理基础，并在显式启用时注入 TeaCache。
不启用 TeaCache 时，统一入口直接调用原始 Wan2.1 `generate.py`，不导入或修改
TeaCache；启用后才绑定 TeaCache 官方 T2V/I2V 采样函数与模型 `forward`，并配置
官方状态和系数。

官方参考文件固定为
`ali-vilab/TeaCache@7c10efc4702c6b619f47805f7abe4a7a08085aa0` 中的
`TeaCache4Wan2.1/teacache_generate.py`。缓存判据、相对 L1 距离、多项式重标定、
残差复用、CFG 条件/无条件双缓存和 retention steps 均与该文件一致。

测试边界按本项目要求替换：PSNR、SSIM、LPIPS 使用兄弟目录 `VideoMetrics/`，
VBench score 使用 `VbenchEvaluation/` 的 Vbench200 子集协议，不使用 TeaCache
仓库自带的 `eval/teacache/`。

## 目录结构

- `generate.py`：统一入口；baseline 直接执行原始 Wan2.1，TeaCache 必须显式开启。
- `inference_timing.py`：可选 CUDA event/单调时钟计时和逐 DiT call block trace。
- `teacache.py`：把官方函数、状态和系数绑定到刚创建的原始 Wan2.1 pipeline。
- `upstream/teacache_generate.py`：只用于来源锁定与方法复用的官方字节副本。
- `LICENSE.upstream.txt`、`NOTICE.md`：上游 Apache-2.0 许可证与来源说明。
- `run_wan21.sh`：检查 Wan2.1 源码版本后运行统一入口。
- `upstream_lock.json`：TeaCache、Wan2.1 版本和关键文件哈希。
- `validate_reproduction.py`：原始 baseline、官方方法、兼容源码和评测边界验证。
- `tests/`：无需模型或 GPU 的静态集成回归测试。
- `experiments/vbench200_t2v/`：固定 Wan2.1-T2V-1.3B 批量生成与统一评测编排。
- `experiments/threshold_zero_smoke/`：一致性、组件 latency/TFLOPs 与统一画质冒烟测试。
- `experiment_results/`：外部实验结果索引；大文件不得保存在代码目录。
- `PROGRESS.md`、`logs/`：本地状态面板和精炼交接记录。

## 上游环境

官方 TeaCache 文件依赖 Wan2.1 源码树。本复现将兼容版本固定为 Wan2.1 初始提交：

```bash
git clone https://github.com/Wan-Video/Wan2.1.git /path/to/Wan2.1
git -C /path/to/Wan2.1 checkout 65386b2e03c490796eede31b0325a6a595cc684e
cd /path/to/Wan2.1
pip install -r requirements.txt
```

Wan2.1 官方要求 PyTorch 2.4.0 或更高版本。模型权重仍放在工作区统一的
`models/` 目录中，不复制进本项目。运行前验证源码：

```bash
python validate_reproduction.py --wan21-root /path/to/Wan2.1
```

## 统一入口：baseline 与 TeaCache

下面两条命令除 TeaCache 开关和阈值外完全相同。no-cache baseline：

```bash
bash run_wan21.sh /path/to/Wan2.1 \
  --task t2v-1.3B \
  --size '832*480' \
  --ckpt_dir /path/to/models/Wan2.1-T2V-1.3B \
  --prompt 'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.' \
  --base_seed 42 \
  --offload_model False \
  --save_file /all/yiran07-disk3/huteng_data/exp/teacache_wan21/no_cache.mp4
```

官方 TeaCache T2V-1.3B fast 配置：

```bash
bash run_wan21.sh /path/to/Wan2.1 \
  --enable_teacache \
  --teacache_thresh 0.08 \
  --task t2v-1.3B \
  --size '832*480' \
  --ckpt_dir /path/to/models/Wan2.1-T2V-1.3B \
  --prompt 'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.' \
  --base_seed 42 \
  --offload_model False \
  --save_file /all/yiran07-disk3/huteng_data/exp/teacache_wan21/teacache_0.08.mp4
```

官方代码通过 checkpoint 路径名中的 `1.3B`、`14B`、`480P` 或 `720P` 选择拟合
系数，因此 checkpoint 目录名必须保留对应标记。T2V-14B 官方报告的阈值为：

| retention steps | slow | fast |
| --- | ---: | ---: |
| 关闭 | 0.14 | 0.20 |
| 开启 | 0.10 | 0.20 |

T2V-1.3B 官方无 retention fast 阈值为 0.08，开启 retention 时 fast 阈值为 0.10。
T2V-14B 默认对比可使用官方 fast 配置 `--use_ret_steps --teacache_thresh 0.20`。
阈值不是跨模型通用超参数，不得在不同 Wan2.1 checkpoint 间混用系数或阈值。
`--teacache_thresh 0` 仅作为一致性诊断配置：它仍绑定并执行官方 TeaCache
采样函数与 `forward`，但内部比较阈值使用 `-inf` sentinel，保证每次 block 都做
full-compute，必须与 no-cache baseline 完全一致。这个零值边界是测试契约；所有
正阈值仍逐字复用官方 TeaCache 函数、判据、状态与系数。不能直接用数值 0 参与官方
拟合判据比较，因为拟合多项式在部分输入区间会输出负值并意外触发复用。

## Latency 与 TFLOPs

统一入口接受可选 `--timing_json /path/to/timing.json`。启用后记录 pipeline 初始化、
完整 pipeline generate、T5/DiT/VAE decode 的 CUDA event 与 host span，以及每次 DiT
forward 实际执行的 transformer block 数；不启用时 baseline 仍走原始直接调用路径。

完整性能冒烟命令和口径见 `experiments/threshold_zero_smoke/README.md`。TFLOPs 使用
本仓库 `CalflopsEvaluation/` 锁定的 Calflops 0.3.2：自动统计可见算子，并对自定义
FlashAttention CUDA kernel 做显式 dense-attention 公式补偿。输出同时区分理论
运算量 `TFLOPs` 和根据 DiT CUDA 时间换算的吞吐 `TFLOP/s`。正式 headline 是按实际
调用轨迹累计的 DiT TFLOPs，同时分别保存两次 UMT5 encoder forward 和一次 VAE
decode 的 TFLOPs；tokenizer、scheduler 和 MP4 导出不在这些组件计数内。

## Vbench200 与成对质量评测

完整命令见 `experiments/vbench200_t2v/README.md`。生成阶段使用锁定的
原始 Wan2.1 `generate.py` 产生 full-compute reference；candidate 通过同一个项目入口
显式启用官方 TeaCache 方法，保持相同 prompt、seed 和采样配置。评测阶段统一输出：

- `psnr_rgb_db`
- `ssim_rgb`
- `lpips_alex_v0_1_spatial`
- Vbench200 Quality、Semantic 和 Total subset scores

这些分数与 TeaCache 官方评测脚本产物属于不同、明确命名的本项目协议。
