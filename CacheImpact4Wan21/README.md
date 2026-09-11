# CacheImpact4Wan21：单步缓存的终局影响

面向远端 Wan2.1-T2V-1.3B 的独立采集子项目。其他步全部重算，仅在指定执行步复用上一步的 Transformer-stack residual；cond/uncond 两分支同步跳过、cache 独立。用于检验局部输出差异 proxy 与 latent 特征对终局影响的预测能力，**不预设 latent 一定更好**。

## 目录

- `experiments/single_skip_v1/`：计划、采集、评测、汇总入口及详细远端说明。
- `tests/`：CPU 合同与干预验证。
- `experiment_results/`：仅放指向外部实验结果的 symlink。
- `PROGRESS.md`、`logs/`：状态与交接。

所有运行使用 **conda `wan2.2`**。Wan2.1 是模型/源码名，不是环境名。
复用相邻 `SeaCache4Wan21` 的 locked forward/sampler 和滤波函数，但以独立 controller 完全替换 threshold 决策；不修改现有方法源码，不加载 Actor，不启用 Exact-K，不使用 interval/threshold。

## 顺序与数量

使用 **1-based 执行步编号**，共50步，首步无缓存不可跳过。

1. 40条全算 baseline。
2. 首轮按 `2,5,10,15,20,25,30,35,40,45,50` 排序，每步跑完40个prompt再进入下一步：440条候选。
3. 按升序补齐其余38个step：1520条候选。

共 `40×49=1960` 条候选，另40 baseline，总计2000条；首轮含baseline为480条。每次启动有额外两条排除计数的全算预热/原生一致性验证视频。
第50步是用户明确要求的干预点，因此本实验允许跳过末步；这是对缓存方法末步保护规则的实验性覆盖，不修改Wan求解器。

## 固定协议和标签

单卡48GB，batch=1，832×480、81帧、16fps；50-step UniPC，shift=5，CFG=5，seed=42；DiT BF16计算，`offload_model=False,t5_cpu=False`，所有组件驻GPU，关闭FSDP/SP和prompt扩展。

每个候选从同seed重新执行完整采样，前缀全部重算，因此上一步必然刷新cache、age=1；完整重放避免手工恢复UniPC历史引入新变量。只跳过Transformer blocks，patch/time/text/head/unpatchify、CFG、scheduler仍执行。后续在干预后的latent上全部重算。

主标签为相对于全算输出的 **pre-encode RGB MSE**：VAE输出clamp到[-1,1]后映射到[0,1]，float32 RGB原样落盘，以float64差分/求均值计算MSE。不得用MP4压缩结果替代这一敏感标签。主标签之外，保留两套质量结果：`quality.json`在原始RGB上复算；标准`video_metrics/`使用原pipeline的MP4解码与`VideoMetrics`计算PSNR/SSIM/LPIPS，PSNR沿用100dB上限。两种输入口径明确分开。VBench用导出MP4评测。

baseline和candidate均保存完整50份`latents/step_000_input.pt`到`step_049_input.pt`（CPU FP16原始输入，含干预后的全部后缀），以及50条step与100条branch记录；sigma/timestep、latent统计/SHA及baseline同step配对齐全。每个step保存cond/uncond的SEA relative-L1和未滤波的timestep-modulated-input relative-L1；后者不是经过多项式校准的完整TeaCache分数。G1特征按正式CNN的FP16→FP32、4×8×8池化、全时间均值/方差后8×8池化布局保存18432维；它是CNN输入，不是学习后的embedding，也未拼接SEA7。全算前缀下previous与cached-input latent相同，这是本实验的状态性质。

终局指标与真实输出差异均不能混入预测输入。脚本不训练预测器；后续比较应按prompt隔离训练/测试，并允许proxy非线性拟合。

## 数据集保存与发布

与`Ours4Wan21/data_collection`对齐保存结构，详见[输出数据合同](experiments/single_skip_v1/DATA_FORMAT.md)。`shared_baselines/`保存参考；`shards/shard_00/candidates/`保存候选；`completed/`保存质量完成的轨迹记录；`published/current/tables/`提供trajectory/step/branch三套JSONL+CSV。主MSE每条生成后即可查看，完整评测后再原子发布连续完成前缀。使用独立v2 schema，不冒充旧threshold/Exact-K训练数据；既有训练器的schema/末步限制需另行适配。

## 运行与交接

见 [远端运行手册](experiments/single_skip_v1/README.md)。结果强制位于
`/all/yiran07-disk3/huteng_data/exp/<run>`，自动在本项目`experiment_results/`创建链接。完整2000视频原始RGB约 **723 GiB**，50步FP16 latent另约 **391 GiB**，合计约 **1.09 TiB**（不含MP4和元数据），首轮合计约267 GiB；原始数据支持不重新生成即可复核小误差。不要把它们复制进代码仓库。

CPU验证：

```bash
conda activate wan2.2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

代码交付不代表已通过真实GPU推理；远端启动会先执行原生等价性检查，失败则在采集前停止。
