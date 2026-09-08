# DiCache VBench200：GPU1/2/3复用baseline

用户明确指定的配置（标签表示目标，不是已验证加速比）：

| GPU | 目标标签 | threshold | retention | probe |
|---|---:|---:|---:|---:|
| 1 | 1.8× | .075 | .2 | 1 |
| 2 | 2.4× | .072 | .2 | 1 |
| 3 | 3.0× | .396 | .2 | 1 |

中档.072按用户原值执行，未擅自改成.172或扫描的.175；.072比低档.075小，不能预先宣称能达到2.4×。
固定832×480/45帧/16fps、50-step DPM++、shift12、CFG(3,4)、boundary.875、seed42、BF16、offload_model=True、t5_cpu=False、无FSDP/SP及prompt扩展。

`audit_baseline.py`先验证共享200视频和原始timing SHA/协议，并将扫描四轮的44条baseline与相同prompt的共享视频逐字节比较。
审计通过后，`run_gpu123.sh`每卡持久一套pipeline、一次不计入报告的full warmup、生成200个候选；三卡错峰加载，避免并发模型加载高峰。
共600个候选、0个新增测量baseline。每档videos/baseline只保存共享视频symlink。
扫描原生baseline、共享baseline逐字节相同不意味着时间相同；报告统一采用共享baseline已有reported_timings，明确保留GPU0健康卡均值修正来源。
分项T5/DiT/VAE时间和TFLOPs照常记录，DiT的reuse包括一个probe block；最终PSNR/SSIM/LPIPS和标准16维VBench自动评测。

```bash
conda activate wan2.2
export WAN22_PYTHON="$CONDA_PREFIX/bin/python"
python experiments/vbench200_reuse_gpu123/audit_baseline.py \
  --output-dir /all/yiran07-disk3/huteng_data/exp/dicache4wan22_baseline_reuse_audit_20260908
bash experiments/vbench200_reuse_gpu123/run_gpu123.sh \
  --output-dir /all/yiran07-disk3/huteng_data/exp/UNIQUE_DICACHE_VBENCH200 --dry-run
```

去掉dry-run开始；已有结果恢复用原命令加`--resume`。`--target-only 1p8x/2p4x/3p0x`可恢复单档。
默认复用已完成扫描round_00的真实形状DiCache profile；缓存核心源码41文件锁保持不变，新增实验入口在各plan的experiment_source内独立SHA锁定，不修改旧profile来源。
父级COMPLETE要求三档report提交；各档GENERATION_COMPLETE不代表质量完成。目标加速比不用于伪造实测值。
`run_gpu123.py`改编自本仓库MagCache对应runner，移除其K、静态schedule和等价参数逻辑，沿用DiCache逐调用trace校验。
