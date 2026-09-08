# Wan21Benchmark

仅供新加入的 `MagCache4Wan21`、`DiCache4Wan21`、`TaylorSeer4Wan21` 使用的本地运行适配器；已有方法及公共计量/质量代码保持原样。

- `protocol.py`：核对锁定Wan21源码、模型/结果路径；在计时前把T5、DiT置于GPU，VAE沿用原生GPU初始化。
- `generation.py`：单GPU持久pipeline，逐prompt重置缓存、重新安装计时器；一次原生full warmup不进入测量；完整generate计时不含MP4导出。
- `experiments/component_profile/profile_calflops.py`：复用本地Wan21的Calflops及dense attention补偿口径，分别profile DiT、T5、VAE。
- `metrics.py`：配对性能汇总；调用现有VideoMetrics与VbenchEvaluation完成PSNR/SSIM/LPIPS和16维VBench200。
- `tests/`：resident协议与实际执行路径计量测试。

环境固定conda `wan2.2`，所有入口显式设置四项BLAS线程为1。推理为Wan2.1-T2V-1.3B，832×480、81帧、16fps、50-step UniPC、shift5、CFG5、seed42；BF16计算、单张48GB卡、offload=False、t5_cpu=False。BF16指与官方一致的autocast计算，不擅自转换原生checkpoint存储dtype。

在 `work/offical-code` 下运行（`WAN21_ROOT` 为锁定提交 `65386b2e03c490796eede31b0325a6a595cc684e` 的官方源码目录）：

```bash
conda activate wan2.2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
python Wan21Benchmark/generation.py --method dicache --baseline \
  --wan21-root "$WAN21_ROOT" --prompts Vbench200/prompts.jsonl \
  --output-dir /all/yiran07-disk3/huteng_data/exp/wan21_baseline
python DiCache4Wan21/experiments/fixed_protocol/run.py \
  --wan21-root "$WAN21_ROOT" --prompts Vbench200/prompts.jsonl --threshold 0.08 \
  --output-dir /all/yiran07-disk3/huteng_data/exp/wan21_dicache
python Wan21Benchmark/experiments/component_profile/profile_calflops.py --wan21-root "$WAN21_ROOT" \
  --checkpoint-dir ../../models/Wan2.1-T2V-1.3B \
  --output /all/yiran07-disk3/huteng_data/exp/wan21_profile.json
python Wan21Benchmark/metrics.py summarize \
  --baseline-dir /all/yiran07-disk3/huteng_data/exp/wan21_baseline \
  --candidate-dir /all/yiran07-disk3/huteng_data/exp/wan21_dicache \
  --profile /all/yiran07-disk3/huteng_data/exp/wan21_profile.json
python Wan21Benchmark/metrics.py evaluate \
  --baseline-dir /all/yiran07-disk3/huteng_data/exp/wan21_baseline \
  --candidate-dir /all/yiran07-disk3/huteng_data/exp/wan21_dicache
```

模型默认在工作区`models/Wan2.1-T2V-1.3B`，可用`--checkpoint-dir`指定models下路径。单prompt检查用`--prompt TEXT`替代`--prompts`；标准VBench评测应使用完整固定200 prompts，上述评测入口会拒绝缺失集合。Calflops和质量权重沿用本地公共工具安装要求。

每次生成使用新目录；不自动续跑、不混用远端offload结果或旧代码trace。`COMPLETE.json`只标记生成完成，质量完成另有`evaluation/COMPLETE.json`。相同checkpoint/协议/prompt集合/GPU型号的本地baseline可配对给三个新方法。多进程卡间并发、阈值搜索与历史服务器调度器不在该入口中。

TFLOPs为估算运算量，非吞吐率。DiCache reuse计入1个probe；MagCache reuse仍计always-on路径。TaylorSeer额外计入三模块差分及缓存预测/残差算术：N=seq_len×hidden_dim，D=hidden_dim，每层full更新增加6N，order0 reuse为5N+6D，order1 reuse为11N+6D。内存复制不是FLOPs；cache gate/DCTA不纳入headline，与本地既有边界一致。T5、VAE另列；完整generate latency包含缓存管理开销。

本次交付仅做CPU测试及源码核验，未执行48GB全模型生成、profile或质量模型。TaylorSeer保留远端一阶eager算法；实际显存容量与速度须以固定协议完整运行结果为准。
