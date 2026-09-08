# GPU1/2/3 断点续扫队列

`queue.py --queue-dir QUEUE_RESULT_DIR` 从结果目录的 `queue_config.json` 读取既有扫描、
前置实验和锁定输入。当前依赖为 CNN e304 VBench10 的完整生成及质量评测：
`COORDINATOR_EXIT.json` 成功退出且 `HELDOUT_COMPLETE.json` 为 e304/50 cells/complete。
该实验生成之后还使用 GPU1 评测，不能只根据 GPU1 暂时空闲就抢占。

依赖成功完成后，确认 GPU1/2/3 连续三个检查周期（10秒间隔）没有计算进程、
显存≤1024MiB、利用率≤5%，再调用既有 `targeted_threshold_scan/run_gpu123.py --resume`。
不终止前置实验，不改变 R/K/E、prompt、baseline、计量或质量协议。

队列使用纯 CPU 等待，复用原始 profile 与通过校验的完整视频。中断时未完成的视频由原 runner
归档并重跑；新进程仍执行原先规定的一次 full warmup。旧 worker 状态在恢复前移入队列结果
`prior_state/`，防止旧 ready/running 标记绕过错开加载；历史停止记录同时保留于该目录。

新增入口独立记录自己的 SHA 于 `queue_config.json`，不改写已运行实验的 `upstream_lock.json`
或 prepared manifest，因此旧的77条完整结果与profile仍满足严格resume合同。
队列记录保存在原扫描结果目录下，沿用原 `experiment_results/` symlink。
