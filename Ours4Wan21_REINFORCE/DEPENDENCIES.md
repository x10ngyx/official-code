# 资源与跨项目依赖审计

**本项目不是自包含发布包。除模型权重和环境外，也有跨项目代码、prompt资源和外部实验输入。仅复制本文件夹不能独立训练。** 本清单依据当前源码和正式8-epoch运行配置核对；机器可读SHA256见 `dependency_inventory.json`。

## 同一Git仓库内、但不在本子项目内

| 路径（相对于official-code） | 用途 |
|---|---|
| `Ours4Wan21/experiments/cnn_mixed3500_v1/{model,features}.py` | CNN架构、G1池化、输入编码 |
| `Ours4Wan21/ours4wan21/` | Exact-K、状态契约、SeaCache runtime、计量及相关特征模块；仅复用工具，不执行IQL训练 |
| `Ours4Wan21/data_collection/src/ours4wan21_data/manifest.py` | 构造默认固定prompt名单及划分 |
| `Ours4Wan21/data_collection/resources/prompts/openvidhd_balanced_5000.upstream.jsonl` | 默认prompt源 |
| `Ours4Wan21/data_collection/configs/seacache_thresholds.wan22_v1.json` | 原选样/manifest配置 |
| `SeaCache4Wan21/` | 缓存控制与Wan推理插桩 |
| `ComponentMetrics/` | T5/DiT/VAE计量、FLOPs报告及协议校验 |
| `Wan21Benchmark/protocol.py` | Wan源码兼容性检查 |
| `DiCache4Wan21/upstream_lock.json` | 上述兼容性检查间接读取的Wan源文件哈希 |
| `VideoMetrics/video_metrics/` | 训练PSNR、评测PSNR/SSIM/LPIPS |
| `VbenchEvaluation/` | 最终评测的custom-input VBench脚本 |
| `Vbench200/VBench200_full_info.json` | VBench构造器默认元数据；custom-input评测仍要求该文件存在 |

`artifacts.source_hashes()`还会锁定若干依赖目录内全部Python文件，即使某些文件不在当前执行分支内。完整仓库布局必须保留。

## 仓库外的必要或可选资源

- **Wan2.1推理源码**：默认 `/home/wangyue/work/Wan2.1`，通过 `--wan-root` 指定。这是源代码，不是模型权重；当前不随本子项目发布。需符合上述源码兼容性锁。
- **组件FLOPs profile**：默认 `/all/yiran06-disk1/wangyue_home/exp/cacheimpact_wan21_compact1000_seed42_v1/component_profile.json`，通过 `--profile` 指定。prepare必需校验，单独clone没有该文件；可用同仓库 `Wan21Benchmark/experiments/component_profile/` 生成匹配协议的profile。
- **复用baseline数据**：默认同一compact1000结果根，读取 `plan.json`、`worker_*/identity.json`、对应cell的 `COMPLETE.json` 和 `video.mp4`。不读取baseline latent。`--no-baseline-reuse` 可取消依赖，由训练临时生成native参考，但仍需有效profile。
- **自定义prompt**：`--prompts` 可替换默认prompt资源，需提供有效train/val/test划分；不应为了凑样本将val/test并入train。
- **结果与续训产物**：写入 `/all/yiran06-disk1/wangyue_home/exp`，checkpoint写入对应models目录；`experiment_results/`中的结果链接指向外部存储。新训练不需要旧REINFORCE轨迹；精确续训则需要原config、来源文件、checkpoint、轨迹与收据。

权重（包括评测模型权重）和Python/Conda环境按用户要求不列为需要纳入项目的资源。运行代码仍有本机绝对路径约束，尚未改为跨机器可直接运行的打包形式。

## 本次Git发布与本机运行版本的差异

审计基线commit：`fdb1977f7e45b5703de461f23c24990821de9e19`。本次仅提交 `Ours4Wan21_REINFORCE/`，不混入其他子项目的未提交改动。

正式运行所用以下共享依赖与该Git基线不同，**本次未提交其本地修改**：

- `SeaCache4Wan21/wan21_integration.py`
- `Wan21Benchmark/protocol.py`

因此，即使clone完整仓库，也不能声称已获得与当前本机训练完全一致的依赖版本。对应本机SHA256已记录在inventory，需另行同步这两份依赖后才能声称版本对齐；本次没有复制、替换或调整它们，避免影响在途训练。

## 本次不上传的内容

遵循仓库 `.gitignore`：实验大文件、结果symlink、checkpoint、环境、内部 `PROGRESS.md` 和 `logs/` 不提交。实验结果不因源码push自动上传。CPU测试在本机依赖环境下验证，不等于全新clone可独立运行。
