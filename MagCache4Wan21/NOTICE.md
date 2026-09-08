# Upstream notice

MagCache4Wan21运行于`Wan-Video/Wan2.1@65386b2e03c490796eede31b0325a6a595cc684e`。`wan21_integration.py`保留Alibaba Wan Team版权声明，Wan2.1采用Apache-2.0许可证。

MagCache控制逻辑与Wan2.1-T2V-1.3B官方magnitude ratio曲线来自`Zehong-Ma/MagCache@df81cb181776c2c61477c08e1d21f87fda1cd938`的`MagCache4Wan2.1/magcache_generate.py`，上游采用Apache-2.0许可证。本项目将其重构为状态隔离的控制器和显式pipeline注入层，并未复制整份上游脚本。

精确来源与SHA256记录在`upstream_lock.json`。质量评估统一调用仓库级`VideoMetrics`和`VbenchEvaluation`，不使用方法私有评估实现。
