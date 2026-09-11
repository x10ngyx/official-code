# IQL剩余训练与就绪组评测并行

用户要求在SEA7 A3单卡训练时，另外三卡先评测已选定的七组。
本目录接管ours21_iql_aggressiveness_2x4_v1后续调度；保留原配置、源码锁、10prompt、
同物理GPU baseline配对、K23/29/35和无VBench分数的协议。

- `run.py`：保留正在运行的SEA7 A3训练子进程，暂停旧父调度器以阻止重复发车；GPU0–2即时开始七组评测。
  训练完成后确认子进程退出、清理旧父调度器，按原方法补选点/Q图；追加第八组，GPU3开始全八组分片。
  各卡生成完即运行本卡质量，最终调用原report.py进行240候选验收与汇总。
- `worker.py`：复用原持久Wan worker，逐视频重读可追加的jobs；已排任务身份不可改变，
  完成数量固定72/72/48/48。模型加载序列化，native预热在各自GPU并行，每卡仅加载一次。
- `dispatch.py`：增量队列和来源校验；新调度源码额外冻结于结果overlap/CONFIG.json，
  generation.json包含补充源码哈希，原实验config.json和既有训练源码不修改。
- `test_dispatch.py`：队列追加、拒绝身份替换/不完整退出、就绪分片数验收。

从Wan2.2环境运行`python experiments/iql_aggressiveness_overlap_v1/run.py`。
补充调度状态在结果overlap/，原groups/与最终REPORT.md/COMPLETE.json路径保持。
不恢复旧20prompt后处理，不重训七组，不更改任何训练或推理超参数。
