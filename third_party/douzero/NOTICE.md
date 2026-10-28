# third_party/douzero

本目录代码来自 Apache-2.0 项目，许可证全文见同目录 `LICENSE`。

| 文件 | 来源 | 修改 |
|---|---|---|
| utils.py、move_generator.py、move_detector.py、move_selector.py | kwai/DouZero@718a5c9 `douzero/env/` | 导入路径改为 `third_party.douzero`，加文件头 |
| encoding.py | kwai/DouZero@718a5c9 `douzero/env/env.py`；pumpkinjw/jjdouzeroAuto@de818a6 `Auto_JJ2/douzero/env/env.py`（Apache-2.0） | 删除 Env 类与 GameEnv 导入；ResNet 历史编码函数改名 `_action_seq_resnet`/`_process_action_seq_resnet` |
| models.py | kwai/DouZero@718a5c9 `douzero/dmc/models.py`；pumpkinjw/jjdouzeroAuto@de818a6 `Auto_JJ2/douzero/dmc/models.py` | 只保留推理所需网络类 |

这些文件保持上游写法，不适用本仓库 AGENTS.md 的中文 Docstring/类型注解要求；本仓库自有代码通过 `decision/` 调用它们。
