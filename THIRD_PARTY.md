# 第三方来源与安全记录

## Han Card Tracker

- 上游：<https://github.com/hey-three/han-card-tracker>
- 许可证：MIT，见根目录 `LICENSE`
- 导入日期：2026-10-28
- 来源形式：用户提供的文本源码快照，不是完整 Git 克隆
- 快照文件数：93（包含未导入的支持文件，不包含 `.github/FUNDING.yml`）
- 快照清单聚合 SHA-256：`3cd99ceb5eb30a1111783d0c373b6950f1563aedd67beb0522f38de2cb418d57`
- 已排除：权重、图片、ONNX、Git 历史、个人配置、诊断数据及赞赏配置
- 已知缺口：快照未包含 `AGENTS.md` 引用的 `Python 代码注释规范.md` 和 `Python 类型注解规范.md`；本项目不伪造这两份上游文件，新增代码遵循现有 `AGENTS.md` 与 `AI开发标准/开发总原则.md`

由于快照没有 `.git`，无法声明准确的上游提交。发布前必须将本记录与公开上游重新核对。

## 未批准的 YOLO 样本

用户提供了三个改名后的 PyTorch ZIP/pickle 容器。它们只做了文件元数据和 SHA-256 检查，没有被 Python、PyTorch 或 Ultralytics 加载，也不会进入发布包：

| 临时名称 | 大小约值 | SHA-256 | 状态 |
|---|---:|---|---|
| `n.pt` | 5.3 MiB | `859cb27960221cc4a4de7c8b048a32b5e93d1792ae42b597eaeab76601ca052e` | 未批准 |
| `s.pt` | 18 MiB | `e2419bb59102168fde02324fe6981a583e4300c524645897c967f521605397b7` | 未批准 |
| `m.pt` | 39 MiB | `a29f4f41a8a7b6e6a3544e93f2506a2b2f45b94bcf6fd301a6440ddee4567d56` | 未批准 |

运行时不得自动加载或回退到这些文件。只有在来源和许可证核验后，才能在隔离环境中转换成经过哈希登记的 ONNX 资产。

## jjdouzeroAuto

- 参考仓库：<https://github.com/pumpkinjw/jjdouzeroAuto>
- 用途：仅用于理解旧版 JJ 界面自动化行为
- 导入策略：不复制其主循环、模板图片、权重或窗口点击实现

## DouZero

- 计划上游：<https://github.com/kwai/DouZero>
- 许可证：Apache-2.0
- 当前状态：尚未导入；实际导入时必须补充准确提交、许可证和修改记录

