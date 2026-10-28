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

## 已批准的运行模型（不入库，由 scripts/fetch_models.py 下载并校验）

| 文件 | 来源（固定提交） | SHA-256 |
|---|---|---|
| `yolov11n_imgsz_960/best.onnx` | hey-three/han-card-tracker@`d24add3` | `85e068d7…e8983e` |
| `yolov11s_imgsz_960/best.onnx` | 同上 | `ead40e45…d36a41` |
| `yolov11m_imgsz_960/best.onnx` | 同上 | `fbca91e8…3d7e2` |
| `resnet/resnet_landlord.ckpt` | pumpkinjw/jjdouzeroAuto@`de818a6` Auto_JJ2/baselines/resnet | `b025086c…047050` |
| `resnet/resnet_landlord_down.ckpt` | 同上 | `9de5bbff…571c40` |
| `resnet/resnet_landlord_up.ckpt` | 同上 | `2d577f09…f87b` |

完整哈希见 `security/model-manifest.json`。三个 ONNX 与上表未批准 `.pt` 来自同一上游提交（`.pt` 哈希一致），ONNX 不含可执行 pickle。三个 `.ckpt` 用 `pickletools` 静态检查，只引用 `collections.OrderedDict`、`torch._utils._rebuild_tensor_v2` 与张量存储类型；运行时仍强制 `torch.load(weights_only=True)`，并以严格参数名/形状载入网络。ResNet 权重随 jjdouzeroAuto 的 `Auto_JJ2/LICENSE`（Apache-2.0）发布，其训练来源为 Douzero_Resnet 系列；jjdouzeroAuto 顶层为 GPL-3.0，本项目不分发这些权重，只在使用者本机下载。

## jjdouzeroAuto

- 参考仓库：<https://github.com/pumpkinjw/jjdouzeroAuto>（核对提交 `de818a607d3ee9070b6256fd51d23a6ccf1f2414`）
- 用途：理解 JJ 界面按钮与流程；不复制其 GPL-3.0 主循环、硬编码坐标、窗口点击实现或图片。
- 引入的代码：仅 `Auto_JJ2/douzero` 中随 Apache-2.0 LICENSE 发布的 ResNet 特征编码 `_get_obs_resnet` 及历史编码、`BasicBlock`/`ResnetModel` 网络定义，见 `third_party/douzero/encoding.py`、`models.py` 文件头。
- 按钮图片：`utils/button_templates.py import-jjdouzero` 只在使用者本机从其自己的克隆复制到被忽略的 `config/jj_buttons/`，不进入本仓库。

## DouZero

- 上游：<https://github.com/kwai/DouZero>，提交 `718a5c920bf3361e34178a38f3b80458e176b351`
- 许可证：Apache-2.0，副本见 `third_party/douzero/LICENSE`
- 引入：`utils.py`、`move_generator.py`、`move_detector.py`、`move_selector.py`（只改导入路径并加来源说明）；`env.py` 中的 `get_obs` 及其辅助函数、`dmc/models.py` 中的 `LandlordLstmModel`/`FarmerLstmModel`（原样，删除 Env/训练包装类）。修改记录见各文件头与 `third_party/douzero/NOTICE.md`。
- `decision/rules.py` 的合法动作生成按 DouZero `GameEnv.get_legal_card_play_actions` 的规则重写。
- 原版 LSTM 权重需使用者从 DouZero README 所列网盘自行下载并登记哈希后才能使用。

