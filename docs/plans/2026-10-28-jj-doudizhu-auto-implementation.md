# JJ 斗地主自动操作助手 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 以 Han 记牌器为识别底座，构建一个在 Windows 上安全地观察、决策并闭环操作标准三人 JJ 斗地主的桌面程序。

**Architecture:** 保留 Han 记牌器现有的 Qt 主进程、推理子进程、截图、YOLO、布局、Tracker 和诊断边界；新增纯逻辑自动化状态机、DouZero 决策适配器、按钮识别、选牌规划和可注入的 Windows 输入后端。任何检测结果都必须经过稳定确认和动作合法性校验；任何鼠标操作都必须经过执行前检查与执行后画面验证。

**Tech Stack:** Python 3.11/3.12、PySide6、Pillow/OpenCV、ONNX Runtime、Ultralytics YOLO、DouZero、Win32 API、`unittest`、GitHub Actions、PyInstaller。

---

## 实施原则

- 全程遵循 `@test-driven-development`：行为代码先写失败测试，确认红灯后再写最小实现。
- 不运行临时目录中的 `.pt`、`.pkl` 或 `.ckpt`；只计算哈希和读取文件元数据。
- 不复制 `jjdouzeroAuto` 的主循环、模板图片或自动点击实现，只把它作为行为参考。
- DouZero 代码只从 `kwai/DouZero` 的 Apache-2.0 上游取得，并记录准确提交。
- Mac 完成纯逻辑、回放和 UI 测试；Win32 输入和真实客户端只在 Windows 观察模式通过后启用。
- 每个任务单独提交；提交前运行该任务测试及完整回归。

### Task 1: 导入 Han 源码快照并建立来源记录

**Files:**
- Import: `main.py`, `AGENTS.md`, `config/`, `core/`, `ui/`, `utils/`, `tests/`, `docs/`
- Import: `README.md`, `README.en.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `requirements.txt`
- Create: `LICENSE`
- Create: `THIRD_PARTY.md`
- Modify: `.gitignore`

**Step 1: 机械导入文本快照**

从用户提供的 `r2/src` 复制文本源文件，不复制 `.github/FUNDING.yml`、个人配置、诊断数据、图片、权重或缓存。保留现有测试与文档，不在导入提交中修改行为。

**Step 2: 恢复许可证并记录来源**

从 `hey-three/han-card-tracker` 官方仓库取得 MIT `LICENSE`，在 `THIRD_PARTY.md` 中记录来源仓库、取得日期、快照文件清单哈希以及“当前快照不是完整 Git 克隆”。记录缺失的两份 Python 规范文件，不伪造其内容。

**Step 3: 运行静态检查和基线测试**

Run:

```bash
python -m compileall -q config core ui utils tests
QT_QPA_PLATFORM=offscreen python -X faulthandler -B -m unittest discover -s tests -v
```

Expected: 编译成功；测试结果被原样记录。若因缺少平台依赖失败，停止并在 Task 2 修复开发环境，不把失败宣称为源码缺陷。

**Step 4: Commit**

```bash
git add .
git commit -m "chore: import audited Han tracker source snapshot"
```

### Task 2: 建立 Mac 开发与 Windows 运行的依赖边界

**Files:**
- Create: `tests/test_dependency_policy.py`
- Create: `requirements/base.txt`
- Create: `requirements/windows.txt`
- Create: `requirements/dev.txt`
- Modify: `requirements.txt`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
class DependencyPolicyTests(unittest.TestCase):
    def test_win32_dependency_is_windows_only(self) -> None:
        text = Path("requirements/windows.txt").read_text(encoding="utf-8")
        self.assertIn("pywin32", text)
        self.assertNotIn("pywin32", Path("requirements/base.txt").read_text())

    def test_runtime_does_not_require_cuda_build_on_mac(self) -> None:
        base = Path("requirements/base.txt").read_text(encoding="utf-8")
        self.assertNotIn("+cu121", base)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_dependency_policy -v`

Expected: FAIL because split requirement files do not exist.

**Step 3: Write minimal configuration**

把 UI、图像和配置依赖放入 `base.txt`，把 `pywin32`、Windows 打包依赖放入 `windows.txt`，把测试工具放入 `dev.txt`。PyTorch/ONNX 的 CPU 与 CUDA 安装命令写入 README，不把 CUDA wheel 强塞给 Mac。

**Step 4: Verify**

Run:

```bash
python -m unittest tests.test_dependency_policy -v
QT_QPA_PLATFORM=offscreen python -X faulthandler -B -m unittest discover -s tests -v
```

Expected: dependency policy PASS；可安装依赖后完整基线测试 PASS。

**Step 5: Commit**

```bash
git add requirements requirements.txt README.md tests/test_dependency_policy.py
git commit -m "build: separate mac development and windows runtime dependencies"
```

### Task 3: 建立模型清单与安全加载策略

**Files:**
- Create: `security/model-manifest.json`
- Create: `core/model_assets.py`
- Create: `tests/test_model_assets.py`
- Modify: `core/yolo_inferencer.py`
- Modify: `config/settings.py`

**Step 1: Write the failing tests**

```python
class ModelAssetTests(unittest.TestCase):
    def test_hash_mismatch_rejects_model(self) -> None:
        with self.assertRaises(ModelIntegrityError):
            verify_model(self.path, "0" * 64)

    def test_runtime_policy_rejects_unapproved_pt(self) -> None:
        with self.assertRaises(UnsafeModelFormatError):
            resolve_runtime_model("unknown.pt", manifest={})

    def test_approved_onnx_is_selected_without_pt_fallback(self) -> None:
        selected = resolve_runtime_model("cards.onnx", manifest=self.manifest)
        self.assertEqual(selected.format, "onnx")
```

**Step 2: Verify RED**

Run: `python -m unittest tests.test_model_assets -v`

Expected: FAIL because `core.model_assets` does not exist.

**Step 3: Implement the minimal policy**

实现流式 SHA-256、清单解析、格式白名单和清晰异常。正式 Windows 模式只接受清单中已批准的 ONNX/engine；`.pt` 只能在显式的隔离转换命令中使用，不能静默回退。

清单记录当前三个未批准样本的哈希，但不把文件复制进仓库：

- `m.pt`: `a29f4f41a8a7b6e6a3544e93f2506a2b2f45b94bcf6fd301a6440ddee4567d56`
- `n.pt`: `859cb27960221cc4a4de7c8b048a32b5e93d1792ae42b597eaeab76601ca052e`
- `s.pt`: `e2419bb59102168fde02324fe6981a583e4300c524645897c967f521605397b7`

**Step 4: Verify GREEN**

Run:

```bash
python -m unittest tests.test_model_assets -v
python -m unittest tests.test_inference_sizes -v
```

Expected: PASS，且现有尺寸回归不受影响。

**Step 5: Commit**

```bash
git add security/model-manifest.json core/model_assets.py core/yolo_inferencer.py config/settings.py tests/test_model_assets.py
git commit -m "security: require verified runtime model artifacts"
```

### Task 4: 定义自动化领域类型和状态机

**Files:**
- Create: `automation/__init__.py`
- Create: `automation/types.py`
- Create: `automation/state_machine.py`
- Create: `tests/test_automation_state_machine.py`

**Step 1: Write the failing tests**

```python
class AutomationStateMachineTests(unittest.TestCase):
    def test_starts_disabled_and_never_emits_action(self) -> None:
        machine = AutomationStateMachine()
        self.assertEqual(machine.mode, AutomationMode.DISABLED)
        self.assertIsNone(machine.accept(valid_play_observation()))

    def test_unknown_dialog_forces_paused_state(self) -> None:
        machine = enabled_machine()
        transition = machine.accept(valid_play_observation(unknown_dialog=True))
        self.assertEqual(machine.phase, AutomationPhase.PAUSED)
        self.assertEqual(transition.reason, "unknown_dialog")

    def test_bidding_action_is_rejected_during_play_phase(self) -> None:
        with self.assertRaises(IllegalAutomationAction):
            machine.plan(BidAction(score=1), valid_play_observation())
```

**Step 2: Verify RED**

Run: `python -m unittest tests.test_automation_state_machine -v`

Expected: FAIL because package does not exist.

**Step 3: Implement minimal types and transitions**

定义 `DISABLED`、`OBSERVE`、`AUTO` 三种模式，以及 `WAITING`、`BIDDING`、`PLAYING`、`VERIFYING`、`PAUSED`、`FINISHED` 阶段。所有 dataclass 使用不可变字段，动作与观察分离。

**Step 4: Verify GREEN and refactor**

Run: `python -m unittest tests.test_automation_state_machine -v`

Expected: PASS。

**Step 5: Commit**

```bash
git add automation tests/test_automation_state_machine.py
git commit -m "feat: add fail-closed automation state machine"
```

### Task 5: 增加连续帧稳定门和安全策略

**Files:**
- Create: `automation/safety.py`
- Create: `tests/test_automation_safety.py`
- Modify: `automation/state_machine.py`

**Step 1: Write the failing tests**

```python
def test_two_disagreeing_frames_never_unlock_action() -> None:
    gate = StableObservationGate(required_frames=3)
    self.assertFalse(gate.observe(frame(hand="345")))
    self.assertFalse(gate.observe(frame(hand="346")))
    self.assertFalse(gate.observe(frame(hand="345")))

def test_window_geometry_change_pauses_even_with_same_cards() -> None:
    policy = SafetyPolicy()
    result = policy.evaluate(previous=frame(rect=(0, 0, 1280, 720)), current=frame(rect=(0, 0, 1600, 900)))
    self.assertEqual(result.pause_reason, "window_geometry_changed")
```

**Step 2: Verify RED**

Run: `python -m unittest tests.test_automation_safety -v`

Expected: FAIL because safety module is missing.

**Step 3: Implement minimal gate**

比较牌局阶段、手牌多重集、上一手、按钮集合、窗口矩形和遮挡标记。未知区域不能当作空区域；窗口变化、未知弹窗或截图失败直接返回暂停原因。

**Step 4: Verify GREEN**

Run: `python -m unittest tests.test_automation_safety -v`

Expected: PASS。

**Step 5: Commit**

```bash
git add automation/safety.py automation/state_machine.py tests/test_automation_safety.py
git commit -m "feat: gate automation on stable verified observations"
```

### Task 6: 实现基于真实牌框的选牌规划

**Files:**
- Create: `automation/card_selection.py`
- Create: `tests/test_card_selection.py`

**Step 1: Write the failing tests**

```python
def test_duplicate_rank_selects_distinct_boxes() -> None:
    plan = plan_card_clicks(action=["A", "A"], detections=three_aces())
    self.assertEqual(len(plan.clicks), 2)
    self.assertEqual(len({click.detection_id for click in plan.clicks}), 2)

def test_missing_rank_rejects_entire_plan() -> None:
    with self.assertRaises(UnplannableAction):
        plan_card_clicks(action=["D", "X"], detections=only_small_joker())

def test_plan_uses_window_relative_centers() -> None:
    plan = plan_card_clicks(action=["3"], detections=[box("3", 100, 400, 140, 500)])
    self.assertEqual(plan.clicks[0].point, (120, 450))
```

**Step 2: Verify RED**

Run: `python -m unittest tests.test_card_selection -v`

Expected: FAIL because planner is missing.

**Step 3: Implement minimal pure planner**

按牌点分组并稳定排序，选择互不重复的检测框；规划只生成窗口相对点击点，不发送输入。目标牌不完整、检测框越界或置信度不足时整体失败。

**Step 4: Verify GREEN**

Run: `python -m unittest tests.test_card_selection -v`

Expected: PASS。

**Step 5: Commit**

```bash
git add automation/card_selection.py tests/test_card_selection.py
git commit -m "feat: plan card selection from current detection boxes"
```

### Task 7: 建立可配置的 JJ 按钮识别层

**Files:**
- Create: `automation/button_detector.py`
- Create: `config/jj_buttons.yaml`
- Create: `tests/test_button_detector.py`
- Create: `tests/fixtures/buttons/README.md`

**Step 1: Write synthetic failing tests**

```python
def test_detector_returns_window_relative_button_box() -> None:
    detector = ButtonDetector({"play": template_image()})
    result = detector.detect(screen_with_template_at(200, 300))
    self.assertEqual(result["play"].center, (220, 315))

def test_ambiguous_matches_are_not_actionable() -> None:
    detector = ButtonDetector({"play": template_image()}, ambiguity_margin=0.02)
    self.assertFalse(detector.detect(screen_with_two_equal_templates())["play"].actionable)
```

**Step 2: Verify RED**

Run: `python -m unittest tests.test_button_detector -v`

Expected: FAIL because detector is missing.

**Step 3: Implement minimal detector**

使用 OpenCV 模板匹配和限定搜索区域，输出置信度、边框和可操作性。不复制旧仓库图片；真实模板必须从用户自己的 JJ 截图裁剪并脱敏后加入。

**Step 4: Verify GREEN**

Run: `python -m unittest tests.test_button_detector -v`

Expected: PASS with synthetic fixtures。

**Step 5: Commit**

```bash
git add automation/button_detector.py config/jj_buttons.yaml tests/test_button_detector.py tests/fixtures/buttons/README.md
git commit -m "feat: add ambiguity-aware JJ button detection"
```

### Task 8: 接入 Apache-2.0 DouZero 决策适配器

**Files:**
- Create: `decision/__init__.py`
- Create: `decision/protocol.py`
- Create: `decision/card_codec.py`
- Create: `decision/douzero_adapter.py`
- Create: `tests/test_decision_adapter.py`
- Create: `third_party/douzero/NOTICE.md`
- Modify: `THIRD_PARTY.md`

**Step 1: Write the failing adapter tests**

```python
def test_adapter_rejects_action_not_in_legal_actions() -> None:
    backend = FakeBackend(action=[30])
    adapter = DouZeroAdapter(backend)
    with self.assertRaises(IllegalDecision):
        adapter.decide(state_with_legal_actions([[3], [4]]))

def test_codec_round_trips_jokers_and_ten() -> None:
    cards = ["10", "J", "小", "大"]
    self.assertEqual(decode_cards(encode_cards(cards)), cards)
```

**Step 2: Verify RED**

Run: `python -m unittest tests.test_decision_adapter -v`

Expected: FAIL because decision package is missing.

**Step 3: Implement protocol, codec and fake-backed adapter**

先定义纯接口和合法动作校验，使测试不需要加载模型。随后从 `kwai/DouZero` 固定提交导入最小推理依赖，保留 Apache-2.0 LICENSE/NOTICE 和修改说明。不要从 `jjdouzeroAuto` 复制实现。

**Step 4: Add model-free integration test**

用确定性 fake backend 验证地主、地主上家、地主下家状态编码及 pass 动作。真实权重测试单独标记为 Windows/asset acceptance，不进入普通单测。

**Step 5: Verify GREEN**

Run: `python -m unittest tests.test_decision_adapter -v`

Expected: PASS without loading any checkpoint。

**Step 6: Commit**

```bash
git add decision third_party/douzero THIRD_PARTY.md tests/test_decision_adapter.py
git commit -m "feat: add licensed DouZero decision boundary"
```

### Task 9: 实现可注入、默认禁用的 Windows 输入后端

**Files:**
- Create: `automation/input_backend.py`
- Create: `automation/windows_input.py`
- Create: `automation/emergency_stop.py`
- Create: `tests/test_input_backend.py`

**Step 1: Write the failing tests**

```python
def test_observe_mode_records_but_never_clicks() -> None:
    native = FakeNativeInput()
    backend = GuardedInputBackend(native)
    backend.execute(plan(), mode=AutomationMode.OBSERVE)
    self.assertEqual(native.clicks, [])

def test_emergency_stop_blocks_remaining_clicks() -> None:
    stop = EmergencyStop()
    native = FakeNativeInput(on_first_click=stop.trigger)
    with self.assertRaises(AutomationStopped):
        GuardedInputBackend(native, stop).execute(two_click_plan(), AutomationMode.AUTO)
    self.assertEqual(len(native.clicks), 1)
```

**Step 2: Verify RED**

Run: `python -m unittest tests.test_input_backend -v`

Expected: FAIL because backend is missing.

**Step 3: Implement minimal guarded backend**

纯接口负责模式、停止标志、窗口句柄和坐标边界检查。`windows_input.py` 才延迟导入 Win32，并使用目标窗口客户区到屏幕坐标转换。非 Windows 导入模块不能产生副作用。

**Step 4: Verify GREEN on Mac**

Run: `python -m unittest tests.test_input_backend -v`

Expected: PASS using fake native input and zero real clicks。

**Step 5: Commit**

```bash
git add automation/input_backend.py automation/windows_input.py automation/emergency_stop.py tests/test_input_backend.py
git commit -m "feat: add guarded windows input backend"
```

### Task 10: 实现闭环自动化协调器

**Files:**
- Create: `automation/coordinator.py`
- Create: `tests/test_automation_coordinator.py`
- Modify: `automation/state_machine.py`

**Step 1: Write the failing tests**

```python
def test_click_is_not_committed_until_post_action_observation_matches() -> None:
    coordinator = coordinator_with_fakes()
    coordinator.handle(pre_action_observation())
    self.assertEqual(coordinator.committed_actions, [])
    coordinator.handle(expected_post_action_observation())
    self.assertEqual(coordinator.committed_actions, [play_three()])

def test_verification_timeout_pauses_without_retry_storm() -> None:
    coordinator = coordinator_with_fakes(max_corrections=1)
    coordinator.handle(pre_action_observation())
    coordinator.handle(unrelated_observation(), now=timeout_time())
    self.assertEqual(coordinator.phase, AutomationPhase.PAUSED)
    self.assertEqual(coordinator.input_backend.execution_count, 1)
```

**Step 2: Verify RED**

Run: `python -m unittest tests.test_automation_coordinator -v`

Expected: FAIL because coordinator is missing.

**Step 3: Implement minimal closed loop**

协调器依次调用稳定门、决策适配器、选牌规划、按钮检测、输入后端和执行后验证。所有依赖通过构造函数注入；未知状态、超时和非法动作统一转为可诊断暂停。

**Step 4: Verify GREEN**

Run: `python -m unittest tests.test_automation_coordinator -v`

Expected: PASS。

**Step 5: Commit**

```bash
git add automation/coordinator.py automation/state_machine.py tests/test_automation_coordinator.py
git commit -m "feat: coordinate decisions with post-click verification"
```

### Task 11: 接入现有进程协议和 UI

**Files:**
- Modify: `core/inference_protocol.py`
- Modify: `core/controller_process.py`
- Modify: `core/inference_process.py`
- Modify: `core/game_controller.py`
- Modify: `ui/main_window.py`
- Modify: `ui/settings_controller.py`
- Modify: `config/settings.py`
- Modify: `config/config.yaml`
- Create: `tests/test_automation_protocol.py`
- Modify: `tests/test_ui.py`
- Modify: `tests/test_worker.py`

**Step 1: Write failing protocol tests**

验证 `set_automation_mode` 使用 request/generation 回执，旧代次不得重新开启自动模式，推理进程故障必须将模式降为 `DISABLED`。

**Step 2: Verify RED**

Run: `python -m unittest tests.test_automation_protocol tests.test_worker -v`

Expected: FAIL because automation commands are unknown.

**Step 3: Implement protocol and controller wiring**

在现有 Command/Response 协议中增加最小字段，保持 UI 不直接操作 Win32。自动化状态作为独立响应字段返回，不能用 `finished` 冒充控制回执。

**Step 4: Write failing UI tests**

验证默认关闭、观察模式可选、自动模式需要明确确认、故障/重启后自动关闭、紧急停止始终可见。

**Step 5: Implement minimal UI**

新增模式控件和状态标签，不重新设计整个界面。任何持久化设置都必须等待后端成功回执；自动模式本身不跨程序启动保持开启。

**Step 6: Verify GREEN**

Run:

```bash
QT_QPA_PLATFORM=offscreen python -m unittest tests.test_automation_protocol tests.test_worker tests.test_ui -v
```

Expected: PASS。

**Step 7: Commit**

```bash
git add core ui config tests/test_automation_protocol.py tests/test_worker.py tests/test_ui.py
git commit -m "feat: expose observe and automatic modes through existing protocol"
```

### Task 12: 扩展诊断证据并默认脱敏

**Files:**
- Modify: `core/diagnostics.py`
- Modify: `core/diagnostic_state.py`
- Modify: `core/diagnostic_storage.py`
- Create: `tests/test_automation_diagnostics.py`
- Modify: `docs/诊断记录.md`

**Step 1: Write the failing tests**

验证诊断包含观察摘要、决策、动作计划、点击坐标、验证结果和暂停原因；不包含窗口标题之外的账户文本、Cookie、令牌或任意桌面区域。

**Step 2: Verify RED**

Run: `python -m unittest tests.test_automation_diagnostics -v`

Expected: FAIL because automation evidence schema is missing.

**Step 3: Implement immutable diagnostic snapshots**

只保存目标 JJ 窗口及结构化元数据，沿用现有内存/磁盘配额和后台写线程。自动化诊断故障不得改变识牌和安全暂停结果。

**Step 4: Verify GREEN and regression**

Run:

```bash
python -m unittest tests.test_automation_diagnostics tests.test_diagnostics tests.test_diagnostic_rounds -v
```

Expected: PASS。

**Step 5: Commit**

```bash
git add core/diagnostics.py core/diagnostic_state.py core/diagnostic_storage.py tests/test_automation_diagnostics.py docs/诊断记录.md
git commit -m "feat: record privacy-bounded automation evidence"
```

### Task 13: 增加 Windows CI、构建和发布清单

**Files:**
- Create: `.github/workflows/test.yml`
- Create: `.github/workflows/windows-build.yml`
- Create: `packaging/jj-auto.spec`
- Create: `scripts/generate_release_manifest.py`
- Create: `tests/test_release_manifest.py`
- Create: `SECURITY.md`
- Modify: `README.md`
- Modify: `THIRD_PARTY.md`

**Step 1: Write the failing manifest test**

```python
def test_manifest_contains_every_distributed_model_hash() -> None:
    manifest = build_manifest(self.release_dir)
    self.assertEqual(manifest["models"][0]["sha256"], sha256(self.onnx_path))
    self.assertIn("source_commit", manifest)
    self.assertIn("third_party_licenses", manifest)
```

**Step 2: Verify RED**

Run: `python -m unittest tests.test_release_manifest -v`

Expected: FAIL because script is missing.

**Step 3: Implement deterministic manifest generation**

清单包括源码提交、Python/依赖版本、模型哈希、发布文件哈希、测试命令和构建时间。构建工作流先运行完整测试，再打包；不从临时网址下载未经哈希验证的模型。

**Step 4: Verify GREEN**

Run:

```bash
python -m unittest tests.test_release_manifest -v
python -X faulthandler -B -m unittest discover -s tests -v
```

Expected: all tests PASS。

**Step 5: Commit**

```bash
git add .github packaging scripts tests/test_release_manifest.py SECURITY.md README.md THIRD_PARTY.md
git commit -m "build: add reproducible windows release evidence"
```

### Task 14: Windows 观察模式和自动模式验收

**Files:**
- Create: `docs/windows-acceptance.md`
- Create: `tests/replay/README.md`
- Modify: `docs/排查与验证.md`
- Modify: `CHANGELOG.md`

**Step 1: Prepare acceptance assets**

从用户 Windows JJ 客户端收集脱敏的标准三人斗地主截图或诊断包，覆盖三个角色、叫分、抢地主、出牌、不出、一局结束、窗口缩放和未知弹窗。任何真实账号区域在进入仓库前裁剪或打码。

**Step 2: Run observation mode**

连续运行多局，只显示识别结果、DouZero 动作和计划点击位置。记录所有误识别、非法动作和暂停原因，不产生鼠标输入。

Expected: 零真实点击；未知或矛盾状态全部 fail closed。

**Step 3: Run guarded automatic mode**

在低风险环境开启自动模式，逐步验证叫牌、选牌、出牌、不出、执行后确认和紧急停止。

Expected: 每个已执行动作都有执行前稳定证据和执行后确认；任何失败均暂停且不会产生重试风暴。

**Step 4: Complete security review**

Run:

```bash
python -X faulthandler -B -m unittest discover -s tests -v
python -m compileall -q automation config core decision ui utils scripts tests
git diff --check main...HEAD
```

检查默认无网络、无遥测、无自动更新、无未知模型加载、无秘密文件和无未脱敏截图。

**Step 5: Commit acceptance evidence**

```bash
git add docs/windows-acceptance.md docs/排查与验证.md tests/replay/README.md CHANGELOG.md
git commit -m "docs: record windows automation acceptance evidence"
```

## 完成标准

- Mac 纯逻辑、回放和 UI 测试全部通过；
- Windows CI 和打包测试全部通过；
- 运行时只接受清单中的已验证模型；
- 观察模式不产生任何输入；
- 自动模式默认关闭，紧急停止独立有效；
- 所有动作均经过合法性校验和执行后验证；
- Windows 真实 JJ 客户端验收证据明确区分已验证与未验证项；
- Release 附源码提交、模型/文件哈希、依赖、许可证和构建日志。

