"""主窗口中的观察/自动模式控件：模式选择、紧急停止和状态显示。"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("disabled", "AI：关闭"),
    ("observe", "AI：只给建议"),
    ("auto", "AI：自动操作"),
)
# 缺少时自动出牌无法工作的模板；其余模板只影响叫抢、加倍等可选功能。
REQUIRED_TEMPLATES: dict[str, str] = {"play": "出牌", "pass": "不出", "cannot_beat": "要不起"}


class AutomationPanel(QWidget):
    """模式选择与状态显示；只发出意图信号，由主窗口与推理服务交互。

    Attributes:
        mode_requested: 用户选择了新模式。
        stop_requested: 用户点击紧急停止。
        mode_combo: 模式下拉框。
        btn_stop: 紧急停止按钮，始终可见。
        status_label: 阶段、角色与建议。
    """

    mode_requested = Signal(str)
    stop_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """创建控件，初始为关闭模式。

        Args:
            parent: 父控件。
        """
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(3)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.mode_combo = QComboBox()
        for key, text in MODE_OPTIONS:
            self.mode_combo.addItem(text, key)
        self.mode_combo.setToolTip("只给建议：识别并显示 AI 出牌建议，不点击。\n自动操作：在确认画面后自动叫分和出牌。")
        self.mode_combo.currentIndexChanged.connect(self._on_index_changed)
        row.addWidget(self.mode_combo)
        self.btn_stop = QPushButton("紧急停止")
        self.btn_stop.setObjectName("BtnEmergencyStop")
        self.btn_stop.setStyleSheet("QPushButton#BtnEmergencyStop { color: white; background: #c62828; "
                                    "font-weight: bold; padding: 2px 10px; }")
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        root.addLayout(row)
        self.status_label = QLabel("")
        self.status_label.setObjectName("InfoLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        root.addWidget(self.status_label)
        self.status_label.hide()

    def current_mode(self) -> str:
        """返回下拉框当前模式键。"""
        return str(self.mode_combo.currentData())

    def set_mode(self, mode: str) -> None:
        """不发出信号地同步下拉框。

        Args:
            mode: 模式键。

        Raises:
            ValueError: 模式未知。
        """
        index = self.mode_combo.findData(mode)
        if index < 0:
            raise ValueError(f"未知模式: {mode}")
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(index)
        self.mode_combo.blockSignals(False)
        if mode == "disabled":
            self.status_label.hide()

    def set_hotkey(self, key: str) -> None:
        """在按钮上显示热键。

        Args:
            key: 热键名。
        """
        self.btn_stop.setText(f"紧急停止 ({key})" if key else "紧急停止")

    @Slot(int)
    def _on_index_changed(self, index: int) -> None:
        """转发用户选择。

        Args:
            index: 下拉框索引。
        """
        self.mode_requested.emit(str(self.mode_combo.itemData(index)))

    def show_status(self, status: Mapping[str, object]) -> None:
        """显示后端状态；后端已关闭时同步下拉框。

        Args:
            status: AutomationStatus.to_dict() 及运行时附加字段。
        """
        mode = str(status.get("mode", "disabled"))
        if mode != self.current_mode() and self.mode_combo.findData(mode) >= 0:
            self.set_mode(mode)
        hotkey = status.get("hotkey")
        if isinstance(hotkey, str):
            self.set_hotkey(hotkey)
        if mode == "disabled" and not status.get("reason"):
            self.status_label.hide()
            return
        parts = [str(status.get("phase_text", ""))]
        if status.get("role"):
            parts.append(str(status["role"]))
        if status.get("suggestion"):
            parts.append(str(status["suggestion"]))
        lines = [" · ".join(part for part in parts if part)]
        candidates = status.get("candidates")
        if isinstance(candidates, list) and candidates:
            lines.append("候选：" + " | ".join(str(item) for item in candidates))
        if status.get("reason"):
            lines.append("⚠ " + str(status["reason"]))
        missing = status.get("missing_templates")
        if isinstance(missing, list):
            required = [REQUIRED_TEMPLATES[name] for name in missing if name in REQUIRED_TEMPLATES]
            if required:
                lines.append("缺少按钮模板：" + "、".join(required) + "（见使用指南“按钮模板”）")
        self.status_label.setText("\n".join(lines))
        self.status_label.show()
