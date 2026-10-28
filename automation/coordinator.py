"""闭环自动化协调器：稳定观测 → 决策 → 选牌/按钮规划 → 受保护点击 → 画面验证。

任何未知、矛盾、超时或执行失败都转为可诊断的暂停；观察模式完整运行但从不点击。
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from automation.bidding import (BID_BUTTONS, DOUBLE_BUTTONS, BidPolicy, bid_level, choose_bid_button,
                                choose_double_button, hand_strength)
from automation.card_selection import (UnplannableAction, plan_card_clicks, raised_flags, resting_baseline,
                                       selected_cards)
from automation.emergency_stop import AutomationStopped
from automation.game_record import ROLE_TEXT, RoundRecord
from automation.input_backend import GuardedInputBackend, UnsafeClick
from automation.safety import StableObservationGate, pause_reason
from automation.types import (PHASE_TEXT, AutomationMode, AutomationPhase, ButtonMatch, CardBox, ClickPlan,
                              ClickPoint, TableObservation)
from decision.card_codec import decode_cards, format_cards
from decision.douzero_adapter import DouZeroAdapter
from decision.protocol import DecisionError

PLAY_BUTTONS: tuple[str, ...] = ("play", "pass", "cannot_beat")
BUTTON_TEXT: dict[str, str] = {
    "bid_1": "叫1分", "bid_2": "叫2分", "bid_3": "叫3分", "bid_pass": "不叫", "call_landlord": "叫地主",
    "no_call": "不叫", "rob_landlord": "抢地主", "no_rob": "不抢", "double": "加倍", "no_double": "不加倍",
    "play": "出牌", "pass": "不出", "cannot_beat": "要不起", "continue": "继续游戏",
}


@dataclass(frozen=True)
class AutomationConfig:
    """自动化参数。

    Attributes:
        required_frames: 执行动作前需要的连续一致帧数。
        raise_ratio: 选中牌上移判定阈值（相对牌框高度）。
        verify_timeout: 点击后等待画面变化的秒数。
        max_corrections: 选牌不符时允许的纠正次数。
        resume_next_round: 暂停后是否在下一局自动恢复。
        auto_continue: 自动模式下是否点击“继续游戏”。
        round_end_seconds: 出牌阶段底牌区持续为空多少秒视为本局结束。
        bid_policy: 叫分/加倍阈值。
    """

    required_frames: int = 3
    raise_ratio: float = 0.3
    verify_timeout: float = 3.0
    max_corrections: int = 1
    resume_next_round: bool = True
    auto_continue: bool = False
    round_end_seconds: float = 3.0
    bid_policy: BidPolicy = field(default_factory=BidPolicy)


@dataclass
class _Pending:
    """一次尚未验证的操作。

    Attributes:
        kind: select、play、pass 或 button。
        deadline: 超时时刻。
        target: 目标出牌。
        button: 按钮操作的按钮名。
        hand_before: 操作前手牌张数。
        attempts: 已纠正次数。
    """

    kind: str
    deadline: float
    target: tuple[str, ...] = ()
    button: str = ""
    hand_before: int = 0
    attempts: int = 0


@dataclass
class AutomationStatus:
    """发给界面的纯数据状态。

    Attributes:
        mode: 当前模式。
        phase: 当前阶段。
        role: 本家角色文本。
        suggestion: 当前建议或已执行动作。
        reason: 暂停或等待原因。
        candidates: 模型前三候选文本。
        plan: 计划点击点 (x, y, 说明)。
        clicks: 累计真实点击数。
    """

    mode: AutomationMode = AutomationMode.DISABLED
    phase: AutomationPhase = AutomationPhase.WAITING
    role: str = ""
    suggestion: str = ""
    reason: str = ""
    candidates: tuple[str, ...] = ()
    plan: tuple[tuple[int, int, str], ...] = ()
    clicks: int = 0

    def to_dict(self) -> dict[str, object]:
        """转换为可跨进程传递的字典。

        Returns:
            只含 str/int/list 的字典。
        """
        return {"mode": self.mode.value, "phase": self.phase.value, "phase_text": PHASE_TEXT[self.phase],
                "role": self.role, "suggestion": self.suggestion, "reason": self.reason,
                "candidates": list(self.candidates), "plan": [list(item) for item in self.plan],
                "clicks": self.clicks}


class AutomationCoordinator:
    """逐帧驱动自动化状态机；所有依赖由构造函数注入。

    Attributes:
        decider: DouZero 适配器；None 时只能叫分不能出牌。
        backend: 受保护的输入执行器。
        config: 参数。
        status: 最近状态。
        record: 当前局记录。
    """

    def __init__(self, decider: DouZeroAdapter | None, backend: GuardedInputBackend,
                 config: AutomationConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """创建协调器，初始为关闭模式。

        Args:
            decider: 出牌决策适配器。
            backend: 输入执行器。
            config: 参数；None 使用默认值。
            clock: 单调时钟，测试可注入。
        """
        self.decider = decider
        self.backend = backend
        self.config = config or AutomationConfig()
        self.status = AutomationStatus()
        self.record: RoundRecord | None = None
        self._clock = clock
        self._gate = StableObservationGate(self.config.required_frames, self.config.raise_ratio)
        self._previous: TableObservation | None = None
        self._pending: _Pending | None = None
        self._baseline: float | None = None
        self._handled: set[str] = set()
        self._need_fresh = False
        self._bottom_empty_since: float | None = None

    # ------------------------------------------------------------------ 模式
    @property
    def mode(self) -> AutomationMode:
        """返回当前模式。"""
        return self.status.mode

    @property
    def active(self) -> bool:
        """观察或自动模式时为 True。"""
        return self.status.mode != AutomationMode.DISABLED

    def set_mode(self, mode: AutomationMode | str) -> None:
        """切换模式；每次切换都从等待开局重新开始。

        Args:
            mode: 新模式。

        Raises:
            ValueError: 模式未知，或缺少决策模型时开启出牌。
        """
        selected = AutomationMode(mode)
        if selected != AutomationMode.DISABLED and self.decider is None:
            raise ValueError("缺少 DouZero 决策模型，无法开启观察/自动模式")
        if selected == AutomationMode.AUTO:
            self.backend.stop.clear()
        self.status = AutomationStatus(mode=selected, clicks=self.backend.executed)
        self.record = None
        self._reset_round_state()
        self._need_fresh = False

    def fail(self, reason: str) -> None:
        """外部异常时暂停自动化，放弃未验证的操作。

        Args:
            reason: 暂停原因。
        """
        if self.active:
            self._pause(reason)

    def reset_round(self) -> None:
        """用户显式重置牌局时丢弃本局记录，等待下一局。"""
        self.record = None
        self._reset_round_state()
        self._need_fresh = True
        if self.active:
            self._set(phase=AutomationPhase.WAITING, reason="已重置，等待下一局", role="", plan=(),
                      candidates=())

    def _reset_round_state(self) -> None:
        """清空一局内的临时状态。"""
        self._pending = None
        self._baseline = None
        self._handled.clear()
        self._gate.reset()
        self._bottom_empty_since = None

    # ------------------------------------------------------------------ 主循环
    def handle(self, observation: TableObservation, hwnd: int | None = None) -> AutomationStatus:
        """处理一帧观测。

        Args:
            observation: 当前观测。
            hwnd: 截图时的目标窗口句柄。

        Returns:
            最新状态。
        """
        if not self.active:
            return self.status
        if self.backend.stop.is_set() and self.mode == AutomationMode.AUTO:
            self.status = replace(self.status, mode=AutomationMode.DISABLED, phase=AutomationPhase.PAUSED,
                                  reason=self.backend.stop.reason or "紧急停止", plan=())
            self._pending = None
            return self.status
        previous, self._previous = self._previous, observation
        reason = pause_reason(previous, observation, self._pending is not None)
        if reason is not None and self.status.phase != AutomationPhase.PAUSED:
            if self.record is None and self._pending is None:
                # 局间弹窗不点击也不锁定暂停，弹窗关闭后自然继续等待开局。
                self._gate.reset()
                self._set(reason=reason)
                return self.status
            self._pause(reason)
            return self.status
        stable = self._gate.observe(observation, self._baseline)
        if stable and self._pending is None and observation.hand:
            self._baseline = resting_baseline(observation.hand, self.config.raise_ratio) or self._baseline
        if stable:
            self._clear_handled(observation)
            if self._round_over(observation):
                self._finish_round()
                return self.status
            if self._need_fresh and (observation.bottom == () or observation.has_any(BID_BUTTONS)):
                self._need_fresh = False
        if self.status.phase == AutomationPhase.PAUSED:
            return self.status
        if self._pending is not None:
            self._advance(observation, stable, hwnd)
            return self.status
        if not stable:
            return self.status
        try:
            self._dispatch(observation, hwnd)
        except AutomationStopped as exc:
            self.status = replace(self.status, mode=AutomationMode.DISABLED, phase=AutomationPhase.PAUSED,
                                  reason=str(exc))
            self._pending = None
        except (UnsafeClick, UnplannableAction) as exc:
            self._pause(str(exc))
        return self.status

    def _dispatch(self, obs: TableObservation, hwnd: int | None) -> None:
        """在稳定帧上按可见按钮选择处理流程。

        Args:
            obs: 稳定观测。
            hwnd: 窗口句柄。
        """
        if obs.has_any(BID_BUTTONS):
            self._bid(obs, hwnd)
            return
        if obs.has_any(DOUBLE_BUTTONS):
            self._double(obs, hwnd)
            return
        if self.record is None and not self._need_fresh and obs.bottom is not None and len(obs.bottom) == 3:
            self.record = RoundRecord(obs.bottom, obs.history)
            self._set(phase=AutomationPhase.PLAYING, reason="", suggestion="底牌已确认，等待出牌")
        if obs.has_any(PLAY_BUTTONS):
            self._turn(obs, hwnd)
            return
        if obs.has_any(("continue",)):
            self._continue(obs, hwnd)
            return
        if self.record is None and self.status.phase not in (AutomationPhase.BIDDING,):
            self._set(phase=AutomationPhase.WAITING, reason="" if not self._need_fresh else "等待下一局")

    # ------------------------------------------------------------------ 叫分/加倍
    def _bid(self, obs: TableObservation, hwnd: int | None) -> None:
        """处理叫分、叫/抢地主按钮。

        Args:
            obs: 稳定观测。
            hwnd: 窗口句柄。
        """
        if self.record is not None or self._need_fresh:
            self.record, self._need_fresh = None, False
            self._reset_round_state()
        self._set(phase=AutomationPhase.BIDDING)
        if "bid" in self._handled:
            return
        hand = obs.hand_cards()
        if len(hand) != 17:
            self._set(reason=f"叫分时识别到 {len(hand)} 张手牌，等待 17 张稳定")
            return
        strength = hand_strength(hand)
        available = {name for name in BID_BUTTONS if obs.has_any((name,))}
        name = choose_bid_button(bid_level(strength, self.config.bid_policy), available)
        self._handled.add("bid")
        if name is None:
            self._set(suggestion=f"手牌强度 {strength:g}，没有匹配的叫分按钮，交给你操作", reason="")
            return
        self._press(obs, hwnd, name, f"手牌强度 {strength:g}，{BUTTON_TEXT[name]}")

    def _double(self, obs: TableObservation, hwnd: int | None) -> None:
        """处理加倍按钮。

        Args:
            obs: 稳定观测。
            hwnd: 窗口句柄。
        """
        self._set(phase=AutomationPhase.BIDDING)
        if "double" in self._handled:
            return
        strength = hand_strength(obs.hand_cards())
        available = {name for name in DOUBLE_BUTTONS if obs.has_any((name,))}
        name = choose_double_button(strength, self.config.bid_policy, available)
        self._handled.add("double")
        if name is None:
            self._set(suggestion=f"手牌强度 {strength:g}，不加倍（等待游戏默认）", reason="")
            return
        self._press(obs, hwnd, name, f"手牌强度 {strength:g}，{BUTTON_TEXT[name]}")

    def _continue(self, obs: TableObservation, hwnd: int | None) -> None:
        """按配置点击“继续游戏”。

        Args:
            obs: 稳定观测。
            hwnd: 窗口句柄。
        """
        if not self.config.auto_continue or "continue" in self._handled:
            return
        self._handled.add("continue")
        self._press(obs, hwnd, "continue", "继续下一局")

    def _press(self, obs: TableObservation, hwnd: int | None, name: str, text: str) -> None:
        """点击一个按钮并等待它消失。

        Args:
            obs: 稳定观测。
            hwnd: 窗口句柄。
            name: 按钮名。
            text: 界面说明。
        """
        plan = ClickPlan((self._button_point(obs.buttons[name]),), text)
        self._set(suggestion=text, reason="", plan=self._plan_tuple(plan))
        if self.mode != AutomationMode.AUTO:
            return
        self._execute(plan, obs, hwnd)
        self._pending = _Pending("button", self._clock()+self.config.verify_timeout, button=name)
        self._set(phase=AutomationPhase.VERIFYING)

    # ------------------------------------------------------------------ 出牌
    def _turn(self, obs: TableObservation, hwnd: int | None) -> None:
        """轮到本家：推算牌局、决策并执行。

        Args:
            obs: 稳定观测。
            hwnd: 窗口句柄。
        """
        if "turn" in self._handled:
            return
        if self.record is None:
            self._set(reason="本局开始时未确认三张底牌，请手动完成本局")
            return
        if self.decider is None:
            self._set(reason="缺少出牌模型")
            return
        hand = obs.hand_cards()
        try:
            preview = self.record.preview(hand, obs.history)
            decision = self.decider.decide(preview.state)
        except DecisionError as exc:
            self._set(reason=f"牌局记录与画面不一致，暂不出牌：{exc}")
            return
        self.record.commit(preview)
        self._handled.add("turn")
        action = tuple(decode_cards(decision.action))
        candidates = tuple(f"{format_cards(decode_cards(cards))} ({value:.3f})"
                           for cards, value in decision.candidates)
        self._set(phase=AutomationPhase.PLAYING, role=ROLE_TEXT[preview.position], reason="",
                  suggestion="建议：" + format_cards(action), candidates=candidates)
        ordered = sorted(obs.hand or (), key=lambda box: box.bbox[0])
        flags = raised_flags(ordered, self.config.raise_ratio, self._baseline)
        if not action:
            name = "pass" if obs.has_any(("pass",)) else ("cannot_beat" if obs.has_any(("cannot_beat",)) else "")
            if not name:
                self._pause("模型选择不出，但界面没有“不出/要不起”按钮")
                return
            plan = ClickPlan((self._button_point(obs.buttons[name]),), BUTTON_TEXT[name])
            self._set(plan=self._plan_tuple(plan))
            if self.mode == AutomationMode.AUTO:
                self.record.expected_own = ()
                self._execute(plan, obs, hwnd)
                self._pending = _Pending("pass", self._clock()+self.config.verify_timeout,
                                         hand_before=len(hand))
                self._set(phase=AutomationPhase.VERIFYING)
            return
        if not obs.has_any(("play",)):
            self._pause("模型选择出牌，但界面没有“出牌”按钮")
            return
        plan = plan_card_clicks(action, ordered, flags, obs.image_size)
        full = ClickPlan(plan.clicks+(self._button_point(obs.buttons["play"]),), plan.description)
        self._set(plan=self._plan_tuple(full))
        if self.mode != AutomationMode.AUTO:
            return
        self.record.expected_own = action
        self._execute(plan, obs, hwnd)
        self._pending = _Pending("select", self._clock()+self.config.verify_timeout, target=action,
                                 hand_before=len(hand))
        self._set(phase=AutomationPhase.VERIFYING)

    # ------------------------------------------------------------------ 验证
    def _advance(self, obs: TableObservation, stable: bool, hwnd: int | None) -> None:
        """推进尚未验证的操作。

        Args:
            obs: 当前观测。
            stable: 是否稳定。
            hwnd: 窗口句柄。
        """
        pending = self._pending
        assert pending is not None
        if self._clock() > pending.deadline:
            self._pause(f"操作超时未生效：{pending.kind} {format_cards(pending.target) if pending.target else pending.button}")
            return
        if not stable:
            return
        try:
            if pending.kind == "select":
                self._verify_selection(obs, hwnd, pending)
            elif pending.kind in ("play", "pass"):
                self._verify_turn(obs, pending)
            elif not obs.has_any((pending.button,)):
                self._pending = None
                self._set(phase=AutomationPhase.PLAYING if self.record else AutomationPhase.BIDDING)
        except AutomationStopped as exc:
            self.status = replace(self.status, mode=AutomationMode.DISABLED, phase=AutomationPhase.PAUSED,
                                  reason=str(exc))
            self._pending = None
        except (UnsafeClick, UnplannableAction) as exc:
            self._pause(str(exc))

    def _verify_selection(self, obs: TableObservation, hwnd: int | None, pending: _Pending) -> None:
        """确认选中的牌与目标一致后点击出牌。

        Args:
            obs: 稳定观测。
            hwnd: 窗口句柄。
            pending: 选牌操作。
        """
        ordered = sorted(obs.hand or (), key=lambda box: box.bbox[0])
        flags = raised_flags(ordered, self.config.raise_ratio, self._baseline)
        if selected_cards(ordered, flags) == Counter(pending.target):
            if not obs.has_any(("play",)):
                self._pause("选牌完成但找不到“出牌”按钮")
                return
            self._execute(ClickPlan((self._button_point(obs.buttons["play"]),), "出牌"), obs, hwnd)
            self._pending = _Pending("play", self._clock()+self.config.verify_timeout, target=pending.target,
                                     hand_before=pending.hand_before)
            return
        if pending.attempts >= self.config.max_corrections:
            self._pause("选中的牌与计划不一致，已停止以免出错牌")
            return
        plan = plan_card_clicks(pending.target, ordered, flags, obs.image_size)
        self._execute(plan, obs, hwnd)
        pending.attempts += 1
        pending.deadline = self._clock()+self.config.verify_timeout

    def _verify_turn(self, obs: TableObservation, pending: _Pending) -> None:
        """确认出牌/不出已被游戏接受。

        Args:
            obs: 稳定观测。
            pending: 出牌或不出操作。
        """
        count = len(obs.hand or ())
        expected = pending.hand_before-len(pending.target)
        # 出牌后张数减少即已接受；王炸后可能立刻再次轮到本家，按钮仍可见。
        if not (pending.target and count == expected):
            if obs.has_any(PLAY_BUTTONS):
                return
            if count != expected:
                self._pause(f"操作后手牌 {count} 张，预期 {expected} 张")
                return
        self._pending = None
        self._handled.discard("turn")
        self._set(phase=AutomationPhase.PLAYING, suggestion="已" + (
            "出 " + format_cards(pending.target) if pending.target else "不出"), plan=())
        if expected == 0:
            self._finish_round()

    # ------------------------------------------------------------------ 结束与工具
    def _round_over(self, obs: TableObservation) -> bool:
        """判断本局是否结束。

        Args:
            obs: 稳定观测。

        Returns:
            出现继续按钮、底牌区清空或有人出完牌时为 True。
        """
        if self.record is None:
            return False
        if obs.has_any(("continue",)):
            return True
        if obs.bottom == ():
            if self._bottom_empty_since is None:
                self._bottom_empty_since = obs.timestamp
            if obs.timestamp-self._bottom_empty_since >= self.config.round_end_seconds:
                return True
        else:
            self._bottom_empty_since = None
        left = self.record.remaining(obs.history)
        return left is not None and any(count <= 0 for count in left.values())

    def _finish_round(self) -> None:
        """结束本局，必要时从暂停恢复到等待下一局。"""
        self.record = None
        self._reset_round_state()
        self._need_fresh = True
        if self.status.phase == AutomationPhase.PAUSED and not self.config.resume_next_round:
            return
        self._set(phase=AutomationPhase.WAITING, reason="本局结束，等待下一局", role="", plan=(), candidates=())

    def _clear_handled(self, obs: TableObservation) -> None:
        """按钮消失后允许下一次同类操作。

        Args:
            obs: 稳定观测。
        """
        if not obs.has_any(BID_BUTTONS):
            self._handled.discard("bid")
        if not obs.has_any(DOUBLE_BUTTONS):
            self._handled.discard("double")
        if not obs.has_any(PLAY_BUTTONS) and self._pending is None:
            self._handled.discard("turn")
        if not obs.has_any(("continue",)):
            self._handled.discard("continue")

    def _execute(self, plan: ClickPlan, obs: TableObservation, hwnd: int | None) -> None:
        """执行点击并清空稳定门，下一次判断必须基于点击后的新帧。

        Args:
            plan: 点击计划。
            obs: 计划所依据的观测。
            hwnd: 窗口句柄。
        """
        self.backend.execute(plan, self.mode, hwnd, obs.window_rect)
        self._gate.reset()
        self._set(clicks=self.backend.executed)

    def _pause(self, reason: str) -> None:
        """暂停到本局结束。

        Args:
            reason: 暂停原因。
        """
        self._pending = None
        self._set(phase=AutomationPhase.PAUSED, reason=reason, plan=())

    def _set(self, **changes: object) -> None:
        """更新状态字段。

        Args:
            **changes: AutomationStatus 的字段。
        """
        self.status = replace(self.status, **changes)  # type: ignore[arg-type]

    @staticmethod
    def _button_point(match: ButtonMatch) -> ClickPoint:
        """返回按钮中心点击点。

        Args:
            match: 按钮匹配。

        Returns:
            点击点。
        """
        x, y = match.center
        return ClickPoint(int(round(x)), int(round(y)), match.name)

    @staticmethod
    def _plan_tuple(plan: ClickPlan) -> tuple[tuple[int, int, str], ...]:
        """把计划转换为界面显示的元组。

        Args:
            plan: 点击计划。

        Returns:
            (x, y, 说明) 元组。
        """
        return tuple((click.x, click.y, click.label) for click in plan.clicks)


def observation_from_frame(timestamp: float, valid: bool, image_size: tuple[int, int],
                           window_rect: tuple[int, int, int, int] | None,
                           region_boxes: Mapping[str, list[dict[str, object]]], known: set[str],
                           mapping: Mapping[str, str], played_regions: Mapping[str, str],
                           buttons: Mapping[str, ButtonMatch],
                           history: Mapping[str, list[list[str]]]) -> TableObservation:
    """把 GameController 的单帧结果转换为自动化观测。

    Args:
        timestamp: 单调时钟秒。
        valid: 截图与模型是否成功。
        image_size: 截图宽高。
        window_rect: 截图时窗口屏幕矩形。
        region_boxes: 分区后的检测框 {区域: [{"bbox", "name"}]}。
        known: 本帧已知区域。
        mapping: YOLO 类别到 Han 牌点。
        played_regions: 座位键到出牌区域名，如 {"left": "opponent_left"}。
        buttons: 按钮匹配。
        history: 记牌器已确认历史。

    Returns:
        自动化观测；未知区域为 None。
    """
    def cards(region: str) -> tuple[str, ...] | None:
        """返回区域牌点，未知为 None。"""
        if not valid or region not in known:
            return None
        return tuple(mapping[str(item["name"])] for item in region_boxes.get(region, []))

    def box(item: Mapping[str, object]) -> CardBox:
        """把单个检测字典转换为手牌框。"""
        raw = item["bbox"]
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError("检测框必须包含四个坐标")
        x1, y1, x2, y2 = (float(value) for value in raw)
        return CardBox(mapping[str(item["name"])], (x1, y1, x2, y2))

    hand: tuple[CardBox, ...] | None = None
    if valid and "player_hand" in known:
        hand = tuple(box(item) for item in region_boxes.get("player_hand", []))
    return TableObservation(
        timestamp=timestamp, valid=valid, image_size=image_size, window_rect=window_rect, hand=hand,
        played={seat: cards(region) for seat, region in played_regions.items()},
        bottom=cards("landlord_cards"), buttons=dict(buttons),
        history={seat: tuple(tuple(item) for item in items) for seat, items in history.items()})
