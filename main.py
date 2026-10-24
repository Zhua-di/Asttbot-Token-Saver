"""群聊会话窗口插件。

群里默认只有被 @ 或被叫到名字时才回复；被叫到后开启固定时长的会话窗口。
窗口内不会无脑插嘴：会先判断"这句是不是在跟她说话"，是在跟她说才接。
窗口结束后再被叫醒，会先表现一下"愣神"。
另带三条限流：回复间隔、每分钟回复上限、每分钟判断次数上限。

实现要点：AstrBot 的 LLM 调用闸门在 ProcessStage 读 ``event.is_at_or_wake_command``，
插件 handler 在它之前执行，所以改这个标志就能决定"回不回"。
"""

import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

RATE_WINDOW_SECONDS = 60.0
SLEEPY_KEY = "_wake_window_sleepy"
DAZE_KEY = "_wake_window_daze"
BUFFER_SIZE = 6


@register(
    "astrbot_plugin_wake_window",
    "dsh",
    "群聊会话窗口：关键词唤醒 + 固定窗口 + 插嘴判断 + 回复间隔/每分钟上限",
    "1.5.0",
)
class WakeWindowPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._active_until: dict[str, float] = {}
        self._recent: dict[str, list[float]] = {}
        self._sleep_until: dict[str, float] = {}
        self._seen: dict[str, bool] = {}
        self._last_reply: dict[str, float] = {}
        self._buffer: dict[str, list[str]] = {}
        self._judge_recent: dict[str, list[float]] = {}

    # ---------- 配置 ----------
    def _keywords(self) -> list[str]:
        raw = self.config.get("keywords", []) or []
        if isinstance(raw, str):
            raw = [raw]
        return [i.strip() for i in raw if isinstance(i, str) and i.strip()]

    def _num(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _window_seconds(self) -> float:
        v = self._num("window_minutes", 5)
        return (v if v > 0 else 5.0) * 60.0

    def _min_interval(self) -> float:
        v = self._num("min_reply_interval_seconds", 6)
        return v if v >= 0 else 6.0

    def _reply_on_at(self) -> bool:
        return bool(self.config.get("reply_on_at", True))

    def _max_per_minute(self) -> int:
        try:
            v = int(self.config.get("max_per_minute", 5))
        except (TypeError, ValueError):
            v = 5
        return max(0, v)

    def _judge_max_per_minute(self) -> int:
        try:
            v = int(self.config.get("judge_max_per_minute", 10))
        except (TypeError, ValueError):
            v = 10
        return max(0, v)

    def _sleepy_seconds(self) -> float:
        v = self._num("sleepy_seconds", 120)
        return v if v > 0 else 120.0

    def _text_of(self, key: str, fallback: str) -> str:
        raw = self.config.get(key, "") or ""
        raw = raw.strip() if isinstance(raw, str) else ""
        return raw or fallback

    # ---------- 缓冲 ----------
    def _push(self, umo: str, line: str) -> None:
        buf = self._buffer.setdefault(umo, [])
        buf.append(line)
        if len(buf) > BUFFER_SIZE:
            del buf[: len(buf) - BUFFER_SIZE]

    @filter.on_decorating_result()
    async def remember_reply(self, event: AstrMessageEvent):
        """记录机器人自己发出的内容，供"是不是在跟她说话"的判断使用。"""
        try:
            if event.is_private_chat():
                return
            result = event.get_result()
            if not result or not getattr(result, "chain", None):
                return
            parts = []
            for comp in result.chain:
                text = getattr(comp, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            joined = "".join(parts).strip()
            if joined:
                self._push(event.unified_msg_origin, "机器人：" + joined[:120])
        except Exception as exc:
            logger.debug(f"[WakeWindow] 记录回复失败: {exc}")

    # ---------- 判断是否在跟她说话 ----------
    async def _addressed_to_bot(self, event: AstrMessageEvent, umo: str) -> bool:
        if not bool(self.config.get("judge_enabled", True)):
            return True

        now = time.time()
        limit = self._judge_max_per_minute()
        if limit > 0:
            stamps = [
                t for t in self._judge_recent.get(umo, [])
                if now - t < RATE_WINDOW_SECONDS
            ]
            if len(stamps) >= limit:
                self._judge_recent[umo] = stamps
                logger.info(
                    f"[WakeWindow] 判断次数已达每分钟 {limit} 次上限，本条不插嘴"
                )
                return False
            stamps.append(now)
            self._judge_recent[umo] = stamps

        provider = None
        try:
            provider = self.context.get_using_provider(umo=umo)
        except Exception as exc:
            logger.warning(f"[WakeWindow] 获取模型失败，放行: {exc}")
            return True
        if provider is None:
            return True

        history = "\n".join(self._buffer.get(umo, [])[-4:])
        prompt = (
            f"以下是群聊最近的对话：\n{history}\n\n"
            f"现在有个群友说：「{event.message_str}」\n"
            "这句是在跟机器人说话（回应它、问它、接着它的话），还是群友之间在聊自己的事？"
            "只输出 1（在跟机器人说）或 0（群友自己在聊）。"
        )
        try:
            resp = await provider.text_chat(
                prompt=prompt,
                session_id=None,
                contexts=[],
                system_prompt=self._text_of(
                    "judge_instruction",
                    "你是一个判断器，只输出一个字符：1 表示「在跟机器人说话」，"
                    "0 表示「群友之间在聊天」。不要输出任何其它内容。",
                ),
            )
            answer = (resp.completion_text or "").strip()
        except Exception as exc:
            logger.warning(f"[WakeWindow] 判断请求失败，按不插嘴处理: {exc}")
            return False

        is_addressed = answer.startswith("1")
        logger.info(
            f"[WakeWindow] 插嘴判断: {'接话' if is_addressed else '不插嘴'} "
            f"(model={answer[:8]!r})"
        )
        return is_addressed

    # ---------- 频率 ----------
    def _interval_blocked(self, umo: str, now: float) -> bool:
        gap = self._min_interval()
        if gap <= 0:
            return False
        return now - self._last_reply.get(umo, 0.0) < gap

    def _rate_blocked(self, event: AstrMessageEvent, umo: str, now: float) -> bool:
        limit = self._max_per_minute()
        if limit <= 0:
            return False
        stamps = [t for t in self._recent.get(umo, []) if now - t < RATE_WINDOW_SECONDS]
        self._recent[umo] = stamps
        if len(stamps) < limit:
            return False
        self._sleep_until[umo] = now + self._sleepy_seconds()
        self._recent[umo] = []
        self._active_until.pop(umo, None)
        event.set_extra(SLEEPY_KEY, True)
        event.is_at_or_wake_command = True
        event.is_wake = True
        logger.info(
            f"[WakeWindow] 一分钟内已达 {limit} 条，让她说困了并静默 "
            f"{self._sleepy_seconds():.0f} 秒（group={event.get_group_id()}）"
        )
        return True

    def _pass(self, event: AstrMessageEvent, umo: str, now: float) -> None:
        self._recent.setdefault(umo, []).append(now)
        self._last_reply[umo] = now
        event.is_at_or_wake_command = True
        event.is_wake = True

    # ---------- 主逻辑 ----------
    @filter.event_message_type(EventMessageType.ALL)
    async def on_group_message(self, event: AstrMessageEvent):
        if event.is_private_chat():
            return

        try:
            group_id = event.get_group_id() or ""
        except Exception:
            group_id = ""
        if not group_id:
            return

        umo = event.unified_msg_origin
        now = time.time()
        window = self._window_seconds()
        text = (event.message_str or "").strip()

        if text:
            self._push(umo, "群友：" + text[:120])

        # 0) 静默期：不管怎么叫（含 @）都不回
        sleep_until = self._sleep_until.get(umo, 0.0)
        if sleep_until:
            if now < sleep_until:
                event.is_at_or_wake_command = False
                event.is_wake = False
                return
            self._sleep_until.pop(umo, None)

        native_wake = bool(event.is_at_or_wake_command)
        hit = next((kw for kw in self._keywords() if kw in text), None)
        in_window = now < self._active_until.get(umo, 0.0)

        if not (native_wake or hit or in_window):
            if self._active_until.get(umo):
                self._active_until.pop(umo, None)
                logger.info(f"[WakeWindow] 群 {group_id} 会话窗口结束")
            return

        # 1) 回复间隔：太急就不接（不调用模型，零费用）
        if self._interval_blocked(umo, now):
            logger.info(f"[WakeWindow] 距上次回复不足 {self._min_interval():.0f} 秒，本条不接")
            return

        # 2) 每分钟上限
        if self._rate_blocked(event, umo, now):
            return

        # 3) @ 她 / 命中关键词：确定是在叫她
        if native_wake:
            if self._reply_on_at():
                self._open_window(event, umo, now, window, group_id)
                self._pass(event, umo, now)
            return
        if hit:
            self._open_window(event, umo, now, window, group_id, keyword=hit)
            self._pass(event, umo, now)
            return

        # 4) 窗口内的普通消息：先判断是不是在跟她说
        #    注意：判断是一次 await，期间可能有多条消息并发进来。
        #    所以判断完成后必须用"最新时间"重新做一次间隔/频率检查，
        #    并且从检查到 _pass() 之间不能有 await，才能保证原子性。
        if not await self._addressed_to_bot(event, umo):
            return

        now = time.time()
        if self._interval_blocked(umo, now):
            logger.info(
                f"[WakeWindow] 判断期间已有回复占位，本条不接"
                f"（间隔 {self._min_interval():.0f} 秒）"
            )
            return
        if self._rate_blocked(event, umo, now):
            return
        if bool(self.config.get("sliding", False)):
            self._active_until[umo] = now + window
        self._pass(event, umo, now)

    def _open_window(
        self,
        event: AstrMessageEvent,
        umo: str,
        now: float,
        window: float,
        group_id: str,
        keyword: str | None = None,
    ) -> None:
        expired = now >= self._active_until.get(umo, 0.0)
        if self._seen.get(umo) and expired and bool(self.config.get("daze_enabled", True)):
            event.set_extra(DAZE_KEY, True)
            logger.info(f"[WakeWindow] 群 {group_id} 重新唤醒，注入「愣神」指令")

        self._seen[umo] = True
        self._active_until[umo] = now + window
        if keyword:
            logger.info(
                f"[WakeWindow] 关键词命中「{keyword}」，群 {group_id} 开启 "
                f"{window / 60:.0f} 分钟会话窗口（到点即止）"
            )

    # ---------- 临时指令注入 ----------
    @filter.on_llm_request()
    async def inject_temp_instruction(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        if event.get_extra(SLEEPY_KEY, False):
            text = self._text_of(
                "sleepy_instruction",
                "你现在困得不行，只能用一两句话表达你想睡觉、要下线了。"
                "不要回答用户这条消息，不要提问，不要邀请对方继续聊，说完就结束。",
            )
            req.system_prompt = (req.system_prompt or "") + (
                "\n\n【临时状态｜优先级最高】" + text
            )
            logger.info("[WakeWindow] 已注入「困了」临时指令")
            return

        if event.get_extra(DAZE_KEY, False):
            text = self._text_of(
                "daze_instruction",
                "你刚在走神发呆，被这一声叫才回过神来。先用一两句话表现出"
                "「啊？愣神了/走神了」的状态，再自然接上对方的话，不要长篇大论。",
            )
            req.system_prompt = (req.system_prompt or "") + "\n\n【临时状态】" + text
            logger.info("[WakeWindow] 已注入「愣神」临时指令")
