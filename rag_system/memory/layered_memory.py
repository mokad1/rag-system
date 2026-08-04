"""分层对话记忆管理。

架构：
- 短期记忆：滑动窗口保留最近 N 轮（默认 3 轮=6 条消息）
- 长期记忆：早期对话经 LLM 压缩为摘要文本
- 检索时拼接：summary + 短期对话 → LLM 上下文

触发条件：消息数 > max_memory_turns * 2（默认 12 条 = 6 轮）
压缩策略：保留最近 short_term_window*2 条，其余合并为 LLM 摘要
"""

from __future__ import annotations

from rag_system.config import config
from rag_system.models.conversation import ConversationState
from rag_system.utils.logger import get_logger

logger = get_logger("memory")

SUMMARY_PROMPT = """请将以下对话历史压缩为一段简洁的摘要（100字以内）。
保留关键信息和上下文，丢掉重复和无关内容。

对话历史：
{history}

摘要："""


class LayeredMemory:
    """分层对话记忆。

    每个 session_id 对应一个 ConversationState。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str) -> ConversationState:
        """获取或创建会话状态。

        Args:
            session_id: 会话 ID。

        Returns:
            ConversationState 实例。
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(
                session_id=session_id,
                short_term_window=config.short_term_window,
                max_memory_turns=config.max_memory_turns,
            )
        return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict[str, object]] | None = None,
    ) -> ConversationState:
        """添加消息到会话。

        Args:
            session_id: 会话 ID。
            role: user / assistant。
            content: 消息内容。
            citations: 该轮回答的引用（仅 assistant 消息）。

        Returns:
            更新后的 ConversationState。
        """
        state = self.get_or_create(session_id)
        state.add_message(role, content, citations)  # type: ignore[arg-type]
        return state

    async def maybe_summarize(
        self,
        session_id: str,
        llm_generate,
    ) -> str:
        """如果需要压缩，生成长期摘要。

        触发条件：消息数超过 max_memory_turns * 2。

        Args:
            session_id: 会话 ID。
            llm_generate: async (prompt, system_prompt) -> str 的 LLM 调用函数。

        Returns:
            更新后的长期摘要文本。
        """
        state = self.get_or_create(session_id)

        if not state.needs_summarize:
            return state.long_term_summary

        # 保留最近的 short_term_window*2 条，其余用于压缩
        keep_count = state.short_term_window * 2
        to_summarize = state.messages[:-keep_count]
        state.messages = state.messages[-keep_count:]

        # 构建摘要 prompt
        history = "\n".join(
            f"{'用户' if m.role == 'user' else '助手'}: {m.content}"
            for m in to_summarize
        )

        try:
            summary = await llm_generate(
                prompt=SUMMARY_PROMPT.format(history=history),
                system_prompt="你是一个对话摘要助手。",
            )
            if state.long_term_summary:
                state.long_term_summary += "\n" + summary
            else:
                state.long_term_summary = summary
            logger.info(
                "Session %s summarized: %d messages → summary (%d chars)",
                session_id[:8], len(to_summarize), len(summary),
            )
        except Exception as e:
            logger.error("Summarization failed for %s: %s", session_id[:8], e)

        return state.long_term_summary

    def get_context(self, session_id: str) -> str:
        """获取当前对话上下文（供检索/生成使用）。

        Args:
            session_id: 会话 ID。

        Returns:
            拼接后的上下文字符串（summary + 短期消息）。
        """
        state = self.get_or_create(session_id)
        return state.get_context_string()

    def clear(self, session_id: str) -> None:
        """清除指定会话。"""
        self._sessions.pop(session_id, None)
