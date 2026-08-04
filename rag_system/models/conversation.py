"""对话记忆数据模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """单条对话消息。"""
    role: str = Field(..., description="user / assistant")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now)
    citations: list[dict[str, Any]] = Field(
        default_factory=list, description="该轮回答的引用列表"
    )


class ConversationState(BaseModel):
    """对话状态。

    - short_term: 滑动窗口内最近 N 轮对话
    - long_term_summary: LLM 生成的早期对话摘要
    - 检索时拼接：summary + 最近 short_term 作为上下文
    """
    session_id: str = Field(..., description="会话 ID")
    messages: list[Message] = Field(default_factory=list, description="全部消息")
    long_term_summary: str = Field(default="", description="长期摘要")
    short_term_window: int = Field(default=3, description="短期窗口大小")
    max_memory_turns: int = Field(default=6, description="触发摘要压缩的阈值")

    @property
    def short_term(self) -> list[Message]:
        """取最近 N 轮作为短期记忆。"""
        return self.messages[-self.short_term_window * 2:]

    @property
    def needs_summarize(self) -> bool:
        """是否需要触发摘要压缩。"""
        return len(self.messages) > self.max_memory_turns * 2

    def add_message(self, role: str, content: str, citations: list[dict[str, Any]] | None = None) -> None:
        """添加消息并触发可能的摘要压缩（外部调用）。"""
        self.messages.append(Message(role=role, content=content, citations=citations or []))

    def get_context_string(self) -> str:
        """构建送给 LLM 的对话上下文字符串。"""
        parts: list[str] = []
        if self.long_term_summary:
            parts.append(f"[对话摘要]\n{self.long_term_summary}\n")
        for msg in self.short_term:
            role_label = "用户" if msg.role == "user" else "助手"
            parts.append(f"{role_label}: {msg.content}")
        return "\n".join(parts)
