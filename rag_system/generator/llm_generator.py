"""LLM 答案生成器。

基于检索到的上下文和对话记忆，调用 LLM 生成带溯源的答案。

设计要点：
- Prompt 严格要求 LLM 使用 [citation:N] 格式引用来源
- 后处理解析引用标记，映射为结构化的 Citation 对象
- 支持对话记忆上下文拼接
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from rag_system.config import config
from rag_system.models.document import Chunk
from rag_system.utils.logger import get_logger

logger = get_logger("generator.llm")

GENERATION_SYSTEM_PROMPT = """你是一个专业的技术文档问答助手。你的回答必须基于提供的参考文档片段。

## 回答规则
1. **仅基于参考文档**：只使用下面提供的文档片段回答问题，不要使用你自己的知识
2. **引用来源**：每句话后标注引用的文档编号，格式为 [citation:1], [citation:2] 等
3. **诚实**：如果文档中没有相关信息，明确说"根据提供的文档，没有找到相关信息"
4. **结构化**：使用清晰的段落结构，必要时使用列表
5. **完整**：提供足够详细的解释，而非一两句敷衍

## 对话历史
{dialog_context}

## 参考文档片段
{contexts}

## 用户问题
{query}

请基于上面的参考文档回答问题。记得标注引用来源。"""


class LLMGenerator:
    """LLM 答案生成器。

    封装 LLM HTTP 调用和答案后处理（溯源解析）。
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=config.llm_base_url,
                headers={
                    "Authorization": f"Bearer {config.llm_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10),
            )
        return self._client

    async def generate(
        self,
        query: str,
        contexts: list[str],
        citations: list[dict[str, Any]],
        dialog_context: str = "",
    ) -> dict[str, Any]:
        """生成带溯源的答案。

        Args:
            query: 用户问题。
            contexts: 检索到的文档片段文本列表。
            citations: 对应的溯源信息列表。
            dialog_context: 对话历史上下文字符串。

        Returns:
            {"answer": str, "citations": list[Citation], "raw_response": str}
        """
        # 构建上下文编号列表
        ctx_text_parts: list[str] = []
        for i, ctx in enumerate(contexts):
            ctx_text_parts.append(f"[{i + 1}] {ctx}")
        ctx_text = "\n\n".join(ctx_text_parts)

        # 构建 prompt
        user_prompt = GENERATION_SYSTEM_PROMPT.format(
            dialog_context=dialog_context or "（无历史对话）",
            contexts=ctx_text,
            query=query,
        )

        try:
            raw_answer = await self._call_llm(user_prompt)
            parsed_citations = self._parse_citations(raw_answer, citations)
            return {
                "answer": raw_answer,
                "citations": parsed_citations,
                "raw_response": raw_answer,
            }
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return {
                "answer": f"抱歉，答案生成失败：{e}",
                "citations": [],
                "raw_response": "",
            }

    async def _call_llm(self, user_prompt: str) -> str:
        """调用 LLM API。

        带指数退避重试，最多 3 次。
        """
        client = await self._get_client()
        body = {
            "model": config.llm_model,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await client.post("/chat/completions", json=body)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content
            except Exception as e:
                last_error = e
                logger.warning("LLM call attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"LLM failed after 3 attempts: {last_error}")

    @staticmethod
    def _parse_citations(
        answer: str, citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """从答案中解析 [citation:N] 标记并映射为结构化引用。

        Args:
            answer: LLM 生成的包含 [citation:N] 标记的文本。
            citations: 候选引用列表（按上下文顺序排列）。

        Returns:
            实际被引用的 Citation 字典列表（去重）。
        """
        used_ids: set[int] = set()
        for match in re.finditer(r"\[citation:(\d+)\]", answer):
            idx = int(match.group(1)) - 1  # 编号从 1 开始
            if 0 <= idx < len(citations):
                used_ids.add(idx)

        return [citations[i] for i in sorted(used_ids)]

    async def generate_raw(
        self, prompt: str, system_prompt: str = "",
    ) -> str:
        """通用 LLM 调用（供 QueryRewriter 等模块使用）。

        Args:
            prompt: User prompt。
            system_prompt: System prompt。

        Returns:
            LLM 原始文本输出。
        """
        client = await self._get_client()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": config.llm_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        response = await client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
