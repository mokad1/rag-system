"""Query Rewrite 查询改写模块。

使用 LLM 从多个视角扩展用户问题，提升召回覆盖：
1. 关键词化改写：提取核心术语做精确匹配
2. 泛化改写：扩展相关概念和同义词
3. 具体化改写：补充常见上下文信息

原始 query + 3 条改写 → 各自检索 → 合并去重 → Reranker 精排。
"""

from __future__ import annotations

from rag_system.utils.logger import get_logger

logger = get_logger("retriever.rewriter")

REWRITE_SYSTEM_PROMPT = """你是一个专业的搜索查询改写助手。你的任务是将用户的原始问题改写为3个不同视角的检索查询，以提升文档检索的召回率。

## 改写规则
1. **关键词化改写**：提取问题的核心关键词和术语，组成适合关键词检索的查询
2. **泛化改写**：将问题中的具体概念替换为更广泛的相关概念和同义词
3. **具体化改写**：补充问题可能隐含的上下文信息，使其更具体

## 输出格式
你必须严格按照以下 JSON 格式输出，不要输出任何其他内容：
```json
{
  "keyword_query": "关键词化的查询文本",
  "generalized_query": "泛化后的查询文本",
  "specific_query": "具体化的查询文本"
}
```

## 示例
用户问题：Python中如何实现线程池？
输出：
```json
{
  "keyword_query": "Python 线程池 ThreadPoolExecutor concurrent.futures 多线程",
  "generalized_query": "编程语言中并发执行任务的线程管理机制",
  "specific_query": "Python使用concurrent.futures.ThreadPoolExecutor创建和管理线程池的代码示例和最佳实践"
}
```
"""

REWRITE_USER_TEMPLATE = "用户问题：{query}\n\n请改写为3个不同视角的检索查询。"


class QueryRewriter:
    """LLM 驱动的查询改写器。

    对原始查询生成 3 视角改写，用于多路检索融合。
    """

    def __init__(self, llm_generate) -> None:
        """初始化。

        Args:
            llm_generate: async callable (prompt, system_prompt) -> str。
        """
        self._generate = llm_generate
        self._system_prompt = REWRITE_SYSTEM_PROMPT

    async def rewrite(self, query: str) -> list[str]:
        """改写查询，返回 [原始, keyword, generalized, specific]。

        如果 LLM 调用失败，降级返回仅含原始 query 的列表。

        Args:
            query: 用户原始查询。

        Returns:
            查询列表（第一条为原始查询）。
        """
        all_queries = [query]

        try:
            user_prompt = REWRITE_USER_TEMPLATE.format(query=query)
            response = await self._generate(
                prompt=user_prompt,
                system_prompt=self._system_prompt,
            )

            # 解析 JSON 响应
            import json
            # 清理可能的 markdown 包裹
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response.strip())

            for key in ["keyword_query", "generalized_query", "specific_query"]:
                if key in data and data[key] and data[key] not in all_queries:
                    all_queries.append(data[key])

            logger.info("Query rewritten: '%s' → %d queries", query[:50], len(all_queries))

        except Exception as e:
            logger.warning("Query rewrite failed for '%s': %s", query[:50], e)
            # 降级：只返回原始查询

        return all_queries

    def rewrite_sync(self, query: str) -> list[str]:
        """同步版改写（用于评测等非异步场景）。

        注意：此方法需要一个同步的 llm_generate 函数。

        Args:
            query: 用户原始查询。

        Returns:
            查询列表。
        """
        return [query]  # 同步版简化，实际由调用者结合 LLM 实现
