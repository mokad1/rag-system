"""查询问答管线。

完整流程：
Query → QueryRewriter → HybridRetriever → Reranker → ContextAssembly → LLM Generator → Answer + Citations

支持消融配置：逐项开关 Parent-Child / Hybrid / Rerank / Rewrite / Memory。
"""

from __future__ import annotations

import time
from typing import Any

from rag_system.config import config
from rag_system.generator.llm_generator import LLMGenerator
from rag_system.memory.layered_memory import LayeredMemory
from rag_system.models.retrieval import RetrievalResult
from rag_system.retriever.hybrid_retriever import HybridRetriever
from rag_system.retriever.query_rewriter import QueryRewriter
from rag_system.retriever.reranker import Reranker
from rag_system.retriever.vector_store import VectorStore
from rag_system.utils.logger import get_logger

logger = get_logger("pipeline.query")


class QueryPipeline:
    """RAG 查询问答管线。

    整合检索、重排、改写、记忆和生成全链路。

    用法：
        qp = QueryPipeline(hybrid_retriever, reranker, generator, memory, vector_store)
        answer = await qp.ask("Python中如何实现线程池？", session_id="demo")
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        generator: LLMGenerator,
        memory: LayeredMemory,
        vector_store: VectorStore,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.memory = memory
        self.vector_store = vector_store

        # Query Rewriter 使用 generator 的 LLM 调用能力
        self.query_rewriter = QueryRewriter(
            llm_generate=generator.generate_raw,
        )

    async def ask(
        self,
        query: str,
        session_id: str = "default",
        # 消融开关
        enable_rewrite: bool = True,
        enable_hybrid: bool = True,
        enable_rerank: bool = True,
        enable_memory: bool = True,
        enable_parent_child: bool = True,
    ) -> dict[str, Any]:
        """执行完整 RAG 问答流程。

        Args:
            query: 用户问题。
            session_id: 会话 ID。
            enable_*: 消融开关，False 时回退到基线行为。

        Returns:
            {
                "answer": str,
                "citations": list[dict],
                "retrieval_result": RetrievalResult,
                "elapsed_ms": float,
                "rewritten_queries": list[str],
            }
        """
        t_start = time.perf_counter()

        # ── Step 1: Query Rewrite ────────────────────
        all_queries = [query]
        if enable_rewrite:
            all_queries = await self.query_rewriter.rewrite(query)

        # ── Step 2: 多路检索 ─────────────────────────
        retrieval = self.hybrid_retriever.search_multi(
            queries=all_queries,
            use_hybrid=enable_hybrid,
        )

        # ── Step 3: Parent-Child 映射 ────────────────
        # 如果命中 child chunk，替换为 parent chunk（更完整上下文）
        if enable_parent_child:
            retrieval = self._resolve_parents(retrieval)

        # ── Step 4: Reranker 精排 ────────────────────
        if enable_rerank and retrieval.results:
            chunks = [sr.chunk for sr in retrieval.results]
            reranked = self.reranker.rerank(query, chunks, top_k=config.final_top_k)
            # 重建 SearchResult 列表
            from rag_system.models.retrieval import SearchResult
            retrieval.results = [
                SearchResult(chunk=c, score=s, source="rerank")
                for c, s in reranked
            ]

        # ── Step 5: 对话记忆 ─────────────────────────
        dialog_context = ""
        if enable_memory:
            dialog_context = self.memory.get_context(session_id)

        # ── Step 6: LLM 答案生成 ─────────────────────
        contexts = retrieval.contexts
        citations_raw = [
            {
                "chunk_id": sr.chunk.chunk_id,
                "source_doc": sr.chunk.source_doc,
                "page": sr.chunk.page,
                "text": sr.chunk.citation_text,
                "score": sr.score,
            }
            for sr in retrieval.results[:config.final_top_k]
        ]

        gen_result = await self.generator.generate(
            query=query,
            contexts=contexts,
            citations=citations_raw,
            dialog_context=dialog_context,
        )

        # ── Step 7: 更新记忆 ─────────────────────────
        if enable_memory:
            self.memory.add_message(session_id, "user", query)
            self.memory.add_message(
                session_id, "assistant",
                gen_result["answer"],
                gen_result.get("citations", []),
            )
            # 异步触发可能的摘要压缩
            await self.memory.maybe_summarize(
                session_id,
                self.generator.generate_raw,
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        retrieval.elapsed_ms = round(elapsed_ms, 1)
        retrieval.rewritten_queries = all_queries[1:] if enable_rewrite else []

        logger.info(
            "Query answered in %.0fms: '%s' → %d citations (rewrite=%s, hybrid=%s, rerank=%s)",
            elapsed_ms, query[:50],
            len(gen_result.get("citations", [])),
            enable_rewrite, enable_hybrid, enable_rerank,
        )

        return {
            "answer": gen_result["answer"],
            "citations": gen_result.get("citations", []),
            "retrieval_result": retrieval,
            "elapsed_ms": elapsed_ms,
            "rewritten_queries": all_queries[1:],
        }

    def _resolve_parents(self, retrieval: RetrievalResult) -> RetrievalResult:
        """将 child chunk 替换为 parent chunk。

        思路：检索命中 child → 通过 parent_id 找 parent → 用 parent 替换。
        这样 LLM 看到的是完整段落而非碎片。
        """
        from rag_system.models.retrieval import SearchResult

        resolved: list[SearchResult] = []
        seen_parents: set[str] = set()

        for sr in retrieval.results:
            chunk = sr.chunk
            if chunk.parent_id and chunk.chunk_type == "child":
                # 查找对应 parent
                parent = None
                for faiss_id, c in self.vector_store.chunk_map.items():
                    if c.chunk_id == chunk.parent_id:
                        parent = c
                        break
                if parent and parent.chunk_id not in seen_parents:
                    resolved.append(SearchResult(
                        chunk=parent, score=sr.score, source=sr.source,
                    ))
                    seen_parents.add(parent.chunk_id)
                elif not parent:
                    # parent 不在缓存中，保留原始 child
                    resolved.append(sr)
            else:
                if chunk.chunk_id not in seen_parents:
                    resolved.append(sr)
                    seen_parents.add(chunk.chunk_id)

        retrieval.results = resolved
        return retrieval

    def ask_sync(
        self,
        query: str,
        session_id: str = "default",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """同步版 ask（供 Streamlit 使用）。"""
        import asyncio
        return asyncio.run(self.ask(query, session_id, **kwargs))
