"""混合检索器：BM25 + 向量检索 → RRF 融合。

Reciprocal Rank Fusion (RRF) 算法：
  score(d) = Σ_{i ∈ retrievers} 1 / (k + rank_i(d))

其中 k=60（经典取值），rank_i(d) 是文档 d 在检索器 i 中的排名。
RRF 不需要归一化原始得分，对得分尺度差异鲁棒。
"""

from __future__ import annotations

import time

import numpy as np

from rag_system.config import config
from rag_system.embedder.bge_embedder import BGEEmbedder
from rag_system.models.document import Chunk
from rag_system.models.retrieval import RetrievalResult, SearchResult
from rag_system.retriever.bm25_index import BM25Index
from rag_system.retriever.vector_store import VectorStore
from rag_system.utils.logger import get_logger

logger = get_logger("retriever.hybrid")


class HybridRetriever:
    """混合检索器。

    结合 BM25 关键词检索和 FAISS 向量检索，通过 RRF 融合排序。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: BGEEmbedder,
        rrf_k: int = 60,
    ) -> None:
        """初始化混合检索器。

        Args:
            vector_store: FAISS 向量存储。
            bm25_index: BM25 关键词索引。
            embedder: BGE Embedding 模型。
            rrf_k: RRF 融合参数（默认 60）。
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        """执行混合检索。

        Args:
            query: 查询文本。
            top_k: 最终返回数（默认从 config 读）。
            use_hybrid: True=混合检索，False=纯向量检索（消融实验用）。

        Returns:
            RetrievalResult 包含排序后的结果。
        """
        t_start = time.perf_counter()
        final_k = top_k or config.hybrid_top_k

        if use_hybrid:
            # 并行执行向量检索和 BM25 检索
            query_emb = self.embedder.encode_query(query)

            vector_results = self.vector_store.search(query_emb, top_k=final_k)
            bm25_results = self.bm25_index.search(query, top_k=final_k)

            # RRF 融合
            merged = self._rrf_fusion(vector_results, bm25_results, top_k=final_k)
        else:
            # 纯向量检索（基线模式）
            query_emb = self.embedder.encode_query(query)
            vector_results = self.vector_store.search(query_emb, top_k=final_k)
            merged = [
                SearchResult(chunk=c, score=s, source="vector")
                for c, s in vector_results
            ]

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        logger.debug(
            "Hybrid search: '%s' → %d results (%.1f ms, hybrid=%s)",
            query[:50], len(merged), elapsed_ms, use_hybrid,
        )

        return RetrievalResult(
            query=query,
            results=merged[:final_k],
            top_k=final_k,
            elapsed_ms=round(elapsed_ms, 1),
        )

    def search_multi(
        self,
        queries: list[str],
        top_k_per_query: int = 10,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        """多查询检索（Query Rewrite 场景）。

        每个查询独立检索，结果合并去重，按 RRF 排名。

        Args:
            queries: 多个查询文本（原始 + 改写）。
            top_k_per_query: 每个查询的召回数。
            use_hybrid: 是否混合检索。

        Returns:
            合并去重后的 RetrievalResult。
        """
        all_results: list[SearchResult] = []
        chunk_scores: dict[str, float] = {}  # chunk_id → 累计 RRF 分数

        for qi, q in enumerate(queries):
            r = self.search(q, top_k=top_k_per_query, use_hybrid=use_hybrid)
            all_results.extend(r.results)
            for rank, sr in enumerate(r.results):
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                cid = sr.chunk.chunk_id
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rrf_score

        # 去重：按累计 RRF 分数重排
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for sr in sorted(all_results, key=lambda x: chunk_scores.get(x.chunk.chunk_id, 0), reverse=True):
            cid = sr.chunk.chunk_id
            if cid not in seen:
                seen.add(cid)
                merged.append(SearchResult(
                    chunk=sr.chunk,
                    score=chunk_scores[cid],
                    source=sr.source,
                ))

        final_k = config.hybrid_top_k
        return RetrievalResult(
            query=queries[0],
            rewritten_queries=queries[1:],
            results=merged[:final_k],
            top_k=final_k,
            elapsed_ms=0,  # 多查询总耗时在调用处设置
        )

    def _rrf_fusion(
        self,
        vector_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[SearchResult]:
        """RRF 融合两个检索列表。

        思路：
        1. 给每个检索结果按 rank 分配 RRF 分数
        2. 同一文档出现在两个列表中的，累加 RRF 分数
        3. 按累加分数降序排列
        """
        rrf_scores: dict[str, float] = {}
        source_map: dict[str, str] = {}
        chunk_map: dict[str, Chunk] = {}

        # 向量检索贡献
        for rank, (chunk, _) in enumerate(vector_results):
            rrf = 1.0 / (self.rrf_k + rank + 1)
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf
            source_map[cid] = "hybrid"  # 两个源都命中
            chunk_map[cid] = chunk

        # BM25 检索贡献
        for rank, (chunk, _) in enumerate(bm25_results):
            rrf = 1.0 / (self.rrf_k + rank + 1)
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf
            if cid not in source_map:
                source_map[cid] = "bm25"
                chunk_map[cid] = chunk
            else:
                source_map[cid] = "hybrid"

        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        return [
            SearchResult(
                chunk=chunk_map[cid],
                score=rrf_scores[cid],
                source=source_map[cid],
            )
            for cid in sorted_ids
        ]
