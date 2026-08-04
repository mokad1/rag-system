"""BGE Reranker 重排序模块。

使用 bge-reranker-base (Cross-Encoder) 对候选 chunk 精排。
Cross-Encoder 将 (query, chunk) 拼接后一起编码，计算细粒度相关性，
相比 Bi-Encoder（向量检索）精度更高，但速度较慢。
因此只在粗排后的 top_k 候选上运行。
"""

from __future__ import annotations

import numpy as np
from FlagEmbedding import FlagReranker

from rag_system.config import config
from rag_system.models.document import Chunk
from rag_system.utils.logger import get_logger

logger = get_logger("retriever.reranker")


class Reranker:
    """BGE Cross-Encoder 重排序器。

    对 (query, chunk_text) 对打分，按得分降序重排。
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """初始化重排序器。

        Args:
            model_name: HuggingFace 模型名，默认 bge-reranker-base。
            device: 设备（cpu/cuda）。
        """
        self.model_name = model_name or config.reranker_model_name
        self.device = device or config.reranker_device

        logger.info("Loading reranker model: %s on %s", self.model_name, self.device)
        self._model = FlagReranker(
            self.model_name,
            use_fp16=(self.device != "cpu"),
            device=self.device,
        )
        logger.info("Reranker model loaded.")

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """对候选 chunks 重排序。

        Args:
            query: 查询文本。
            chunks: 候选 Chunk 列表。
            top_k: 返回数量（默认 config.final_top_k）。

        Returns:
            [(Chunk, rerank_score), ...] 按得分降序。
        """
        final_k = top_k or config.final_top_k

        if not chunks:
            return []

        # 构建 (query, chunk_text) 对
        pairs = [[query, c.content] for c in chunks]
        scores = self._model.compute_score(pairs, normalize=True)

        # 处理单个结果的情况
        if isinstance(scores, float):
            scores = [scores]

        # 按得分排序
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:final_k]

    def rerank_search_results(
        self,
        query: str,
        chunks_with_scores: list[tuple[Chunk, float]],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """对已排序的检索结果重排（保留原始得分作为参考）。

        Args:
            query: 查询文本。
            chunks_with_scores: [(chunk, original_score), ...]。
            top_k: 返回数量。

        Returns:
            [(Chunk, rerank_score), ...] 按重排得分降序。
        """
        chunks = [c for c, _ in chunks_with_scores]
        return self.rerank(query, chunks, top_k)
