"""BM25 关键词检索索引。

使用 jieba 分词 + rank_bm25 库实现稀疏检索。
与向量检索互补：BM25 擅长精确关键词匹配，向量擅长语义相似。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from rag_system.models.document import Chunk
from rag_system.utils.logger import get_logger

logger = get_logger("retriever.bm25")


class BM25Index:
    """BM25 关键词检索索引。

    内部维护：
    - _chunks: 所有已索引的 chunk 列表
    - _bm25: rank_bm25 的 BM25Okapi 实例
    - _tokenized_corpus: 所有文本的 jieba 分词结果
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def add(self, chunks: list[Chunk]) -> None:
        """批量添加 chunk 并重建 BM25 索引。

        Args:
            chunks: 要添加的 Chunk 列表。
        """
        new_tokens = [self._tokenize(c.content) for c in chunks]
        self._chunks.extend(chunks)
        self._tokenized_corpus.extend(new_tokens)

        # 重建 BM25 索引（rank_bm25 不支持增量更新）
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index rebuilt: %d documents", len(self._chunks))

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """搜索最相关的 top_k 个 chunk。

        Args:
            query: 查询文本。
            top_k: 返回数量。

        Returns:
            [(Chunk, bm25_score), ...] 按得分降序排列。
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 取 top_k 个最高分
        actual_k = min(top_k, len(scores))
        # numpy argsort 返回升序，取最大的 actual_k 个
        top_indices = scores.argsort()[-actual_k:][::-1]

        results: list[tuple[Chunk, float]] = []
        for idx in top_indices:
            results.append((self._chunks[idx], float(scores[idx])))

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """使用 jieba 搜索引擎模式分词。"""
        return list(jieba.cut_for_search(text))

    @property
    def count(self) -> int:
        return len(self._chunks)

    def save(self, path: Path) -> None:
        """保存 BM25 索引到磁盘（保存 tokenized corpus 和 chunks）。"""
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "tokenized_corpus": self._tokenized_corpus,
            "chunks": [c.model_dump(mode="json") for c in self._chunks],
        }
        with open(path / "bm25_index.pkl", "wb") as f:
            pickle.dump(data, f)
        logger.info("BM25 index saved to %s (%d docs)", path, len(self._chunks))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """从磁盘加载 BM25 索引。"""
        from rag_system.models.document import Chunk

        with open(path / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)

        inst = cls.__new__(cls)
        inst._tokenized_corpus = data["tokenized_corpus"]
        inst._chunks = [Chunk.model_validate(c) for c in data["chunks"]]
        inst._bm25 = BM25Okapi(inst._tokenized_corpus)
        logger.info("BM25 index loaded from %s (%d docs)", path, len(inst._chunks))
        return inst
