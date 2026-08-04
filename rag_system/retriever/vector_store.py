"""FAISS 向量存储。

封装 FAISS 索引的创建、添加、搜索、保存/加载。
使用内积索引（IndexFlatIP）+ L2 归一化向量 = cosine 相似度。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag_system.config import config
from rag_system.models.document import Chunk
from rag_system.utils.logger import get_logger

logger = get_logger("retriever.vector_store")


class VectorStore:
    """FAISS 向量存储封装。

    存储结构：
    - faiss_index: IndexFlatIP（内积=cosine，因为向量已 L2 归一化）
    - chunk_map: {faiss_id: Chunk} 映射
    - chunk_id_to_faiss_id: {chunk_id: faiss_id} 反向映射
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or config.embedding_dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dim)
        self.chunk_map: dict[int, Chunk] = {}
        self.chunk_id_to_faiss_id: dict[str, int] = {}
        self._next_id: int = 0

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """批量添加 chunk 和对应 embedding。

        Args:
            chunks: Chunk 列表（仅 child/fixed 类型用于检索）。
            embeddings: shape=(len(chunks), dim) 的 embedding 数组。
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        start_id = self._next_id
        self.index.add(embeddings.astype(np.float32))

        for i, chunk in enumerate(chunks):
            faiss_id = start_id + i
            self.chunk_map[faiss_id] = chunk
            self.chunk_id_to_faiss_id[chunk.chunk_id] = faiss_id

        self._next_id += len(chunks)
        logger.info("Added %d vectors to FAISS. Total: %d", len(chunks), self._next_id)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 20,
    ) -> list[tuple[Chunk, float]]:
        """检索最相似的 top_k 个 chunk。

        Args:
            query_embedding: shape=(1, dim) 或 (dim,) 的查询向量。
            top_k: 返回数量。

        Returns:
            [(Chunk, score), ...] 按得分降序排列。
        """
        if self._next_id == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        actual_k = min(top_k, self._next_id)
        scores, indices = self.index.search(query, actual_k)

        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx not in self.chunk_map:
                continue
            results.append((self.chunk_map[idx], float(score)))

        return results

    def save(self, path: Path) -> None:
        """保存 FAISS 索引和 chunk 映射到磁盘。

        Args:
            path: 保存目录路径。
        """
        path.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self.index, str(path / "faiss.index"))

        # 保存映射和元数据
        meta = {
            "dim": self.dim,
            "next_id": self._next_id,
            "chunk_map": {k: v.model_dump(mode="json") for k, v in self.chunk_map.items()},
            "chunk_id_to_faiss_id": self.chunk_id_to_faiss_id,
        }
        with open(path / "store_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("Saved vector store to %s (%d vectors)", path, self._next_id)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """从磁盘加载 FAISS 索引和 chunk 映射。

        Args:
            path: 保存目录路径。

        Returns:
            VectorStore 实例。
        """
        from rag_system.models.document import Chunk

        store = cls.__new__(cls)
        store.index = faiss.read_index(str(path / "faiss.index"))

        with open(path / "store_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        store.dim = meta["dim"]
        store._next_id = meta["next_id"]
        store.chunk_map = {
            int(k): Chunk.model_validate(v)
            for k, v in meta["chunk_map"].items()
        }
        store.chunk_id_to_faiss_id = meta["chunk_id_to_faiss_id"]

        logger.info("Loaded vector store from %s (%d vectors)", path, store._next_id)
        return store

    @property
    def count(self) -> int:
        return self._next_id

    def get_parent_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """根据 child chunk_id 获取对应的 parent chunk。

        用于 Parent-Child 检索：child 命中后返回 parent 作为上下文。

        Args:
            chunk_ids: child chunk ID 列表。

        Returns:
            去重后的 parent Chunk 列表。
        """
        parents: dict[str, Chunk] = {}
        for cid in chunk_ids:
            faiss_id = self.chunk_id_to_faiss_id.get(cid)
            if faiss_id is None:
                continue
            child = self.chunk_map[faiss_id]
            if child.parent_id:
                # 查找 parent
                for faiss_id_p, chunk_p in self.chunk_map.items():
                    if chunk_p.chunk_id == child.parent_id:
                        parents[chunk_p.chunk_id] = chunk_p
                        break
            else:
                # 非 child（fixed 类型），直接返回
                parents[child.chunk_id] = child
        return list(parents.values())
