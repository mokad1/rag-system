"""BGE Embedding 模型封装。

使用 sentence-transformers 加载 bge-small-zh，输出 384 维向量。
支持批量编码以充分利用 GPU/CPU 并行。
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_system.config import config
from rag_system.utils.logger import get_logger

logger = get_logger("embedder.bge")


class BGEEmbedder:
    """BGE 系列 Embedding 模型封装。

    默认 bge-small-zh-v1.5，384 维输出。
    对 query 自动添加 BGE 格式的 instruction prefix。
    """

    # BGE 模型要求的 query 前缀
    QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """初始化 Embedding 模型。

        Args:
            model_name: HuggingFace 模型名，默认从 config 读取。
            device: 设备（cpu/cuda），默认从 config 读取。
        """
        self.model_name = model_name or config.embedding_model_name
        self.device = device or config.embedding_device
        self.dim = config.embedding_dim

        logger.info("Loading embedding model: %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        # warm up
        _ = self._model.encode(["warmup"], show_progress_bar=False)
        logger.info("Embedding model loaded. Dim=%d", self.dim)

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文档文本。

        Args:
            texts: 文本列表。
            batch_size: 编码批次大小。

        Returns:
            shape=(len(texts), dim) 的 numpy 数组。
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 归一化，便于 FAISS 内积 = cosine
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询文本（自动添加 BGE instruction prefix）。

        Args:
            query: 用户查询文本。

        Returns:
            shape=(1, dim) 的 numpy 数组。
        """
        text = f"{self.QUERY_INSTRUCTION}{query}"
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """批量编码多个查询（用于 Query Rewrite 多路检索）。

        Args:
            queries: 查询文本列表。

        Returns:
            shape=(len(queries), dim) 的 numpy 数组。
        """
        texts = [f"{self.QUERY_INSTRUCTION}{q}" for q in queries]
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
