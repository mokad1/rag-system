"""文档入库管线。

完整流程：
PDF/MD 文件 → Loader → TextCleaner → Chunker → Embedder → FAISS + BM25
"""

from __future__ import annotations

from pathlib import Path

from rag_system.chunker.fixed_chunker import FixedChunker
from rag_system.chunker.parent_child_chunker import ParentChildChunker
from rag_system.embedder.bge_embedder import BGEEmbedder
from rag_system.loader.md_loader import MarkdownLoader
from rag_system.loader.pdf_loader import PDFLoader
from rag_system.models.document import Chunk, Document
from rag_system.retriever.bm25_index import BM25Index
from rag_system.retriever.vector_store import VectorStore
from rag_system.utils.logger import get_logger

logger = get_logger("pipeline.indexing")

# 文件后缀 → Loader 映射
_LOADER_MAP = {
    ".pdf": PDFLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
}


class IndexingPipeline:
    """文档入库管线。

    支持父子分块（优化版）和固定分块（基线版）两种模式。

    用法：
        pipeline = IndexingPipeline(embedder, vector_store, bm25_index)
        pipeline.index_directory("docs/", use_parent_child=True)
    """

    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.parent_child_chunker = ParentChildChunker()
        self.fixed_chunker = FixedChunker()

    def index_file(
        self,
        file_path: Path,
        use_parent_child: bool = True,
    ) -> Document:
        """索引单个文件。

        Args:
            file_path: 文件路径。
            use_parent_child: True=父子分块，False=固定分块。

        Returns:
            处理后的 Document。
        """
        suffix = file_path.suffix.lower()
        loader_cls = _LOADER_MAP.get(suffix)
        if loader_cls is None:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Step 1: 加载文档
        loader = loader_cls()
        doc = loader.load(file_path)

        # Step 2: 分块
        if use_parent_child:
            all_chunks = self.parent_child_chunker.chunk(doc)
        else:
            all_chunks = self.fixed_chunker.chunk(doc)

        if not all_chunks:
            logger.warning("No chunks generated for %s", file_path.name)
            return doc

        # Step 3: 只对 child/fixed 类型 chunk 做 embedding 和索引
        index_chunks = [
            c for c in all_chunks
            if c.chunk_type in ("child", "fixed")
        ]
        texts = [c.content for c in index_chunks]

        # Step 4: Embedding + 入库
        embeddings = self.embedder.encode_documents(texts)
        self.vector_store.add(index_chunks, embeddings)
        self.bm25_index.add(index_chunks)

        logger.info(
            "Indexed %s: %d chunks → FAISS + BM25",
            file_path.name, len(index_chunks),
        )
        return doc

    def index_directory(
        self,
        directory: Path | str,
        use_parent_child: bool = True,
        recursive: bool = True,
    ) -> list[Document]:
        """批量索引目录下的所有支持文档。

        Args:
            directory: 目录路径。
            use_parent_child: True=父子分块。
            recursive: 是否递归子目录。

        Returns:
            Document 列表。
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        docs: list[Document] = []

        for file_path in dir_path.glob(pattern):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix not in _LOADER_MAP:
                continue

            try:
                doc = self.index_file(file_path, use_parent_child=use_parent_child)
                docs.append(doc)
            except Exception as e:
                logger.error("Failed to index %s: %s", file_path, e)

        logger.info(
            "Indexed %d files: %d total vectors, %d BM25 docs",
            len(docs), self.vector_store.count, self.bm25_index.count,
        )
        return docs

    def save(self, directory: Path | str) -> None:
        """持久化所有索引到磁盘。"""
        dir_path = Path(directory)
        self.vector_store.save(dir_path / "faiss")
        self.bm25_index.save(dir_path / "bm25")
        logger.info("Index saved to %s", dir_path)

    @staticmethod
    def load(
        directory: Path | str,
        embedder: BGEEmbedder,
    ) -> "IndexingPipeline":
        """从磁盘加载索引。"""
        dir_path = Path(directory)
        vs = VectorStore.load(dir_path / "faiss")
        bm25 = BM25Index.load(dir_path / "bm25")
        return IndexingPipeline(embedder, vs, bm25)
