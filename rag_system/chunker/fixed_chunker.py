"""固定长度分块器（消融实验基线用）。

按固定字符数切分，带 overlap 保持上下文连贯。
"""

from __future__ import annotations

from rag_system.chunker.text_cleaner import TextCleaner
from rag_system.config import config
from rag_system.models.document import Chunk, Document
from rag_system.utils.logger import get_logger

logger = get_logger("chunker.fixed")


class FixedChunker:
    """固定长度分块。

    基线系统使用此分块器，与 Parent-Child 对比评测召回差距。
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or config.fixed_chunk_size
        self.overlap = overlap or config.chunk_overlap
        self.cleaner = TextCleaner()

    def chunk(self, document: Document) -> list[Chunk]:
        """将文档切分为固定大小的 chunk 列表。

        思路：滑动窗口，步长 = chunk_size - overlap。
        每个 chunk 带完整的溯源信息。
        """
        text = self.cleaner.clean(document.raw_text)
        chunks: list[Chunk] = []

        if len(text) <= self.chunk_size:
            chunks.append(self._make_chunk(
                doc=document, content=text, idx=0, page=1,
            ))
            return chunks

        step = self.chunk_size - self.overlap
        if step <= 0:
            step = self.chunk_size // 2

        idx = 0
        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # 估算页码（基于 [P数字] 标记出现次数）
            page = self._estimate_page(text[:start])

            chunks.append(self._make_chunk(
                doc=document, content=chunk_text, idx=idx, page=page,
            ))
            idx += 1

            if end >= len(text):
                break

        logger.info(
            "Fixed chunker: %d chunks from %s (size=%d, overlap=%d)",
            len(chunks), document.doc_id, self.chunk_size, self.overlap,
        )
        return chunks

    def _make_chunk(
        self, doc: Document, content: str, idx: int, page: int,
    ) -> Chunk:
        """创建 Chunk 对象。"""
        return Chunk(
            chunk_id=f"{doc.doc_id}_fixed_{idx}",
            doc_id=doc.doc_id,
            content=content,
            chunk_type="fixed",
            page=page,
            chunk_index=idx,
            source_doc=doc.file_path,
        )

    @staticmethod
    def _estimate_page(text_before: str) -> int:
        """通过 [P数字] 标记估算当前页码。"""
        import re
        markers = re.findall(r"\[P(\d+)\]", text_before)
        return int(markers[-1]) if markers else 1
