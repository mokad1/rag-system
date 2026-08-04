"""父子分块器（Parent-Child Chunking）。

核心思路：
- Parent chunk（大块 800-1000 字符）：提供完整上下文给 LLM
- Child chunk（小块 200-300 字符）：用于向量检索，精度更高
- 每个 child 通过 parent_id 关联其 parent
- 检索流程：child 被命中 → 返回对应 parent 作为 LLM 上下文

相比固定分块的优势：
- 小块检索更精准（不会因为上下文过长而稀释语义向量）
- 大块上下文更完整（LLM 看到的不是碎片）
"""

from __future__ import annotations

from rag_system.chunker.text_cleaner import TextCleaner
from rag_system.config import config
from rag_system.models.document import Chunk, Document
from rag_system.utils.logger import get_logger

logger = get_logger("chunker.parent_child")


class ParentChildChunker:
    """父子分块器。

    先切大块（parent），再将每个 parent 细分为 child。
    返回全部 chunk（parent + child），child 携带 parent_id。
    """

    def __init__(
        self,
        parent_size: int | None = None,
        child_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self.parent_size = parent_size or config.parent_chunk_size
        self.child_size = child_size or config.child_chunk_size
        self.overlap = overlap or config.chunk_overlap
        self.cleaner = TextCleaner()

    def chunk(self, document: Document) -> list[Chunk]:
        """执行父子分块。

        步骤：
        1. 先按 parent_size 切大块（带 overlap）
        2. 再将每个 parent 按 child_size 细切
        3. child 记录其 parent_id
        4. 返回所有 chunk（parent 用于 LLM 上下文，child 用于检索）

        Args:
            document: 入库文档。

        Returns:
            全部 Chunk 列表（含 parent 和 child 类型）。
        """
        text = self.cleaner.clean(document.raw_text)
        all_chunks: list[Chunk] = []

        # Step 1: 先切 parent 块
        parents: list[tuple[int, str, int]] = []  # (idx, text, page)
        parent_step = self.parent_size - self.overlap
        if parent_step <= 0:
            parent_step = self.parent_size // 2

        p_idx = 0
        for start in range(0, len(text), parent_step):
            end = min(start + self.parent_size, len(text))
            parent_text = text[start:end]
            page = self._estimate_page(text[:start])
            parents.append((p_idx, parent_text, page))
            p_idx += 1
            if end >= len(text):
                break

        # Step 2: 将每个 parent 细分为 child
        c_idx = 0
        child_step = self.child_size - self.overlap
        if child_step <= 0:
            child_step = self.child_size // 2

        for p_idx, parent_text, page in parents:
            parent_id = f"{document.doc_id}_parent_{p_idx}"

            # 添加 parent chunk
            all_chunks.append(Chunk(
                chunk_id=parent_id,
                doc_id=document.doc_id,
                content=parent_text,
                chunk_type="parent",
                page=page,
                chunk_index=p_idx,
                source_doc=document.file_path,
            ))

            # 细切 child（仅当 parent 长度超过 child_size 时）
            if len(parent_text) <= self.child_size:
                child_chunk = Chunk(
                    chunk_id=f"{document.doc_id}_child_{c_idx}",
                    doc_id=document.doc_id,
                    content=parent_text,
                    chunk_type="child",
                    parent_id=parent_id,
                    page=page,
                    chunk_index=c_idx,
                    source_doc=document.file_path,
                )
                all_chunks.append(child_chunk)
                c_idx += 1
            else:
                for cs in range(0, len(parent_text), child_step):
                    ce = min(cs + self.child_size, len(parent_text))
                    child_text = parent_text[cs:ce]
                    child_chunk = Chunk(
                        chunk_id=f"{document.doc_id}_child_{c_idx}",
                        doc_id=document.doc_id,
                        content=child_text,
                        chunk_type="child",
                        parent_id=parent_id,
                        page=page,
                        chunk_index=c_idx,
                        source_doc=document.file_path,
                    )
                    all_chunks.append(child_chunk)
                    c_idx += 1

        parent_count = sum(1 for c in all_chunks if c.chunk_type == "parent")
        child_count = sum(1 for c in all_chunks if c.chunk_type == "child")
        logger.info(
            "Parent-Child chunker: %d parents + %d children from %s",
            parent_count, child_count, document.doc_id,
        )
        return all_chunks

    @staticmethod
    def _estimate_page(text_before: str) -> int:
        """通过 [P数字] 标记估算当前页码。"""
        import re
        markers = re.findall(r"\[P(\d+)\]", text_before)
        return int(markers[-1]) if markers else 1
