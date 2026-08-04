"""Markdown 文档加载器。

支持 .md 文件的解析，保留标题层级结构作为元数据。
"""

from __future__ import annotations

from pathlib import Path

from rag_system.loader.base import BaseLoader
from rag_system.models.document import Document
from rag_system.utils.logger import get_logger

logger = get_logger("loader.md")


class MarkdownLoader(BaseLoader):
    """Markdown 文档加载器。

    完整读取 .md 文件内容，提取一级标题作为文档标题。
    """

    def load(self, file_path: Path) -> Document:
        """加载 Markdown 文件。

        Args:
            file_path: .md 文件路径。

        Returns:
            Document：包含全文内容。
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        doc_id = file_path.stem
        logger.info("Loading Markdown: %s", file_path.name)

        text = file_path.read_text(encoding="utf-8")

        # 提取一级标题作为文档标题
        title = ""
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                title = line[2:].strip()
                break

        logger.info("Markdown loaded: %s, %d chars", file_path.name, len(text))

        return Document(
            doc_id=doc_id,
            file_path=str(file_path),
            file_type="markdown",
            raw_text=text,
            metadata={"title": title},
            page_count=1,
        )

    def supported_suffixes(self) -> list[str]:
        return [".md", ".markdown"]
