"""文档加载器抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from rag_system.models.document import Document


class BaseLoader(ABC):
    """文档加载器抽象。

    所有加载器实现 load() 方法，返回统一的 Document 模型。
    """

    @abstractmethod
    def load(self, file_path: Path) -> Document:
        """加载单个文件为 Document。"""
        ...

    @abstractmethod
    def supported_suffixes(self) -> list[str]:
        """返回支持的文件后缀列表。"""
        ...
