"""文档与 Chunk 数据结构。

设计要点：
- Chunk 携带完整的溯源信息（source_doc, page, chunk_index）
- Parent-Child 关系通过 parent_id 字段维护
- Document 是入库的基本单位
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """入库文档。"""
    doc_id: str = Field(..., description="文档唯一标识")
    file_path: str = Field(..., description="原始文件路径")
    file_type: Literal["pdf", "markdown"] = Field(..., description="文件类型")
    raw_text: str = Field(default="", description="原始文本内容")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="文档元数据（标题、作者等）"
    )
    page_count: int = Field(default=0, description="总页数")


class Citation(BaseModel):
    """答案溯源引用。"""
    chunk_id: str = Field(..., description="引用 Chunk ID")
    source_doc: str = Field(..., description="来源文档名")
    page: int = Field(default=0, description="页码")
    text: str = Field(..., description="引用原文片段（截取 200 字）")
    score: float = Field(default=0.0, description="检索/重排得分")


class Chunk(BaseModel):
    """文档分块。

    chunk_type 区分 parent/child：
    - parent：大块，用于 LLM 上下文，不被检索
    - child：小块，用于向量检索
    """
    chunk_id: str = Field(..., description="Chunk 唯一标识")
    doc_id: str = Field(..., description="所属文档 ID")
    content: str = Field(..., description="文本内容")
    chunk_type: Literal["parent", "child", "fixed"] = Field(
        default="fixed", description="分块类型"
    )
    parent_id: str | None = Field(
        default=None, description="父块 ID（child 类型时必填）"
    )
    page: int = Field(default=0, description="所在页码")
    chunk_index: int = Field(default=0, description="在文档中的序号")
    source_doc: str = Field(default="", description="来源文件名")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation_text(self) -> str:
        """截取前 200 字符作为引用片段。"""
        return self.content[:200]
