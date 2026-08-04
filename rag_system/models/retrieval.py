"""检索结果数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_system.models.document import Chunk


class SearchResult(BaseModel):
    """单条检索结果。"""
    chunk: Chunk = Field(..., description="检索到的 Chunk")
    score: float = Field(..., description="相关性得分")
    source: str = Field(default="vector", description="来源：vector/bm25/rerank")


class RetrievalResult(BaseModel):
    """完整检索结果集。"""
    query: str = Field(..., description="原始查询")
    rewritten_queries: list[str] = Field(
        default_factory=list, description="改写后的查询列表"
    )
    results: list[SearchResult] = Field(
        default_factory=list, description="检索结果（按得分降序）"
    )
    top_k: int = Field(default=5, description="返回条数")
    elapsed_ms: float = Field(default=0.0, description="检索耗时（毫秒）")

    @property
    def contexts(self) -> list[str]:
        """提取检索结果的文本列表（供 LLM 使用）。"""
        return [r.chunk.content for r in self.results[:self.top_k]]

    @property
    def citations(self) -> list[dict[str, object]]:
        """提取溯源信息列表。"""
        from rag_system.models.document import Citation

        return [
            Citation(
                chunk_id=r.chunk.chunk_id,
                source_doc=r.chunk.source_doc,
                page=r.chunk.page,
                text=r.chunk.citation_text,
                score=r.score,
            )
            for r in self.results[:self.top_k]
        ]
