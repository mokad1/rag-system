"""RAG 管线：文档入库 + 查询问答。"""

from rag_system.pipeline.indexing import IndexingPipeline
from rag_system.pipeline.query import QueryPipeline

__all__ = ["IndexingPipeline", "QueryPipeline"]
