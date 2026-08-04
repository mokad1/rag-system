"""FastAPI 后端服务。

提供：
- POST /index — 索引文档目录
- POST /ask  — RAG 问答
- GET  /stats — 索引状态查询
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_system.config import config
from rag_system.embedder.bge_embedder import BGEEmbedder
from rag_system.generator.llm_generator import LLMGenerator
from rag_system.memory.layered_memory import LayeredMemory
from rag_system.pipeline.indexing import IndexingPipeline
from rag_system.pipeline.query import QueryPipeline
from rag_system.retriever.bm25_index import BM25Index
from rag_system.retriever.hybrid_retriever import HybridRetriever
from rag_system.retriever.reranker import Reranker
from rag_system.retriever.vector_store import VectorStore
from rag_system.utils.logger import setup_logging

# ── 全局组件（延迟初始化）───────────────────────────────────

_embedder: BGEEmbedder | None = None
_vector_store: VectorStore | None = None
_bm25_index: BM25Index | None = None
_indexing: IndexingPipeline | None = None
_query_pipeline: QueryPipeline | None = None
_memory: LayeredMemory | None = None
_generator: LLMGenerator | None = None
_reranker: Reranker | None = None
_hybrid_retriever: HybridRetriever | None = None


def _get_components() -> dict[str, Any]:
    """懒加载全局组件。"""
    global _embedder, _vector_store, _bm25_index, _indexing
    global _query_pipeline, _memory, _generator, _reranker, _hybrid_retriever

    if _embedder is None:
        setup_logging()
        _embedder = BGEEmbedder()
        _vector_store = VectorStore()
        _bm25_index = BM25Index()
        _indexing = IndexingPipeline(_embedder, _vector_store, _bm25_index)
        _hybrid_retriever = HybridRetriever(_vector_store, _bm25_index, _embedder)
        _reranker = Reranker()
        _generator = LLMGenerator()
        _memory = LayeredMemory()
        _query_pipeline = QueryPipeline(
            _hybrid_retriever, _reranker, _generator, _memory, _vector_store,
        )

    return {
        "embedder": _embedder,
        "vector_store": _vector_store,
        "bm25_index": _bm25_index,
        "indexing": _indexing,
        "query_pipeline": _query_pipeline,
        "memory": _memory,
        "generator": _generator,
        "reranker": _reranker,
        "hybrid_retriever": _hybrid_retriever,
    }


# ── API Models ───────────────────────────────────────────────

class IndexRequest(BaseModel):
    directory: str = Field(default="./docs", description="文档目录路径")
    use_parent_child: bool = Field(default=True, description="是否使用父子分块")
    recursive: bool = Field(default=True, description="是否递归子目录")


class AskRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    session_id: str = Field(default="default", description="会话 ID")
    enable_rewrite: bool = Field(default=True)
    enable_hybrid: bool = Field(default=True)
    enable_rerank: bool = Field(default=True)
    enable_memory: bool = Field(default=True)
    enable_parent_child: bool = Field(default=True)


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    rewritten_queries: list[str]
    elapsed_ms: float
    retrieval_count: int


class StatsResponse(BaseModel):
    vector_count: int
    bm25_count: int
    embedding_model: str
    embedding_dim: int


# ── Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载组件，关闭时清理资源。"""
    _get_components()  # 触发初始化
    yield
    # 关闭 HTTP 客户端
    comps = _get_components()
    await comps["generator"].close()


app = FastAPI(title="RAG Knowledge Base API", version="1.0.0", lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────

@app.post("/index", response_model=dict[str, Any])
async def index_documents(req: IndexRequest) -> dict[str, Any]:
    """索引文档目录。

    POST /index
    {"directory": "./docs", "use_parent_child": true, "recursive": true}
    """
    comps = _get_components()
    pipeline = comps["indexing"]

    docs = pipeline.index_directory(
        Path(req.directory),
        use_parent_child=req.use_parent_child,
        recursive=req.recursive,
    )

    # 自动保存
    data_dir = config.data_dir
    pipeline.save(data_dir / "index")

    return {
        "status": "ok",
        "indexed_files": len(docs),
        "total_vectors": comps["vector_store"].count,
        "total_bm25_docs": comps["bm25_index"].count,
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest) -> AskResponse:
    """RAG 问答。

    POST /ask
    {"query": "Python中如何实现线程池？", "session_id": "demo"}
    """
    comps = _get_components()
    qp = comps["query_pipeline"]

    result = await qp.ask(
        query=req.query,
        session_id=req.session_id,
        enable_rewrite=req.enable_rewrite,
        enable_hybrid=req.enable_hybrid,
        enable_rerank=req.enable_rerank,
        enable_memory=req.enable_memory,
        enable_parent_child=req.enable_parent_child,
    )

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        rewritten_queries=result.get("rewritten_queries", []),
        elapsed_ms=result["elapsed_ms"],
        retrieval_count=len(result["retrieval_result"].results),
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """获取索引统计。"""
    comps = _get_components()
    return StatsResponse(
        vector_count=comps["vector_store"].count,
        bm25_count=comps["bm25_index"].count,
        embedding_model=config.embedding_model_name,
        embedding_dim=config.embedding_dim,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Entry ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
