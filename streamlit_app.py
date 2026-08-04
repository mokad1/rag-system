"""Streamlit 可视化演示界面。

功能：
- 文档索引管理（上传/索引/统计）
- RAG 问答（支持消融开关对比）
- 检索过程可视化（BM25/向量/Reranker 结果对比）
- 对话记忆展示
- 答案溯源引用高亮
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="📚",
    layout="wide",
)

st.title("📚 进阶全链路 RAG 知识库问答系统")
st.caption("Parent-Child Chunking | Hybrid Retrieval | bge-reranker | Query Rewrite | Layered Memory")

# ── 初始化 ──────────────────────────────────────────────────

setup_logging("WARNING")


@st.cache_resource
def get_components() -> dict:
    """初始化所有组件（缓存避免重复加载模型）。"""
    embedder = BGEEmbedder()
    vector_store = VectorStore()
    bm25_index = BM25Index()
    indexing = IndexingPipeline(embedder, vector_store, bm25_index)
    hybrid_retriever = HybridRetriever(vector_store, bm25_index, embedder)
    reranker = Reranker()
    generator = LLMGenerator()
    memory = LayeredMemory()
    query_pipeline = QueryPipeline(hybrid_retriever, reranker, generator, memory, vector_store)

    # 尝试加载已有索引
    data_dir = config.data_dir
    idx_path = data_dir / "index"
    if (idx_path / "faiss" / "faiss.index").exists():
        try:
            new_vs = VectorStore.load(idx_path / "faiss")
            new_bm25 = BM25Index.load(idx_path / "bm25")
            vector_store.index = new_vs.index
            vector_store.chunk_map = new_vs.chunk_map
            vector_store.chunk_id_to_faiss_id = new_vs.chunk_id_to_faiss_id
            vector_store._next_id = new_vs._next_id
            bm25_index._chunks = new_bm25._chunks
            bm25_index._tokenized_corpus = new_bm25._tokenized_corpus
            bm25_index._bm25 = new_bm25._bm25
            st.sidebar.success(f"✅ Loaded index: {vector_store.count} vectors")
        except Exception as e:
            st.sidebar.warning(f"Failed to load index: {e}")

    return {
        "embedder": embedder,
        "vector_store": vector_store,
        "bm25_index": bm25_index,
        "indexing": indexing,
        "query_pipeline": query_pipeline,
        "memory": memory,
        "generator": generator,
    }


comps = get_components()

# ── 侧边栏 ──────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    st.metric("Vectors", comps["vector_store"].count)
    st.metric("BM25 Docs", comps["bm25_index"].count)
    st.metric("Embedding", config.embedding_model_name.split("/")[-1])
    st.metric("LLM", config.llm_model)

    st.divider()

    st.header("📁 Document Indexing")
    index_dir = st.text_input("Document directory", "./docs")
    use_pc = st.checkbox("Parent-Child Chunking", value=True)
    if st.button("🔨 Index Documents", use_container_width=True):
        with st.spinner("Indexing..."):
            try:
                docs = comps["indexing"].index_directory(
                    Path(index_dir), use_parent_child=use_pc,
                )
                comps["indexing"].save(config.data_dir / "index")
                st.success(f"Indexed {len(docs)} files!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()

    st.header("🔬 Ablation Controls")
    enable_rewrite = st.checkbox("Query Rewrite", value=True)
    enable_hybrid = st.checkbox("Hybrid Retrieval", value=True)
    enable_rerank = st.checkbox("Reranker", value=True)
    enable_memory = st.checkbox("Layered Memory", value=True)
    enable_pc = st.checkbox("Parent-Child Resolution", value=True)

    st.divider()
    st.header("💬 Session")
    session_id = st.text_input("Session ID", "demo")

    if st.button("🧹 Clear Memory", use_container_width=True):
        comps["memory"].clear(session_id)
        st.success("Memory cleared!")

# ── 主区域 ──────────────────────────────────────────────────

tab1, tab2 = st.tabs(["💬 Q&A", "🔍 Retrieval Debug"])

with tab1:
    query = st.text_area(
        "Enter your question",
        placeholder="e.g., Python中如何实现线程池？",
        height=100,
    )

    if st.button("🔍 Ask", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question.")
        elif comps["vector_store"].count == 0:
            st.warning("Please index documents first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    result = comps["query_pipeline"].ask_sync(
                        query=query.strip(),
                        session_id=session_id,
                        enable_rewrite=enable_rewrite,
                        enable_hybrid=enable_hybrid,
                        enable_rerank=enable_rerank,
                        enable_memory=enable_memory,
                        enable_parent_child=enable_pc,
                    )

                    # 答案展示
                    st.markdown("### 📝 Answer")
                    st.markdown(result["answer"])

                    # 耗时和查询改写
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Response Time", f"{result['elapsed_ms']:.0f}ms")
                    with col2:
                        st.metric("Retrieved Chunks", len(result["retrieval_result"].results))

                    if result.get("rewritten_queries"):
                        with st.expander("🔄 Rewritten Queries"):
                            for q in result["rewritten_queries"]:
                                st.text(f"• {q}")

                    # 溯源引用
                    citations = result.get("citations", [])
                    if citations:
                        with st.expander(f"📎 Sources ({len(citations)} references)"):
                            for i, c in enumerate(citations):
                                st.markdown(f"**[{i+1}]** {c['source_doc']} — Page {c['page']} (score: {c['score']:.3f})")
                                st.text(c.get("text", "")[:200] + "...")
                                st.divider()

                except Exception as e:
                    st.error(f"Error: {e}")

    # 对话历史
    with st.expander("💾 Conversation History"):
        state = comps["memory"].get_or_create(session_id)
        if state.long_term_summary:
            st.info(f"**Summary**: {state.long_term_summary}")
        for msg in state.messages[-12:]:
            role_label = "🧑 User" if msg.role == "user" else "🤖 Assistant"
            st.caption(role_label)
            st.text(msg.content[:300] + ("..." if len(msg.content) > 300 else ""))


with tab2:
    st.header("🔍 Retrieval Debug")

    debug_query = st.text_input("Debug query", placeholder="Enter a query to see retrieval details...")

    if st.button("Debug Retrieve", use_container_width=True) and debug_query:
        comps2 = get_components()
        hr = comps2["hybrid_retriever"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vector Retrieval")
            vr = hr.search(debug_query, top_k=10, use_hybrid=False)
            for i, sr in enumerate(vr.results):
                st.text(f"#{i+1} [score={sr.score:.3f}] {sr.chunk.content[:100]}...")

        with col2:
            st.subheader("BM25 Retrieval")
            bm25 = hr.bm25_index.search(debug_query, top_k=10)
            for i, (chunk, score) in enumerate(bm25):
                st.text(f"#{i+1} [score={score:.3f}] {chunk.content[:100]}...")

        st.subheader("Hybrid (RRF Fusion)")
        hybrid_r = hr.search(debug_query, top_k=10, use_hybrid=True)
        for i, sr in enumerate(hybrid_r.results):
            st.text(f"#{i+1} [score={sr.score:.3f}][{sr.source}] {sr.chunk.content[:100]}...")

st.divider()
st.caption("进阶全链路 RAG 知识库问答系统 • Parent-Child + Hybrid + Reranker + Query Rewrite + Layered Memory")
