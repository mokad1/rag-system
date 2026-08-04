"""全局配置管理。

加载 .env 环境变量，提供所有模块使用的配置常量。
Embedding/Reranker 模型路径从环境变量读取，支持离线部署。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()


class Config(BaseModel):
    """全局配置单例。"""

    # ── LLM (OpenAI-compatible API) ─────────────────────
    llm_api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    )
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat"))

    # ── Embedding 模型 (bge-small-zh) ───────────────────
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )
    )
    embedding_dim: int = 384  # bge-small-zh 输出维度
    embedding_device: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu")
    )

    # ── Reranker 模型 (bge-reranker-base) ───────────────
    reranker_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-base"
        )
    )
    reranker_device: str = Field(
        default_factory=lambda: os.getenv("RERANKER_DEVICE", "cpu")
    )

    # ── 分块参数 ────────────────────────────────────────
    parent_chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("PARENT_CHUNK_SIZE", "800"))
    )
    child_chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHILD_CHUNK_SIZE", "300"))
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    fixed_chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("FIXED_CHUNK_SIZE", "500"))
    )

    # ── 检索参数 ────────────────────────────────────────
    hybrid_top_k: int = Field(
        default_factory=lambda: int(os.getenv("HYBRID_TOP_K", "20"))
    )
    final_top_k: int = Field(
        default_factory=lambda: int(os.getenv("FINAL_TOP_K", "5"))
    )
    rrf_k: int = 60  # RRF 融合参数

    # ── 记忆参数 ────────────────────────────────────────
    short_term_window: int = Field(
        default_factory=lambda: int(os.getenv("SHORT_TERM_WINDOW", "3"))
    )
    max_memory_turns: int = Field(
        default_factory=lambda: int(os.getenv("MAX_MEMORY_TURNS", "6"))
    )

    # ── 日志 ────────────────────────────────────────────
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ── 数据目录 ────────────────────────────────────────
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "./data"))
    )


config = Config()


def get_config() -> Config:
    """获取全局配置单例。"""
    return config
