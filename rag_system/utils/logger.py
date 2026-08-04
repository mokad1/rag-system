"""结构化日志工具。"""

from __future__ import annotations

import logging
import sys

from rag_system.config import config


def setup_logging(level: str | None = None) -> None:
    """初始化全局日志。"""
    lvl = level or config.log_level
    root = logging.getLogger("rag_system")
    root.setLevel(getattr(logging, lvl.upper(), logging.INFO))
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    ))
    root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。"""
    return logging.getLogger(f"rag_system.{name}")
