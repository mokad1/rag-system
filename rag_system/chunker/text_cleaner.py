"""文本清洗器。

对文档原始文本做规范化处理：
- 统一换行符
- 压缩多余空白
- 移除控制字符
- 保留页码标记
"""

from __future__ import annotations

import re


class TextCleaner:
    """文本清洗工具。

    提供静态方法对文档文本做规范化处理。
    """

    # 保留的页码标记模式：[P数字]
    _PAGE_MARKER = re.compile(r"\[P\d+\]")

    @staticmethod
    def clean(text: str) -> str:
        """清洗文本。

        步骤：
        1. 统一换行符为 \n
        2. 移除不可见控制字符（保留换行和页码标记）
        3. 压缩连续空行为单个空行
        4. 去除首尾空白

        Args:
            text: 原始文本。

        Returns:
            清洗后的文本。
        """
        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 移除不可见控制字符（保留换行、制表符）
        # \x00-\x08, \x0b, \x0c, \x0e-\x1f 是控制字符
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        # 压缩连续空行（3+ → 2）
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 压缩连续空格为单个空格（但保留换行结构）
        lines = text.split("\n")
        cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def extract_pages(text: str) -> dict[int, str]:
        """按页码标记拆分文本。

        Args:
            text: 含 [P数字] 标记的文本。

        Returns:
            {page_number: page_text} 映射。
        """
        pages: dict[int, str] = {}
        # 按页码标记分割
        parts = re.split(r"\[P(\d+)\]\n?", text)
        # parts[0] 是第一个页码标记之前的内容，parts[1] 是第一个页码数字...
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            page_text = parts[i + 1] if i + 1 < len(parts) else ""
            pages[page_num] = page_text.strip()
        return pages
