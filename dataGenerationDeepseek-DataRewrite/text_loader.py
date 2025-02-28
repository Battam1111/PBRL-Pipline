#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
text_loader.py

本模块用于加载输入文本样本。
假设输入文件中每一行均为一个文本描述。
"""

import os
from utils import log

class TextLoader:
    def __init__(self, input_file: str):
        self.input_file = input_file

    def load_texts(self):
        """
        读取输入文件中所有文本，每行作为一个样本，返回文本列表。
        """
        if not os.path.isfile(self.input_file):
            log(f"[TextLoader] 输入文件不存在：{self.input_file}")
            return []
        texts = []
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
            log(f"[TextLoader] 成功加载 {len(texts)} 个文本样本。")
        except Exception as e:
            log(f"[TextLoader] 读取文件失败：{e}")
        return texts
