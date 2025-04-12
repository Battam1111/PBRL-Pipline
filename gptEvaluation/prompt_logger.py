#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prompt_logger.py

本模块提供 PromptLogger 类，用于记录每次发送给 GPT 的完整 Prompt 信息，
并将记录以 JSON 格式保存到本地文件。保存的文件名包含源数据文件名（或 demo 标识）以及当前时间戳，
便于后续调试和追溯。
"""

import os
import json
import datetime
import logging

logger = logging.getLogger(__name__)

class PromptLogger:
    """
    PromptLogger 用于记录并保存发送给 GPT 的提示信息（Prompt）日志。
    每条日志记录包含：question_id、prompt 内容以及记录的时间戳。
    """
    def __init__(self, source_file: str):
        """
        初始化 PromptLogger 实例。

        参数:
            source_file (str): 被评估数据的文件名，用于日志文件命名（若为 demo，则传入 "demo"）。
        """
        self.source_file = source_file
        self.logs = []  # 日志列表，每个元素为包含 question_id, prompt, timestamp 的字典
    
    def log(self, question_id: str, prompt: str):
        """
        记录一次发送给 GPT 的提示信息。

        参数:
            question_id (str): 当前评估实例的唯一标识符。
            prompt (str): 发送给 GPT 的完整提示信息。
        """
        entry = {
            "question_id": question_id,
            "prompt": prompt,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.logs.append(entry)
    
    def save_logs(self, output_dir: str = "gptEvaluation/logs/gpt_inputs"):
        """
        将记录的日志保存到本地 JSON 文件中。
        文件名格式：gpt_input_<source_file>_<YYYYmmdd_HHMMSS>.json

        参数:
            output_dir (str): 保存日志的目录，默认为 "logs/gpt_inputs"。
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(self.source_file)[0]
        file_name = f"gpt_input_{base_name}_{timestamp}.json"
        output_path = os.path.join(output_dir, file_name)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.logs, f, ensure_ascii=False, indent=4)
            logger.info(f"GPT输入日志已保存至 {output_path}")
        except Exception as e:
            logger.error(f"保存GPT输入日志失败: {e}")
