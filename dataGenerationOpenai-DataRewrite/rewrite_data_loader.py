#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rewrite_data_loader.py

本模块用于加载重写数据集。数据集位于 DATA_REWRITE_DIR 目录，
每个 JSON 文件包含多个条目，每个条目结构如下：
{
  "id": "Safe_101619_0_2",
  "point": "Safe_101619_0_2.npy",
  "conversations": [
      {"from": "human", "value": "<point>\n..."},
      {"from": "gpt", "value": "The hinge knob attaches to the hinge door."}
  ]
}

本模块提取每个条目中所有 {"from": "gpt", "value": ...} 部分的文本作为待改写文本，
同时返回原始数据以便后续将改写结果合并回原始数据中。

【新增功能】支持根据配置参数 MAX_TEXT_ITEMS 控制处理任务条数。
"""
import os
import json
from utils import log
from config import MAX_TEXT_ITEMS

class RewriteDataLoader:
    def __init__(self, json_file: str):
        self.json_file = json_file

    def load_rewrite_tasks(self):
        """
        读取 JSON 数据文件，提取所有待改写文本。
        返回：
          tasks: 列表，每个元素为 {"custom_id": ..., "input_text": ...}
          original_data: 原始数据列表，供后续合并改写结果使用
          
        如果 MAX_TEXT_ITEMS 不为 None，则只保留前 MAX_TEXT_ITEMS 条任务。
        """
        if not os.path.isfile(self.json_file):
            log(f"[RewriteDataLoader] 文件不存在：{self.json_file}")
            return [], []
        tasks = []
        original_data = []
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, item in enumerate(data):
                original_data.append(item)
                convs = item.get("conversations", [])
                for j, conv in enumerate(convs):
                    if conv.get("from") == "gpt":
                        # custom_id 格式：{文件基本名}_{条目索引}_{对话索引}
                        base_name = os.path.splitext(os.path.basename(self.json_file))[0]
                        custom_id = f"{base_name}_{idx}_{j}"
                        input_text = conv.get("value", "").strip()
                        if input_text:
                            tasks.append({"custom_id": custom_id, "input_text": input_text})
            log(f"[RewriteDataLoader] 从 {self.json_file} 加载了 {len(tasks)} 条待改写文本。")
            if MAX_TEXT_ITEMS is not None:
                tasks = tasks[:MAX_TEXT_ITEMS]
                log(f"[RewriteDataLoader] 根据配置 MAX_TEXT_ITEMS，只保留前 {MAX_TEXT_ITEMS} 条任务。")
        except Exception as e:
            log(f"[RewriteDataLoader] 读取或解析文件失败：{e}")
        return tasks, original_data
