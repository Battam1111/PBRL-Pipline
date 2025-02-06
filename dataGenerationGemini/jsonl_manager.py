#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jsonl_manager.py

本模块负责根据样本对和样本数据生成大 JSONL 文件，用于后续调用 Gemini API。
主要功能：
  1. 根据预定义对话模板构造任务请求 JSONL，每一行包含 custom_id、请求方法、URL 以及请求体（包含 prompt 信息）。
  2. 将大 JSONL 文件按指定行数分块，便于批量处理。
  3. 将多个处理结果文件合并为最终输出文件，并根据 custom_id 排序。
  
注：构造任务请求时，将原 OpenAI 的 "messages" 格式转换为 Gemini 的 "contents" 格式，
     其中 "contents" 数组中每个元素包含 "role" 与 "parts" 字段。
"""

import os
import re
import json
import time
from typing import List, Dict
from image_stitcher import stitch_images
from config import SYSTEM_PROMPT, GEMINI_MODEL, single_round_template, GEMINI_API_URL_TEMPLATE, GEMINI_API_KEY

class JSONLManager:
    def __init__(self, env_name: str, objective: str, samples_dict: Dict[int, dict], hf_uploader):
        """
        初始化 JSONLManager 实例。

        参数：
          env_name: 环境名称。
          objective: 当前环境的任务目标描述。
          samples_dict: 样本数据字典，由 SampleLoader 加载。
          hf_uploader: HuggingFaceUploader 实例，用于图像上传。
        """
        self.env_name = env_name
        self.objective = objective
        self.samples_dict = samples_dict
        self.hf_uploader = hf_uploader
        self.sid2urls: Dict[int, str] = {}

    def _gather_all_views_for_sid(self, sid: int) -> str:
        """
        对给定样本 ID，将所有视角图像拼接后上传，并返回拼接图像的远程访问 URL。

        参数：
          sid: 样本 ID。
        返回：
          拼接后图像的远程访问 URL（若无视角图像，则返回空字符串）。
        """
        if sid in self.sid2urls:
            return self.sid2urls[sid]

        info = self.samples_dict.get(sid)
        if not info:
            self.sid2urls[sid] = ""
            return ""
        views = info.get("views", [])
        if not views:
            self.sid2urls[sid] = ""
            return ""
        meta_dir = os.path.dirname(info.get("meta_path", ""))
        image_dir = os.path.dirname(meta_dir)
        local_paths = []
        for view in views:
            local_path = os.path.join(image_dir, view)
            local_paths.append(local_path)
        if len(local_paths) == 1:
            final_image_path = local_paths[0]
        else:
            stitched_dir = os.path.join(image_dir, "stitched")
            os.makedirs(stitched_dir, exist_ok=True)
            first_image_name = os.path.basename(local_paths[0])
            # 去除文件名中的 "_view" 部分
            stitched_name = re.sub(r'_view\w+', '', first_image_name)
            final_image_path = os.path.join(stitched_dir, stitched_name)
            try:
                stitch_images(local_paths, final_image_path, layout="grid")
            except Exception as e:
                print(f"[JSONLManager] 拼接图像失败，使用原始图像。错误：{e}")
                final_image_path = local_paths[0]
        hf_url = self.hf_uploader.upload_image(self.env_name, final_image_path)
        self.sid2urls[sid] = hf_url
        return hf_url

    def create_big_jsonl(self, pairs: List[tuple], out_file: str) -> None:
        """
        根据样本对列表生成大 JSONL 文件，每一行构造一个任务请求。

        参数：
          pairs: 样本对列表，每个元素为 (sidA, sidB, 标签)。
          out_file: 输出 JSONL 文件路径。
        """
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as outf:
            for idx, (sidA, sidB, tag_info) in enumerate(pairs, start=1):
                # 获取两个样本的图像（或拼接后图像）的远程 URL
                situation1_url = self._gather_all_views_for_sid(sidA)
                situation2_url = self._gather_all_views_for_sid(sidB)
                # 根据预定义模板构造用户输入
                user_content = single_round_template.format(
                    situation1_urls=situation1_url,
                    situation2_urls=situation2_url,
                    objective=self.objective
                )
                custom_id = f"{self.env_name}-{idx}"
                # 构造 Gemini 请求体，采用 "contents" 数组格式
                body = {
                    "model": GEMINI_MODEL,
                    "contents": [
                        {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
                        {"role": "user", "parts": [{"text": user_content}]}
                    ],
                    "maxOutputTokens": 2000
                }
                data = {
                    "custom_id": custom_id,
                    "method": "POST",
                    # 使用官方 Gemini API 端点，填入 GEMINI_API_KEY（此处可直接填入配置中的密钥）
                    "url": GEMINI_API_URL_TEMPLATE.format(model=GEMINI_MODEL, api_key=GEMINI_API_KEY),
                    "body": body
                }
                outf.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"[JSONLManager] 已生成大 JSONL 文件：{out_file}，任务数 = {len(pairs)}")

    def chunk_file(self, file_path: str, chunk_size: int) -> List[str]:
        """
        将大 JSONL 文件按每 chunk_size 行分块。

        参数：
          file_path: 大 JSONL 文件路径。
          chunk_size: 每个子文件最大行数。
        返回：
          分块后生成的子文件路径列表。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        if total <= chunk_size:
            return [file_path]
        base_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_files = []
        idx = 0
        for start in range(0, total, chunk_size):
            chunk_lines = lines[start: start + chunk_size]
            chunk_file_path = os.path.join(base_dir, f"{base_name}_chunk{idx}.jsonl")
            with open(chunk_file_path, "w", encoding="utf-8") as cf:
                cf.writelines(chunk_lines)
            out_files.append(chunk_file_path)
            idx += 1
        print(f"[JSONLManager] 分块后生成 {len(out_files)} 个子文件。")
        return out_files

    def merge_outputs(self, chunk_out_files: List[str], final_out: str) -> None:
        """
        合并多个子结果文件为最终输出文件，并按 custom_id 排序后保存。

        参数：
          chunk_out_files: 子结果文件路径列表。
          final_out: 最终合并后的输出文件路径。
        """
        merged = []
        for cf in chunk_out_files:
            if not os.path.isfile(cf):
                continue
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        merged.append(json.loads(line))
                    except Exception as e:
                        print(f"[JSONLManager] 解析行失败：{e}")
                        continue
        def parse_idx(cid: str) -> int:
            try:
                return int(cid.rsplit("-", 1)[-1])
            except Exception:
                return 999999
        merged.sort(key=lambda x: parse_idx(x.get("custom_id", "")))
        os.makedirs(os.path.dirname(final_out), exist_ok=True)
        with open(final_out, "w", encoding="utf-8") as outf:
            for item in merged:
                outf.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[JSONLManager] 最终合并文件生成：{final_out}，总行数 = {len(merged)}")
