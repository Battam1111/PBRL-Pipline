#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jsonl_manager.py

本模块负责根据生成的样本对构造多模态 JSONL 请求文件，每行一个 JSON 对象，
包含 custom_id、请求方法、URL、body（内含系统 prompt 与用户消息）。
同时提供将同一环境下的多视角图像拼接为单图的功能（若配置为拼接），或直接上传全部图片（不拼接），
并返回图像 URL（或 URL 列表）。

主要功能：
  - 构造 JSONL 文件（支持图片 URL 的动态替换，根据是否拼接分别使用不同占位符）
  - 将大文件按行分块
  - 将多个输出文件合并为一个，并按 custom_id 排序

所有操作均附有详细中文注释说明。
"""

import os
import re
import json
from typing import List, Dict, Union
from image_stitcher import stitch_images
from config import SYSTEM_PROMPT, MODEL, USER_MESSAGE_TEMPLATE, STITCH_IMAGES
from utils import log
import copy

def replace_placeholder(url_template: str, situation: str, url_list: List[str]) -> str:
    """
    辅助函数：对给定的 url_template 字符串中的占位符进行替换。
    占位符格式：例如对于 situation="situation1_url"，模板中应包含 {situation1_url1}、{situation1_url2} 等，
    替换时根据数字顺序分别替换为 url_list 中对应的 URL；若索引超出则返回空字符串。
    
    :param url_template: 待替换的模板字符串
    :param situation: 占位符前缀，例如 "situation1_url" 或 "situation2_url"
    :param url_list: 对应的 URL 列表
    :return: 替换后的字符串
    """
    pattern = r"\{" + situation + r"(\d+)\}"
    def repl(match):
        index = int(match.group(1)) - 1  # 占位符数字从1开始
        if 0 <= index < len(url_list):
            return url_list[index]
        else:
            return ""
    return re.sub(pattern, repl, url_template)

class JSONLManager:
    def __init__(self, env_name: str, objective: str, samples_dict: Dict[int, dict], hf_uploader):
        """
        初始化 JSONLManager
        
        :param env_name: 环境名称
        :param objective: 当前环境目标描述
        :param samples_dict: 样本字典（sample_id -> 信息）
        :param hf_uploader: HuggingFaceUploader 实例，用于上传图像
        """
        self.env_name = env_name
        self.objective = objective
        self.samples_dict = samples_dict
        self.hf_uploader = hf_uploader
        self.sid2url: Dict[int, Union[str, List[str]]] = {}

    def _gather_all_views_for_sid(self, sid: int) -> Union[str, List[str]]:
        """
        将指定样本的所有视角图像上传后返回 URL 信息。
        
        根据配置参数 STITCH_IMAGES 判断：
          - 若 STITCH_IMAGES 为 True：若该样本包含多于 1 张图像，则进行拼接后上传，返回拼接后图像的 URL（字符串）；否则直接返回单张图像 URL。
          - 若 STITCH_IMAGES 为 False：不进行拼接，直接对所有图像进行自然排序后逐一上传，返回一个 URL 列表。
        
        同时，如果该样本已缓存，则直接返回缓存结果。
        
        :param sid: 样本 ID
        :return: 上传后返回的图像 URL（字符串）或 URL 列表
        """
        if sid in self.sid2url:
            return self.sid2url[sid]

        info = self.samples_dict.get(sid)
        if not info:
            self.sid2url[sid] = ""
            return ""
        views = info.get("views", [])
        if not views:
            self.sid2url[sid] = ""
            return ""
        
        # 假设 meta 目录与图像目录在同一层级，此处也可根据实际路径微调
        env_root = os.path.dirname(info.get("meta_path", ""))  
        local_paths = [os.path.join(self.env_root_dir(), v) for v in views]

        if STITCH_IMAGES:
            if len(local_paths) == 1:
                final_image_path = local_paths[0]
            else:
                stitched_dir = os.path.join(self.env_root_dir(), "stitched")
                os.makedirs(stitched_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(local_paths[0]))[0]
                cleaned_name = re.sub(r'_view\w+', '', base_name)
                stitched_name = f"{cleaned_name}_sid{sid}.jpg"
                final_image_path = os.path.join(stitched_dir, stitched_name)
                try:
                    stitch_images(local_paths, final_image_path, layout="grid")
                except Exception as e:
                    log(f"[JSONLManager] 图像拼接失败：{e}，退回使用第一张图像：{local_paths[0]}")
                    final_image_path = local_paths[0]
            hf_url = self.hf_uploader.upload_image(self.env_name, final_image_path)
            self.sid2url[sid] = hf_url
            return hf_url
        else:
            # 不拼接，直接上传每张视角图像
            sorted_paths = sorted(local_paths, key=lambda p: p.lower())
            urls = []
            for path in sorted_paths:
                url = self.hf_uploader.upload_image(self.env_name, path)
                urls.append(url)
            self.sid2url[sid] = urls
            return urls

    def env_root_dir(self):
        """辅助函数：获取当前环境的根目录（含图像文件）。"""
        # 由于 config 中定义了 RENDER_ROOT_DIR，如 data/renderPointCloud/metaworld_soccer-v2
        # 也可根据项目实际情况进一步拼接
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "renderPointCloud", self.env_name)

    def create_big_jsonl(self, pairs: List[tuple], out_file: str) -> None:
        """
        根据生成的样本对构造 JSONL 请求文件，每行包含一个完整的请求 payload。
        
        :param pairs: 样本对列表，每项为 (sampleA, sampleB, strategyTag)
        :param out_file: 输出 JSONL 文件路径
        """
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
        with open(out_file, "w", encoding="utf-8") as outf:
            for idx, (sidA, sidB, tag_info) in enumerate(pairs, start=1):
                # 获取两个样本的图像信息（可能是拼接后单 URL，也可能是 URL 列表）
                situation1_val = self._gather_all_views_for_sid(sidA)
                situation2_val = self._gather_all_views_for_sid(sidB)
                
                # 构造用户消息列表（深拷贝模板）
                user_message = []
                for item in USER_MESSAGE_TEMPLATE:
                    item_copy = copy.deepcopy(item)
                    if item_copy.get("type") == "image_url" and isinstance(item_copy.get("image_url"), dict):
                        url_template = item_copy["image_url"].get("url", "")
                        if isinstance(situation1_val, str) and isinstance(situation2_val, str):
                            # 拼接模式：只需替换 {situation1_url} / {situation2_url}
                            url_template = url_template.replace("{situation1_url}", situation1_val)
                            url_template = url_template.replace("{situation2_url}", situation2_val)
                        elif isinstance(situation1_val, list) and isinstance(situation2_val, list):
                            # 非拼接模式：依次替换 {situation1_url1} / {situation1_url2} / ...
                            url_template = replace_placeholder(url_template, "situation1_url", situation1_val)
                            url_template = replace_placeholder(url_template, "situation2_url", situation2_val)
                        item_copy["image_url"]["url"] = url_template
                    elif item_copy.get("type") == "text":
                        txt_val = item_copy["text"]
                        txt_val = txt_val.replace("{objective}", self.objective)
                        item_copy["text"] = txt_val
                    user_message.append(item_copy)
                
                custom_id = f"{self.env_name}-{idx}"
                data = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": SYSTEM_PROMPT
                                    }
                                ]
                            },
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ],
                        "max_tokens": 2000
                    }
                }
                outf.write(json.dumps(data, ensure_ascii=False) + "\n")
        log(f"[JSONLManager] 已生成 JSONL 文件：{out_file}，共 {len(pairs)} 条任务。")

    def chunk_file(self, file_path: str, chunk_size: int) -> List[str]:
        """
        将大 JSONL 文件按指定行数分块，返回各子文件路径列表。
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
            chunk_lines = lines[start:start+chunk_size]
            chunk_file_path = os.path.join(base_dir, f"{base_name}_chunk{idx}.jsonl")
            with open(chunk_file_path, "w", encoding="utf-8") as cf:
                cf.writelines(chunk_lines)
            out_files.append(chunk_file_path)
            idx += 1
        log(f"[JSONLManager] 分块生成 {len(out_files)} 个子文件。")
        return out_files

    def merge_outputs(self, chunk_out_files: List[str], final_out: str) -> None:
        """
        将多个输出文件合并为一个，并根据 custom_id 排序后写入 final_out 文件。
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
                        log(f"[JSONLManager] 解析行失败：{e}")
        def parse_idx(custom_id: str) -> int:
            try:
                return int(custom_id.rsplit("-", 1)[-1])
            except:
                return 999999
        merged.sort(key=lambda x: parse_idx(x.get("custom_id", "")))
        os.makedirs(os.path.dirname(final_out), exist_ok=True)
        with open(final_out, "w", encoding="utf-8") as outf:
            for item in merged:
                outf.write(json.dumps(item, ensure_ascii=False) + "\n")
        log(f"[JSONLManager] 已合并输出文件：{final_out}，共 {len(merged)} 行。")
