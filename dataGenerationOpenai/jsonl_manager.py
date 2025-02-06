#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jsonl_manager.py

本模块负责根据样本对和样本数据生成大 JSONL 文件，用于后续调用 OpenAI API。
主要功能包括：
  1. 根据预定义对话模板构造任务请求 JSONL，每一行包含 custom_id、请求方法、URL 以及请求体（含 prompt 和 max_tokens 限制）。
  2. 将大 JSONL 文件按指定行数分块，便于批量处理。
  3. 将多个处理结果文件合并为最终输出文件，并按 custom_id 排序。

特别说明：
  - 构造每个任务请求时，会将每个样本的所有视角图像拼接成一张图像后上传至 Hugging Face，
    并将拼接后的图像 URL 传递给模型，以避免因上传过多图像而导致大模型产生幻觉。
  - 由于 meta 文件位于环境子目录中，而实际图像保存在环境根目录下，因此需要适当调整路径。
  - 拼接后的图像文件命名不再采用简略形式，而是从原始视角图像文件名中提取所有信息，仅去掉“view”相关部分，
    以便保留样本的全部关键信息。

注：所有模块依赖均在文件头部统一导入，保证代码结构统一、清晰。
"""

import os
import re
import json
import time
from typing import List, Dict
# 统一导入 image_stitcher 模块，用于图像拼接操作
from image_stitcher import stitch_images
from config import SYSTEM_PROMPT, MODEL, single_round_template


class JSONLManager:
    def __init__(self, env_name: str, objective: str, samples_dict: Dict[int, dict], hf_uploader):
        """
        初始化 JSONLManager 实例

        :param env_name: 环境名称（如 "metaworld_soccer-v2"）
        :param objective: 当前环境的任务目标描述
        :param samples_dict: 样本数据字典，由 SampleLoader 加载的有效样本数据
        :param hf_uploader: HuggingFaceUploader 实例，用于图像上传
        """
        self.env_name = env_name
        self.objective = objective
        self.samples_dict = samples_dict
        self.hf_uploader = hf_uploader
        # 内部缓存：记录每个样本所有视角图像拼接后的 URL，避免重复上传
        self.sid2urls: Dict[int, str] = {}

    def _gather_all_views_for_sid(self, sid: int) -> str:
        """
        根据给定样本 ID，将其所有视角图像拼接成一张图像后上传，并返回拼接后图像的远程 URL。
        
        具体流程：
          1. 根据样本信息从 meta 文件路径中确定环境根目录，然后构造所有视角图像的完整本地路径列表。
          2. 如果该样本仅有一张视角图像，则直接使用该图像，无需拼接。
          3. 如果存在多张视角图像，则在环境根目录下创建 "stitched" 子目录，
             根据第一个视角图像的文件名构造拼接后图像的文件名：去掉文件名中包含“view”部分，
             保留原有其它信息（例如 "pc_1_bin_2_r_1.0_t_2.0_emb_abc_view1.jpg" 经过处理后变为 "pc_1_bin_2_r_1.0_t_2.0_emb_abc.jpg"）。
          4. 调用 image_stitcher 模块的 stitch_images() 函数，采用网格布局将所有视角图像拼接成一张图像，
             并保存至上述构造的目标路径。
          5. 将拼接后（或直接使用单张视角图像）的文件通过 hf_uploader.upload_image() 上传至 Hugging Face，
             获取并返回对应的远程访问 URL。

        :param sid: 样本 ID
        :return: 拼接后图像的远程访问 URL；若该样本无视角图像，则返回空字符串。
        """
        # 若已处理过该样本，则直接返回缓存中的 URL
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

        # 获取 meta 文件所在目录，例如 "data/renderPointCloud/metaworld_soccer-v2/meta"
        meta_dir = os.path.dirname(info.get("meta_path", ""))
        # 环境根目录为 meta 目录的上级，例如 "data/renderPointCloud/metaworld_soccer-v2"
        image_dir = os.path.dirname(meta_dir)
        # 构造所有视角图像的完整本地路径列表
        local_paths = []
        for view in views:
            local_path = os.path.join(image_dir, view)
            local_paths.append(local_path)

        # 如果只有一张视角图像，则直接使用该图像
        if len(local_paths) == 1:
            final_image_path = local_paths[0]
        else:
            # 在环境根目录下创建用于存放拼接后图像的 "stitched" 目录
            stitched_dir = os.path.join(image_dir, "stitched")
            os.makedirs(stitched_dir, exist_ok=True)
            # 取第一张图像的文件名，用于构造拼接后图像的名称
            first_image_name = os.path.basename(local_paths[0])
            # 采用正则表达式将文件名中的 “_view…” 部分去除，例如：
            # "pc_1_bin_2_r_1.0_t_2.0_emb_abc_view1.jpg" -> "pc_1_bin_2_r_1.0_t_2.0_emb_abc.jpg"
            stitched_name = re.sub(r'_view\w+', '', first_image_name)
            # 若存在多个视角，可以在文件名上附加标识，例如添加 "_stitched" 后缀（可选）
            # stitched_name = re.sub(r'_view\w+', '_stitched', first_image_name)
            final_image_path = os.path.join(stitched_dir, stitched_name)
            # 调用 image_stitcher 模块进行图像拼接，采用网格布局
            try:
                stitch_images(local_paths, final_image_path, layout="grid")
            except Exception as e:
                print(f"[JSONLManager] 拼接图像失败，使用原始图像。错误：{e}")
                # 若拼接失败，则退回使用第一张视角图像
                final_image_path = local_paths[0]

        # 上传最终图像，并获取远程访问 URL
        hf_url = self.hf_uploader.upload_image(self.env_name, final_image_path)
        self.sid2urls[sid] = hf_url
        return hf_url

    def create_big_jsonl(self, pairs: List[tuple], out_file: str) -> None:
        """
        根据生成的样本对列表构造大 JSONL 文件，每一行构成一个任务请求，
        请求中包含自定义任务 ID、请求方法、URL 以及请求体（含 prompt、模型、max_tokens 等参数）。

        流程说明：
          1. 对于每个样本对 (sidA, sidB, tag_info)，调用 _gather_all_views_for_sid() 分别获取 Situation 1 和 Situation 2 的图像 URL；
          2. 根据预定义的对话模板 single_round_template 填充对话内容，其中包含图像 URL 和任务目标；
          3. 构造包含 custom_id、请求方法、URL 和请求体的 JSON 对象，并按行写入输出文件。

        :param pairs: 样本对列表，每个元素为 (sidA, sidB, 标签)
        :param out_file: 输出 JSONL 文件路径
        """
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as outf:
            for idx, (sidA, sidB, tag_info) in enumerate(pairs, start=1):
                # 分别获取两个样本的图像（或拼接后图像）的远程 URL
                situation1_url = self._gather_all_views_for_sid(sidA)
                situation2_url = self._gather_all_views_for_sid(sidB)
                # 根据预定义模板构造用户输入内容
                user_content = single_round_template.format(
                    situation1_urls=situation1_url,
                    situation2_urls=situation2_url,
                    objective=self.objective
                )
                custom_id = f"{self.env_name}-{idx}"
                data = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content}
                        ],
                        "max_tokens": 2000
                    }
                }
                outf.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"[JSONLManager] 已生成大 JSONL 文件：{out_file}，任务数 = {len(pairs)}")

    def chunk_file(self, file_path: str, chunk_size: int) -> List[str]:
        """
        将大 JSONL 文件按每 chunk_size 行分块，生成多个子文件，以便后续批量处理。

        :param file_path: 大 JSONL 文件路径
        :param chunk_size: 每个子文件的最大行数
        :return: 分块后生成的子文件路径列表
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
        将多个处理结果文件合并为最终输出文件，并根据 custom_id 中的数字部分排序后输出。

        :param chunk_out_files: 子结果文件路径列表
        :param final_out: 最终合并后的输出文件路径
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

        # 定义辅助函数：从 custom_id 中提取末尾数字用于排序
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
