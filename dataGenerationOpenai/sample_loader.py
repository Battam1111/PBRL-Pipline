#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sample_loader.py

本模块用于加载指定环境下的样本数据：
  1. 从 meta 目录加载 JSON 格式的样本信息，构建样本字典；
  2. 根据正则表达式匹配环境根目录下的图像文件，将各视角图像文件与对应样本关联；
  3. 筛选出至少包含一张视角图像的有效样本。

详细错误处理和日志输出保证数据加载过程鲁棒。
"""

import os
import json
import re
from typing import Dict
from config import FILENAME_REGEX
from utils import log

class SampleLoader:
    def __init__(self, env_name: str, env_root: str, meta_dir: str):
        """
        初始化 SampleLoader 实例
        :param env_name: 环境名称（如 "metaworld_soccer-v2"）
        :param env_root: 环境根目录（包含图像文件）
        :param meta_dir: 存放 meta JSON 文件的目录
        """
        self.env_name = env_name
        self.env_root = env_root
        self.meta_dir = meta_dir

    def load_all_samples(self) -> Dict[int, dict]:
        """
        加载 meta 目录中所有 JSON 文件，构造样本数据字典。
        为避免 sample_id 重复，若发现冲突则使用 “sample_id + 文件名哈希” 作为唯一标识。
        """
        samples_dict = {}
        if not os.path.isdir(self.meta_dir):
            log(f"[SampleLoader] meta 目录不存在：{self.meta_dir}")
            return samples_dict

        for fname in os.listdir(self.meta_dir):
            if not (fname.lower().endswith(".json") and fname.startswith("meta_")):
                continue
            meta_path = os.path.join(self.meta_dir, fname)
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_obj = json.load(f)
                sid_raw = meta_obj.get("sample_id")
                if sid_raw is None:
                    continue
                sid = int(sid_raw)
                if sid in samples_dict:
                    sid = int(f"{sid}{abs(hash(fname)) % 10000}")
                samples_dict[sid] = {
                    "sample_id": sid,
                    "reward": meta_obj.get("reward", 0.0),
                    "bin_id": meta_obj.get("bin_id", -1),
                    "embedding_dim": meta_obj.get("embedding_dim", 0),
                    "embedding": meta_obj.get("embedding", []),
                    "short_hash": meta_obj.get("short_hash", "nohash"),
                    "meta_path": meta_path,
                    "views": []
                }
            except Exception as e:
                log(f"[SampleLoader] 解析文件 {meta_path} 失败：{e}")
                continue
        return samples_dict

    def populate_views(self, samples_dict: Dict[int, dict]) -> None:
        """
        遍历环境根目录下所有 .jpg 文件，根据正则表达式匹配样本 ID，
        将匹配的视角图像文件名添加到对应样本的 views 列表中。
        """
        if not os.path.isdir(self.env_root):
            log(f"[SampleLoader] 环境目录不存在：{self.env_root}")
            return

        pattern = re.compile(FILENAME_REGEX)
        for fname in os.listdir(self.env_root):
            if not fname.lower().endswith(".jpg"):
                continue
            match = pattern.match(fname)
            if match:
                try:
                    sid = int(match.group(1))
                except ValueError:
                    continue
                if sid in samples_dict:
                    samples_dict[sid]["views"].append(fname)

        # 对于没有匹配到视角图的样本，输出警告日志
        for sid, info in samples_dict.items():
            if not info.get("views"):
                log(f"[SampleLoader] 警告：样本 {sid} 没有匹配到任何视角图像文件。")

    def filter_no_view_samples(self, samples_dict: Dict[int, dict]) -> Dict[int, dict]:
        """
        筛选出至少包含一张视角图像的样本
        """
        return {sid: info for sid, info in samples_dict.items() if info.get("views")}

    def load_samples(self) -> Dict[int, dict]:
        """
        综合加载 meta 文件、填充视角图像，并过滤无效样本，返回有效样本字典。
        """
        samples = self.load_all_samples()
        self.populate_views(samples)
        samples = self.filter_no_view_samples(samples)
        log(f"[SampleLoader] 完成加载，共获得 {len(samples)} 个有效样本。")
        return samples
