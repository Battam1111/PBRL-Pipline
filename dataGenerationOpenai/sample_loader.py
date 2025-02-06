#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sample_loader.py

本模块用于加载和处理某个环境下的样本数据，主要功能包括：
  1. 从指定 meta 文件目录加载所有 JSON 格式的 meta 信息，构建样本数据字典
  2. 根据预定义正则表达式匹配环境根目录下的图像文件，并将对应视角图像关联到样本中
  3. 筛选出至少包含一张视角图像的有效样本，保证后续处理的数据有效
"""

import os
import json
import re
from typing import Dict
from config import FILENAME_REGEX

class SampleLoader:
    def __init__(self, env_name: str, env_root: str, meta_dir: str):
        """
        初始化 SampleLoader 实例
        :param env_name: 环境名称（例如 "metaworld_soccer-v2"）
        :param env_root: 环境根目录（包含图像文件）
        :param meta_dir: 存放 meta_xxx.json 文件的目录
        """
        self.env_name = env_name
        self.env_root = env_root
        self.meta_dir = meta_dir

    def load_all_samples(self) -> Dict[int, dict]:
        """
        从 meta 目录中加载所有 meta 文件，构建样本数据字典
        :return: 样本字典，键为 sample_id，值为样本信息字典
        """
        samples_dict = {}
        if not os.path.isdir(self.meta_dir):
            print(f"[SampleLoader] meta目录不存在：{self.meta_dir}")
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
                sid = meta_obj.get("sample_id")
                if sid is None:
                    continue
                # 构造样本字典，同时预留 views 字段用于存储视角图像文件名
                samples_dict[sid] = {
                    "sample_id": sid,
                    "reward": meta_obj.get("reward", 0.0),
                    "bin_id": meta_obj.get("bin_id", -1),
                    "embedding_dim": meta_obj.get("embedding_dim", 0),
                    "embedding": meta_obj.get("embedding", []),
                    "short_hash": meta_obj.get("short_hash", "nohash"),
                    "meta_path": meta_path,
                    "views": []  # 后续填充对应视角图像文件
                }
            except Exception as e:
                print(f"[SampleLoader] 解析文件 {meta_path} 失败：{e}")
                continue
        return samples_dict

    def populate_views(self, samples_dict: Dict[int, dict]) -> None:
        """
        遍历环境根目录下的所有 .jpg 文件，通过正则表达式匹配样本ID，
        将匹配到的视角图像文件名添加到对应样本的 'views' 列表中
        :param samples_dict: load_all_samples() 得到的样本字典
        """
        if not os.path.isdir(self.env_root):
            print(f"[SampleLoader] 环境目录不存在：{self.env_root}")
            return

        for fname in os.listdir(self.env_root):
            if not fname.lower().endswith(".jpg"):
                continue
            match = re.match(FILENAME_REGEX, fname)
            if match:
                sid_str = match.group(1)
                try:
                    sid = int(sid_str)
                except ValueError:
                    continue
                if sid in samples_dict:
                    samples_dict[sid]["views"].append(fname)

    def filter_no_view_samples(self, samples_dict: Dict[int, dict]) -> Dict[int, dict]:
        """
        筛选出至少包含一张视角图像的样本，过滤掉无效数据
        :param samples_dict: 原始样本字典
        :return: 仅包含有效视角图像的样本字典
        """
        filtered = {sid: info for sid, info in samples_dict.items() if info.get("views")}
        return filtered

    def load_samples(self) -> Dict[int, dict]:
        """
        综合调用加载 meta 文件、填充视角图像并过滤无效样本
        :return: 最终的有效样本字典
        """
        samples = self.load_all_samples()
        self.populate_views(samples)
        samples = self.filter_no_view_samples(samples)
        print(f"[SampleLoader] 加载完成，共获得 {len(samples)} 个有效样本。")
        return samples
