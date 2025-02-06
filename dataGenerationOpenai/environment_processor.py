#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
environment_processor.py

本模块负责处理单个环境的数据生成及任务构建流程，主要包括：
  1. 加载环境样本数据（meta 文件和图像），并进行预处理（过滤无视角样本）
  2. 根据样本数据生成样本对（支持跨 bin、邻近 bin、同一 bin 内距离比较等策略）
  3. 根据样本对构建大 JSONL 文件，供 OpenAI API 调用
  4. 调用 HuggingFaceUploader 上传图像，并根据配置选择使用 Batch API 或 Direct API 处理任务，
     同时支持断点续跑和结果合并
"""

import os
import time
import json
from typing import Dict, List, Tuple, Optional

from config import RENDER_ROOT_DIR, objective_env_prompts, CHUNK_SIZE_MIN, USE_BATCH_API, OPENAI_API_KEYS
from uploader import HuggingFaceUploader
from sample_loader import SampleLoader
from pair_generator import PairGenerator
from jsonl_manager import JSONLManager
from openai_client import DirectOpenAIManager, MultiOpenAIBatchManager

class EnvironmentProcessor:
    def __init__(self, env_name: str, objective: str, hf_uploader: HuggingFaceUploader, flip_pair_order: bool = False):
        """
        初始化 EnvironmentProcessor 实例
        :param env_name: 环境名称（例如 "metaworld_soccer-v2"）
        :param objective: 环境任务目标描述
        :param hf_uploader: HuggingFaceUploader 实例（用于图像上传）
        :param flip_pair_order: 是否翻转样本对顺序（可用于测试大模型响应一致性）
        """
        self.env_name = env_name
        self.objective = objective
        self.hf_uploader = hf_uploader
        self.flip_pair_order = flip_pair_order

        # 构造环境目录和 meta 文件目录路径
        self.env_root = os.path.join(RENDER_ROOT_DIR, self.env_name)
        self.meta_dir = os.path.join(self.env_root, "meta")
        self.samples_dict: Dict[int, Dict] = {}

    def generate_pairs(self, max_pairs: Optional[int] = None, random_seed: int = 42, default_usage_limit: int = 50, topN_intra_bin: int = 5) -> List[Tuple[int, int, str]]:
        """
        加载样本数据后生成样本对列表，支持多种配对策略，并可选择翻转对顺序
        :param max_pairs: 最终样本对数量上限
        :param random_seed: 随机种子
        :param default_usage_limit: 每个样本在所有对中最多出现次数
        :param topN_intra_bin: 同一 bin 内选择的 topN 配对数
        :return: 样本对列表，每个元素为 (sampleA, sampleB, 标签)
        """
        loader = SampleLoader(self.env_name, self.env_root, self.meta_dir)
        samples = loader.load_samples()
        if not samples:
            print(f"[EnvProcessor][{self.env_name}] 无有效样本，处理结束。")
            return []
        self.samples_dict = samples

        pair_gen = PairGenerator(samples)
        pairs = pair_gen.generate_pairs(max_pairs=max_pairs, random_seed=random_seed, usage_limit=default_usage_limit, topN_intra_bin=topN_intra_bin)
        if not pairs:
            print(f"[EnvProcessor][{self.env_name}] 生成样本对为空。")
            return []
        if self.flip_pair_order:
            pairs = [(b, a, f"{tag}(FLIPPED)") for (a, b, tag) in pairs]
            print(f"[EnvProcessor][{self.env_name}] 翻转了样本对顺序，共 {len(pairs)} 对。")
        return pairs

    def create_big_jsonl(self, pairs: List[Tuple[int, int, str]], out_file: str) -> None:
        """
        根据样本对列表生成大 JSONL 文件，供 OpenAI API 调用使用
        :param pairs: 样本对列表
        :param out_file: 输出 JSONL 文件路径
        """
        jsonl_mgr = JSONLManager(self.env_name, self.objective, self.samples_dict, self.hf_uploader)
        jsonl_mgr.create_big_jsonl(pairs, out_file)

    def chunk_file(self, file_path: str, chunk_size: int) -> List[str]:
        """
        对大 JSONL 文件进行分块，返回分块后的文件路径列表
        :param file_path: 大 JSONL 文件路径
        :param chunk_size: 每个子文件最大行数
        :return: 分块文件列表
        """
        jsonl_mgr = JSONLManager(self.env_name, self.objective, self.samples_dict, self.hf_uploader)
        return jsonl_mgr.chunk_file(file_path, chunk_size)

    def merge_outputs(self, chunk_out_files: List[str], final_out: str) -> None:
        """
        合并多个子结果文件为最终输出文件
        :param chunk_out_files: 子结果文件路径列表
        :param final_out: 最终输出文件路径
        """
        jsonl_mgr = JSONLManager(self.env_name, self.objective, self.samples_dict, self.hf_uploader)
        jsonl_mgr.merge_outputs(chunk_out_files, final_out)

    def _build_resume_map(self, chunk_files: List[str]) -> Dict[str, str]:
        """
        构建 resume_map：若某个 chunk 文件已存在对应输出文件（规则：chunk文件名+"_output.jsonl"），则记录映射
        :param chunk_files: 分块后的 JSONL 文件列表
        :return: resume_map 字典，键为 chunk 文件路径，值为对应输出文件路径
        """
        resume_map = {}
        for cf in chunk_files:
            outp = os.path.splitext(cf)[0] + "_output.jsonl"
            if os.path.isfile(outp):
                resume_map[cf] = outp
        return resume_map

    def process_environment(self, max_pairs: Optional[int] = None, random_seed: int = 42, default_usage_limit: int = 50, topN_intra_bin: int = 5) -> None:
        """
        环境处理整体流程：
          1. 生成样本对
          2. 生成大 JSONL 文件
          3. 上传剩余图像（强制 flush 缓冲区）
          4. 根据配置选择 Batch API 或 Direct API 处理任务，并最终合并结果
        :param max_pairs: 限制生成样本对数量（可选）
        """
        print(f"\n=== [EnvProcessor] 处理环境：{self.env_name}，flip_pair_order={self.flip_pair_order} ===")
        pairs = self.generate_pairs(max_pairs=max_pairs, random_seed=random_seed, default_usage_limit=default_usage_limit, topN_intra_bin=topN_intra_bin)
        if not pairs:
            print(f"[EnvProcessor][{self.env_name}] 未生成任何样本对，处理结束。")
            return

        # 创建输出目录，使用时间戳确保唯一性
        ts_str = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join("dataCollection", "Dataset", self.env_name, ts_str)
        os.makedirs(out_dir, exist_ok=True)

        big_json_file = os.path.join(out_dir, "batch_input_all.jsonl")
        self.create_big_jsonl(pairs, big_json_file)

        # 强制提交所有未提交的上传操作
        self.hf_uploader.finalize(self.env_name)

        if USE_BATCH_API:
            print(f"[EnvProcessor][{self.env_name}] 使用 Batch API 处理任务。")
            chunk_files = self.chunk_file(big_json_file, CHUNK_SIZE_MIN)
            resume_map = self._build_resume_map(chunk_files)
            batch_manager = MultiOpenAIBatchManager(resume_map=resume_map)
            batch_manager.load_handlers(OPENAI_API_KEYS)
            out_files = batch_manager.process_chunk_files_official(chunk_files, out_dir)
            final_out = os.path.join(out_dir, "batch_output_merged.jsonl")
            self.merge_outputs(out_files, final_out)
            print(f"[EnvProcessor][{self.env_name}] Batch API 任务处理完成，输出文件：{final_out}")
        else:
            print(f"[EnvProcessor][{self.env_name}] 使用 Direct API 处理任务。")
            direct_manager = DirectOpenAIManager()
            direct_manager.load_handlers(OPENAI_API_KEYS)
            tasks = []
            with open(big_json_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        tasks.append(json.loads(line))
                    except Exception as e:
                        print(f"[EnvProcessor] 解析任务请求失败：{e}")
            final_out = os.path.join(out_dir, "direct_api_results.jsonl")
            results = direct_manager.process_tasks(tasks, resume_file=final_out)
            merged_results = {}
            if os.path.isfile(final_out):
                with open(final_out, "r", encoding="utf-8") as rf:
                    for line in rf:
                        try:
                            obj = json.loads(line)
                            cid = obj.get("custom_id")
                            if cid:
                                merged_results[cid] = obj
                        except Exception as e:
                            print(f"[EnvProcessor] 解析历史记录失败：{e}")
            for cid, resp in results.items():
                merged_results[cid] = {"custom_id": cid, "response": resp}
            def parse_idx(custom_id: str) -> int:
                try:
                    return int(custom_id.rsplit("-", 1)[-1])
                except:
                    return 999999
            sorted_ids = sorted(merged_results.keys(), key=lambda cid: parse_idx(cid))
            with open(final_out, "w", encoding="utf-8") as wf:
                for cid in sorted_ids:
                    wf.write(json.dumps(merged_results[cid], ensure_ascii=False) + "\n")
            print(f"[EnvProcessor][{self.env_name}] Direct API 任务处理完成，输出文件：{final_out}")
