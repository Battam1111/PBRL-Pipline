#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
environment_processor.py

本模块负责处理单个环境的数据生成及任务构建流程，主要包括：
  1. 加载环境样本数据并进行预处理（过滤无视角样本）。
  2. 生成样本对（支持跨 bin、邻近 bin 及基于 embedding 距离配对）。
  3. 根据样本对构造大 JSONL 文件（任务请求）。
  4. 强制提交所有未上传图像，并调用 Gemini API 处理任务（支持 Batch 模式或直连 Direct 模式）。
  5. 合并各个任务处理结果输出为最终结果文件。

所有步骤均附有详细中文注释，确保代码逻辑清晰且鲁棒。
"""

import os
import time
import json
from typing import Dict, List, Tuple, Optional

from config import RENDER_ROOT_DIR, objective_env_prompts, CHUNK_SIZE_MIN, USE_BATCH_API, GEMINI_API_KEY
from uploader import HuggingFaceUploader
from sample_loader import SampleLoader
from pair_generator import PairGenerator
from jsonl_manager import JSONLManager
from gemini_client import DirectGeminiManager, MultiGeminiBatchManager

class EnvironmentProcessor:
    def __init__(self, env_name: str, objective: str, hf_uploader: HuggingFaceUploader, flip_pair_order: bool = False):
        """
        初始化 EnvironmentProcessor 实例。

        参数：
          env_name: 环境名称（例如 "metaworld_soccer-v2"）。
          objective: 环境任务目标描述。
          hf_uploader: 用于图像上传的 HuggingFaceUploader 实例。
          flip_pair_order: 是否翻转样本对顺序（用于测试模型一致性）。
        """
        self.env_name = env_name
        self.objective = objective
        self.hf_uploader = hf_uploader
        self.flip_pair_order = flip_pair_order

        self.env_root = os.path.join(RENDER_ROOT_DIR, self.env_name)
        self.meta_dir = os.path.join(self.env_root, "meta")
        self.samples_dict: Dict[int, Dict] = {}

    def generate_pairs(self, max_pairs: Optional[int] = None, random_seed: int = 42, usage_limit: int = 50, topN_intra_bin: int = 5) -> List[Tuple[int, int, str]]:
        """
        加载样本数据后生成样本对列表，支持多种配对策略。

        参数：
          max_pairs: 最终样本对数量上限。
          random_seed: 随机种子，用于复现结果。
          usage_limit: 每个样本在所有对中最多出现次数。
          topN_intra_bin: 同一 bin 内选择的 topN 对数。
        返回：
          样本对列表，每个元素为 (sampleA, sampleB, 标签)。
        """
        loader = SampleLoader(self.env_name, self.env_root, self.meta_dir)
        samples = loader.load_samples()
        if not samples:
            print(f"[EnvProcessor][{self.env_name}] 无有效样本，处理结束。")
            return []
        self.samples_dict = samples

        pair_gen = PairGenerator(samples)
        pairs = pair_gen.generate_pairs(max_pairs=max_pairs, random_seed=random_seed, usage_limit=usage_limit, topN_intra_bin=topN_intra_bin)
        if not pairs:
            print(f"[EnvProcessor][{self.env_name}] 生成样本对为空。")
            return []
        if self.flip_pair_order:
            pairs = [(b, a, f"{tag}(FLIPPED)") for (a, b, tag) in pairs]
            print(f"[EnvProcessor][{self.env_name}] 翻转了样本对顺序，共 {len(pairs)} 对。")
        return pairs

    def create_big_jsonl(self, pairs: List[Tuple[int, int, str]], out_file: str) -> None:
        """
        根据样本对生成大 JSONL 文件，供 Gemini API 任务调用使用。

        参数：
          pairs: 样本对列表。
          out_file: 输出 JSONL 文件路径。
        """
        jsonl_mgr = JSONLManager(self.env_name, self.objective, self.samples_dict, self.hf_uploader)
        jsonl_mgr.create_big_jsonl(pairs, out_file)

    def chunk_file(self, file_path: str, chunk_size: int) -> List[str]:
        """
        对大 JSONL 文件进行分块，返回分块后的文件列表。

        参数：
          file_path: 大 JSONL 文件路径。
          chunk_size: 每个子文件最大行数。
        返回：
          分块后生成的子文件路径列表。
        """
        jsonl_mgr = JSONLManager(self.env_name, self.objective, self.samples_dict, self.hf_uploader)
        return jsonl_mgr.chunk_file(file_path, chunk_size)

    def merge_outputs(self, chunk_out_files: List[str], final_out: str) -> None:
        """
        合并多个子结果文件为最终输出文件。

        参数：
          chunk_out_files: 子结果文件列表。
          final_out: 最终合并后的输出文件路径。
        """
        jsonl_mgr = JSONLManager(self.env_name, self.objective, self.samples_dict, self.hf_uploader)
        jsonl_mgr.merge_outputs(chunk_out_files, final_out)

    def _build_resume_map(self, chunk_files: List[str]) -> Dict[str, str]:
        """
        构建断点续跑映射：若某个分块文件已存在对应输出文件，则记录映射关系。

        参数：
          chunk_files: 分块后的 JSONL 文件列表。
        返回：
          resume_map 字典：键为分块文件路径，值为对应输出文件路径。
        """
        resume_map = {}
        for cf in chunk_files:
            outp = os.path.splitext(cf)[0] + "_output.jsonl"
            if os.path.isfile(outp):
                resume_map[cf] = outp
        return resume_map

    def process_environment(self, max_pairs: Optional[int] = None, random_seed: int = 42, usage_limit: int = 50, topN_intra_bin: int = 5) -> None:
        """
        环境处理整体流程：
          1. 生成样本对。
          2. 生成大 JSONL 文件（任务请求）。
          3. 强制提交所有未上传图像。
          4. 根据配置选择使用 Batch 模式或直连 Direct 模式处理任务，并最终合并结果。

        参数：
          max_pairs, random_seed, usage_limit, topN_intra_bin: 见 generate_pairs 方法说明。
        """
        print(f"\n=== [EnvProcessor] 处理环境：{self.env_name}，flip_pair_order={self.flip_pair_order} ===")
        pairs = self.generate_pairs(max_pairs=max_pairs, random_seed=random_seed, usage_limit=usage_limit, topN_intra_bin=topN_intra_bin)
        if not pairs:
            print(f"[EnvProcessor][{self.env_name}] 未生成任何样本对，处理结束。")
            return

        ts_str = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join("dataCollection", "Dataset", self.env_name, ts_str)
        os.makedirs(out_dir, exist_ok=True)

        big_json_file = os.path.join(out_dir, "batch_input_all.jsonl")
        self.create_big_jsonl(pairs, big_json_file)

        # 强制提交所有未提交的图像上传操作
        self.hf_uploader.finalize(self.env_name)

        if USE_BATCH_API:
            print(f"[EnvProcessor][{self.env_name}] 使用 Batch 模式处理任务。")
            chunk_files = self.chunk_file(big_json_file, CHUNK_SIZE_MIN)
            resume_map = self._build_resume_map(chunk_files)
            batch_manager = MultiGeminiBatchManager(resume_map=resume_map)
            # 加载 Gemini API Key（支持多 Key 扩展，此处仅使用一个）
            batch_manager.load_handlers([GEMINI_API_KEY])
            out_files = batch_manager.process_chunk_files_official(chunk_files, out_dir)
            final_out = os.path.join(out_dir, "batch_output_merged.jsonl")
            self.merge_outputs(out_files, final_out)
            print(f"[EnvProcessor][{self.env_name}] Batch 模式任务处理完成，输出文件：{final_out}")
        else:
            print(f"[EnvProcessor][{self.env_name}] 使用直连 Direct 模式处理任务。")
            direct_manager = DirectGeminiManager()
            direct_manager.load_handlers([GEMINI_API_KEY])
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
            print(f"[EnvProcessor][{self.env_name}] Direct 模式任务处理完成，输出文件：{final_out}")
