#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
environment_processor.py

本模块负责单个环境下的完整处理流程：
  1. 加载样本数据；
  2. 生成样本配对；
  3. 生成 JSONL 请求文件；
  4. 强制上传图像（调用 HuggingFaceUploader.finalize）；
  5. 根据 API 调用模式（Direct 或 Batch）调用 OpenAI 接口，并合并结果。

所有流程均附有详细日志输出与异常处理，保证整体流程鲁棒。
"""

import os
import time
import json
from typing import Dict, List, Tuple, Optional

from config import (
    RENDER_ROOT_DIR,
    objective_env_prompts,
    CHUNK_SIZE_MIN,
    USE_BATCH_API,
    OPENAI_API_KEYS,
    MAXPAIRS
)
from uploader import HuggingFaceUploader
from sample_loader import SampleLoader
from pair_generator import PairGenerator
from jsonl_manager import JSONLManager
from openai_client import DirectOpenAIManager, MultiOpenAIBatchManager
from utils import log

class EnvironmentProcessor:
    def __init__(self,
                 env_name: str,
                 objective: str,
                 hf_uploader: HuggingFaceUploader,
                 flip_pair_order: bool = False):
        """
        初始化 EnvironmentProcessor 实例
        :param env_name: 环境名称
        :param objective: 当前环境目标描述
        :param hf_uploader: HuggingFaceUploader 实例
        :param flip_pair_order: 是否翻转生成的配对顺序
        """
        self.env_name = env_name
        self.objective = objective
        self.hf_uploader = hf_uploader
        self.flip_pair_order = flip_pair_order
        self.env_root = os.path.join(RENDER_ROOT_DIR, self.env_name)
        self.meta_dir = os.path.join(self.env_root, "meta")
        self.samples_dict: Dict[int, Dict] = {}

    def generate_pairs(self,
                       max_pairs: Optional[int] = None,
                       random_seed: int = 42,
                       default_usage_limit: int = 50,
                       topN_intra_bin: int = 5) -> List[Tuple[int, int, str]]:
        """
        加载样本并生成样本配对
        """
        loader = SampleLoader(self.env_name, self.env_root, self.meta_dir)
        samples = loader.load_samples()
        self.samples_dict = samples

        if len(samples) < 2:
            log(f"[EnvProcessor][{self.env_name}] 样本数不足 2 个，无法生成有效配对。")
            return []

        pair_gen = PairGenerator(samples)
        pairs = pair_gen.generate_pairs(
            max_pairs=max_pairs,
            random_seed=random_seed,
            usage_limit=default_usage_limit,
            topN_intra_bin=topN_intra_bin,
            allow_duplicates=False,
            preserve_strategy=False,
            strict_deduplicate=True
        )

        if self.flip_pair_order and pairs:
            pairs = [(b, a, f"{tag}(FLIPPED)") for (a, b, tag) in pairs]
            log(f"[EnvProcessor][{self.env_name}] 配对顺序已翻转，共 {len(pairs)} 对。")

        log(f"[EnvProcessor][{self.env_name}] 最终生成样本对数：{len(pairs)}")
        return pairs

    def create_big_jsonl(self, pairs: List[Tuple[int,int,str]], out_file: str) -> None:
        jsonl_mgr = JSONLManager(self.env_name, self.objective, self.samples_dict, self.hf_uploader)
        jsonl_mgr.create_big_jsonl(pairs, out_file)

    def chunk_file(self, file_path: str, chunk_size: int) -> List[str]:
        jsonl_mgr = JSONLManager(self.env_name, self.objective, self.samples_dict, self.hf_uploader)
        return jsonl_mgr.chunk_file(file_path, chunk_size)

    def merge_outputs(self, chunk_out_files: List[str], final_out: str) -> None:
        jsonl_mgr = JSONLManager(self.env_name, self.objective, self.samples_dict, self.hf_uploader)
        jsonl_mgr.merge_outputs(chunk_out_files, final_out)

    def _build_resume_map(self, chunk_files: List[str]) -> Dict[str, str]:
        """
        构建断点续跑的映射：对于每个子文件，若存在对应的输出文件则记录映射关系
        """
        resume_map = {}
        for cf in chunk_files:
            outp = os.path.splitext(cf)[0] + "_output.jsonl"
            if os.path.isfile(outp):
                resume_map[cf] = outp
        return resume_map

    def process_environment(self,
                            max_pairs: Optional[int] = None,
                            random_seed: int = 42,
                            default_usage_limit: int = 50,
                            topN_intra_bin: int = 5) -> None:
        """
        串起整个环境的处理流程：
          1. 生成样本配对；
          2. 生成大 JSONL 文件；
          3. 强制上传图像（提交缓冲区）；
          4. 根据 API 模式调用 OpenAI 接口（Direct 或 Batch），并输出结果文件。
        """
        log(f"\n=== [EnvProcessor] 正在处理环境：{self.env_name}，flip_pair_order={self.flip_pair_order} ===")
        pairs = self.generate_pairs(
            max_pairs=max_pairs,
            random_seed=random_seed,
            default_usage_limit=default_usage_limit,
            topN_intra_bin=topN_intra_bin
        )
        if not pairs:
            log(f"[EnvProcessor][{self.env_name}] 无可用配对，流程终止。")
            return

        ts_str = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join("dataCollection", "Dataset", self.env_name, ts_str)
        os.makedirs(out_dir, exist_ok=True)

        big_json_file = os.path.join(out_dir, "batch_input_all.jsonl")
        self.create_big_jsonl(pairs, big_json_file)

        # 提交所有缓冲中的图像上传操作
        self.hf_uploader.finalize(self.env_name)

        if USE_BATCH_API:
            log(f"[EnvProcessor][{self.env_name}] 使用 Batch API 模式。")
            chunk_files = self.chunk_file(big_json_file, CHUNK_SIZE_MIN)
            resume_map = self._build_resume_map(chunk_files)
            batch_manager = MultiOpenAIBatchManager(resume_map=resume_map)
            batch_manager.load_handlers(OPENAI_API_KEYS)
            out_files = batch_manager.process_chunk_files(chunk_files, out_dir)
            final_out = os.path.join(out_dir, "batch_output_merged.jsonl")
            self.merge_outputs(out_files, final_out)
            log(f"[EnvProcessor][{self.env_name}] Batch 模式完成，输出文件：{final_out}")
        else:
            log(f"[EnvProcessor][{self.env_name}] 使用 Direct API 模式。")
            direct_manager = DirectOpenAIManager()
            direct_manager.load_handlers(OPENAI_API_KEYS)
            tasks = []
            with open(big_json_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        tasks.append(json.loads(line))
                    except Exception as e:
                        log(f"[EnvProcessor] 解析任务行异常：{e}")
            final_out = os.path.join(out_dir, "direct_api_results.jsonl")
            results_map = direct_manager.process_tasks(tasks, resume_file=final_out)
            # 合并已有记录与新结果
            merged_results = {}
            if os.path.isfile(final_out):
                with open(final_out, "r", encoding="utf-8") as rf:
                    for line in rf:
                        try:
                            obj = json.loads(line)
                            cid = obj.get("custom_id")
                            if cid:
                                merged_results[cid] = obj
                        except:
                            pass
            for cid, resp in results_map.items():
                merged_results[cid] = {"custom_id": cid, "response": resp}
            def parse_idx(custom_id: str) -> int:
                try:
                    return int(custom_id.rsplit("-",1)[-1])
                except:
                    return 999999
            sorted_ids = sorted(merged_results.keys(), key=lambda c: parse_idx(c))
            with open(final_out, "w", encoding="utf-8") as wf:
                for cid in sorted_ids:
                    wf.write(json.dumps(merged_results[cid], ensure_ascii=False) + "\n")
            log(f"[EnvProcessor][{self.env_name}] Direct 模式完成，输出文件：{final_out}")
