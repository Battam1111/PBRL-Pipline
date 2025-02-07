#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
environment_processor.py

本模块负责处理单个环境的数据生成及任务构造流程，主要包括：
  1. 加载环境样本数据并进行预处理（过滤无视角样本）。
  2. 生成样本对（支持跨 bin、邻近 bin 及基于 embedding 距离配对）。
  3. 根据样本对构造大 JSONL 文件（任务请求）。
  4. 强制提交所有未上传图像，并调用 Gemini API 处理任务。
     根据配置选择使用 Batch 模式或直连 Direct 模式处理任务，并最终合并所有任务结果。
  5. 如果批量任务中部分任务返回错误（例如限流错误），则自动对这些任务进行重试（采用直连 Direct 模式）。
  6. 输出最终结果文件。

所有步骤均附有详细中文注释，确保代码逻辑清晰且鲁棒。
"""

import os
import time
import json
import logging
from typing import Dict, List, Tuple, Optional

from config import (
    RENDER_ROOT_DIR, objective_env_prompts, CHUNK_SIZE_MIN, USE_BATCH_API,
    GEMINI_MODEL, MAXPAIRS, MAX_TASK_RETRIES,  # MAX_TASK_RETRIES统一配置
    BATCH_INPUT_BUCKET, BATCH_OUTPUT_BUCKET,
    BATCH_INPUT_PATH_PREFIX, BATCH_OUTPUT_PATH_PREFIX
)
from uploader import HuggingFaceUploader
from sample_loader import SampleLoader
from pair_generator import PairGenerator
from jsonl_manager import JSONLManager
from gemini_client import GeminiAPIClient, DirectGeminiManager
from gemini_batch_client import GeminiBatchClient
from gcs_helper import upload_file_to_gcs
from api_key_manager import APIKeyManager

# 配置日志输出（INFO级别）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class EnvironmentProcessor:
    def __init__(self, env_name: str, objective: str, hf_uploader: HuggingFaceUploader,
                 key_manager: APIKeyManager, flip_pair_order: bool = False):
        """
        初始化 EnvironmentProcessor 实例。

        参数：
          env_name: 环境名称（例如 "metaworld_soccer-v2"）。
          objective: 环境任务目标描述。
          hf_uploader: 用于图像上传的 HuggingFaceUploader 实例。
          key_manager: APIKeyManager 实例，用于管理多个 Gemini API Key 和项目 ID。
          flip_pair_order: 是否翻转样本对顺序（用于测试模型一致性）。
        """
        self.env_name = env_name
        self.objective = objective
        self.hf_uploader = hf_uploader
        self.key_manager = key_manager
        self.flip_pair_order = flip_pair_order

        self.env_root = os.path.join(RENDER_ROOT_DIR, self.env_name)
        self.meta_dir = os.path.join(self.env_root, "meta")
        self.samples_dict: Dict[int, Dict] = {}

    def generate_pairs(self, max_pairs: Optional[int] = None, random_seed: int = 42,
                       usage_limit: int = 50, topN_intra_bin: int = 5) -> List[Tuple[int, int, str]]:
        """
        加载样本数据后生成样本对列表，支持多种配对策略。

        参数：
          max_pairs: 最终生成的样本对数量上限。
          random_seed: 随机种子，用于结果复现。
          usage_limit: 每个样本在所有对中最多出现次数。
          topN_intra_bin: 同一 bin 内选择的 topN 对数。
        返回：
          样本对列表，每个元素为 (sampleA, sampleB, 标签)。
        """
        loader = SampleLoader(self.env_name, self.env_root, self.meta_dir)
        samples = loader.load_samples()
        if not samples:
            logging.info(f"[EnvProcessor][{self.env_name}] 无有效样本，处理结束。")
            return []
        self.samples_dict = samples

        pair_gen = PairGenerator(samples)
        pairs = pair_gen.generate_pairs(max_pairs=max_pairs, random_seed=random_seed,
                                        usage_limit=usage_limit, topN_intra_bin=topN_intra_bin)
        if not pairs:
            logging.info(f"[EnvProcessor][{self.env_name}] 生成样本对为空。")
            return []
        if self.flip_pair_order:
            pairs = [(b, a, f"{tag}(FLIPPED)") for (a, b, tag) in pairs]
            logging.info(f"[EnvProcessor][{self.env_name}] 翻转了样本对顺序，共 {len(pairs)} 对。")
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

    def process_environment(self, max_pairs: Optional[int] = None, random_seed: int = 42,
                            usage_limit: int = 50, topN_intra_bin: int = 5) -> None:
        """
        环境处理整体流程：
          1. 生成样本对。
          2. 生成大 JSONL 文件（任务请求）。
          3. 强制提交所有未提交的图像上传操作。
          4. 根据配置选择使用 Batch 模式或直连 Direct 模式处理任务，并最终合并结果。
          5. 对于批量模式下返回错误的任务（例如限流错误），自动使用直连模式重试，直到达到最大重试次数。

        参数：
          max_pairs, random_seed, usage_limit, topN_intra_bin：参见 generate_pairs 方法说明。
        """
        logging.info(f"=== [EnvProcessor] 处理环境：{self.env_name}，flip_pair_order={self.flip_pair_order} ===")
        pairs = self.generate_pairs(max_pairs=max_pairs, random_seed=random_seed,
                                     usage_limit=usage_limit, topN_intra_bin=topN_intra_bin)
        if not pairs:
            logging.info(f"[EnvProcessor][{self.env_name}] 未生成任何样本对，处理结束。")
            return

        ts_str = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join("dataCollection", "Dataset", self.env_name, ts_str)
        os.makedirs(out_dir, exist_ok=True)

        # 生成大 JSONL 文件（任务请求文件）
        big_json_file = os.path.join(out_dir, "batch_input_all.jsonl")
        self.create_big_jsonl(pairs, big_json_file)

        # 强制提交所有未上传的图像（确保图片已上传到 Hugging Face 仓库）
        self.hf_uploader.finalize(self.env_name)

        final_out = os.path.join(out_dir, "gemini_results.jsonl")

        if USE_BATCH_API:
            logging.info(f"[EnvProcessor][{self.env_name}] 使用 Batch 模式处理任务。")
            # 上传 JSONL 文件到 GCS
            gcs_input_path = os.path.join(BATCH_INPUT_PATH_PREFIX, self.env_name, ts_str, "batch_input_all.jsonl")
            input_uri = upload_file_to_gcs(big_json_file, BATCH_INPUT_BUCKET, gcs_input_path)
            logging.info(f"[EnvProcessor][{self.env_name}] JSONL 输入文件已上传至 GCS：{input_uri}")
            # 构造 GCS 输出路径前缀
            output_uri_prefix = f"gs://{BATCH_OUTPUT_BUCKET}/{BATCH_OUTPUT_PATH_PREFIX}/{self.env_name}/{ts_str}"
            # 初始化 GeminiBatchClient，传入 APIKeyManager
            batch_client = GeminiBatchClient(
                key_manager=self.key_manager,
                location="us-central1",  # 请确保区域与配置一致
                model=GEMINI_MODEL
            )
            display_name = f"{self.env_name}_Batch_{ts_str}"
            try:
                job_name = batch_client.submit_batch_job(
                    display_name=display_name,
                    input_uri=input_uri,
                    output_uri=output_uri_prefix
                )
                job_info = batch_client.poll_job(job_name, poll_interval=30, timeout=3600)
                # 下载批量任务结果
                results = batch_client.download_results(
                    output_uri_prefix=output_uri_prefix,
                    local_output_dir=os.path.join(out_dir, "batch_output")
                )
                logging.info(f"[EnvProcessor][{self.env_name}] 批量任务初次处理完成，已获得 {len(results)} 条结果。")
                # 检查结果，找出出错的任务（错误字段存在或回复为空）
                task_results = {item.get("custom_id"): item.get("response", {}) for item in results}
                error_tasks = {}
                # 重新读取原始任务请求文件，确保所有任务信息可用
                tasks = []
                with open(big_json_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            task = json.loads(line)
                            tasks.append(task)
                        except Exception as e:
                            logging.info(f"[EnvProcessor] 解析任务请求失败：{e}")
                for task in tasks:
                    cid = task.get("custom_id")
                    resp = task_results.get(cid, {})
                    # 如果响应中存在 error 字段或返回空文本，则视为错误
                    if not resp.get("text") or resp.get("error"):
                        error_tasks[cid] = task
                # 对出错任务进行重试（使用直连 Direct 模式），最多重试 MAX_TASK_RETRIES 次
                retry_count = 0
                while error_tasks and retry_count < MAX_TASK_RETRIES:
                    logging.info(f"[EnvProcessor][{self.env_name}] 重试第 {retry_count+1} 次，待重试任务数：{len(error_tasks)}")
                    # 使用直连模式重试错误任务
                    gemini_client = GeminiAPIClient(key_manager=self.key_manager)
                    direct_manager = DirectGeminiManager(gemini_client)
                    # 重试任务列表
                    retry_tasks = list(error_tasks.values())
                    retry_results = direct_manager.process_tasks(retry_tasks)
                    # 更新 task_results：对于此次重试成功的任务，覆盖原有错误结果
                    for cid, resp in retry_results.items():
                        # 如果重试后回复非空且无 error，则更新
                        if resp.get("text") and not resp.get("error"):
                            task_results[cid] = resp
                            if cid in error_tasks:
                                del error_tasks[cid]
                    retry_count += 1
                    if error_tasks:
                        logging.info(f"[EnvProcessor][{self.env_name}] 重试后仍有 {len(error_tasks)} 个任务出错。")
                        time.sleep(2)  # 等待一段时间后继续重试
                if error_tasks:
                    logging.error(f"[EnvProcessor][{self.env_name}] 经 {MAX_TASK_RETRIES} 次重试后，仍有 {len(error_tasks)} 个任务失败。")
                # 整合最终结果（包括批量任务中成功的和重试后成功的）
                final_results = []
                for cid, resp in task_results.items():
                    final_results.append({"custom_id": cid, "response": resp})
                # 对结果按 custom_id 排序后输出
                def parse_idx(custom_id: str) -> int:
                    try:
                        return int(custom_id.rsplit("-", 1)[-1])
                    except:
                        return 999999
                sorted_ids = sorted([item.get("custom_id") for item in final_results], key=lambda cid: parse_idx(cid))
                with open(final_out, "w", encoding="utf-8") as wf:
                    for cid in sorted_ids:
                        # 找到对应结果
                        for item in final_results:
                            if item.get("custom_id") == cid:
                                wf.write(json.dumps(item, ensure_ascii=False) + "\n")
                                break
                logging.info(f"[EnvProcessor][{self.env_name}] 最终输出文件生成：{final_out}")
            except Exception as e:
                logging.error(f"[EnvProcessor][{self.env_name}] Batch 模式任务处理出现错误：{e}")
        else:
            logging.info(f"[EnvProcessor][{self.env_name}] 使用直连 Direct 模式处理任务。")
            tasks = []
            with open(big_json_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        tasks.append(json.loads(line))
                    except Exception as e:
                        logging.info(f"[EnvProcessor] 解析任务请求失败：{e}")
            gemini_client = GeminiAPIClient(key_manager=self.key_manager)
            direct_manager = DirectGeminiManager(gemini_client)
            results = direct_manager.process_tasks(tasks)
            final_results = {}
            for cid, resp in results.items():
                final_results[cid] = {"custom_id": cid, "response": resp}
            def parse_idx(custom_id: str) -> int:
                try:
                    return int(custom_id.rsplit("-", 1)[-1])
                except:
                    return 999999
            sorted_ids = sorted(final_results.keys(), key=lambda cid: parse_idx(cid))
            with open(final_out, "w", encoding="utf-8") as wf:
                for cid in sorted_ids:
                    wf.write(json.dumps(final_results[cid], ensure_ascii=False) + "\n")
            logging.info(f"[EnvProcessor][{self.env_name}] Direct 模式任务处理完成，输出文件：{final_out}")
