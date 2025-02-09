#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
openai_client.py

本模块封装了 OpenAI API 的两种调用模式：
1. 直接调用模式（Direct）：实时请求，调用 /v1/chat/completions 接口；
2. 批处理模式（Batch）：适用于大批量异步任务，通过上传任务文件、轮询状态、下载结果完成任务。

【多 API Key 协同优化说明】：
- 同时加载多个 API Key，每个 API Key 封装为独立处理器实例；
- 采用线程安全的轮询调度，确保请求均衡分配，同时对每个 API Key 单独维护预留的 Token 数（单个 API Key 的上限由 ENQUEUED_TOKEN_LIMIT 定义）；
- 当某个 API Key 因余额不足（HTTP 402）或其他严重异常被禁用时，会自动跳过；
- 批处理模式中，各关键步骤（文件上传、任务创建、轮询、下载）均采用重试与超时机制，支持断点续传，避免重复处理已完成的任务；
- 对于任务轮询，当任务状态为“validating”、“running”、“in_progress”、“finalizing”等时，继续等待；若状态为“failed”、“cancelled”、“expired”时则尝试重建批任务；
- 对于批任务的 token 管理，采用预留／释放机制，并引入 APIKey 健康度管理（失败计数），确保每个 API Key 的累计预留 token 数不超过 ENQUEUED_TOKEN_LIMIT，同时允许偶发失败后重用当前 APIKey（失败次数未超阈值）。

本模块依赖 config.py 中的常量（如 API_URL、BATCH_API_URL、FILES_API_URL、MODEL、SYSTEM_PROMPT、MAX_CONCURRENT_BATCHES、ENQUEUED_TOKEN_LIMIT、CHUNK_SIZE_MAX 等）以及 utils.py 中的 log() 函数。
"""

import os
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
import requests
from config import API_URL, MODEL, SYSTEM_PROMPT, BATCH_API_URL, FILES_API_URL, MAX_CONCURRENT_BATCHES, ENQUEUED_TOKEN_LIMIT, CHUNK_SIZE_MAX
from utils import log

# =============================================================================
# 自定义异常：余额不足异常
# =============================================================================
class InsufficientBalanceError(Exception):
    """
    当 API Key 余额不足或被禁用时抛出此异常。
    """
    pass

# =============================================================================
# 基础处理类：封装重试机制与日志输出
# =============================================================================
class BaseOpenAIHandler:
    """
    OpenAI API 请求处理基类。
    
    提供统一的重试请求方法，采用指数退避策略，并详细记录日志；
    当遇到 HTTP 402（余额不足）时，将禁用当前 API Key 并抛出异常。
    """
    def __init__(self, api_key: str, key_index: int, max_retries: int = 99999, initial_delay: int = 3):
        self.api_key = api_key
        self.key_index = key_index
        self.disabled = False
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        # 默认使用 JSON 请求；对于文件上传后续会自动调整 headers
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def name_tag(self) -> str:
        """返回用于日志输出的 API Key 标识。"""
        return f"APIKey#{self.key_index}"

    def robust_request(self, func: Callable, *args, **kwargs):
        """
        通用请求重试函数：
          - 出现异常时采用指数退避策略重试；
          - 针对 HTTP 429（限流）和 HTTP 402（余额不足）分别处理。
        
        :param func: 执行请求的函数
        :return: 请求成功返回的结果
        :raises RuntimeError: 超过最大重试次数后抛出异常
        """
        delay = self.initial_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                status = e.response.status_code if (hasattr(e, 'response') and e.response) else None
                if status == 429:
                    log(f"[{self.name_tag()}] HTTP 429 限流，等待 {delay} 秒，重试第 {attempt} 次。")
                elif status == 402:
                    log(f"[{self.name_tag()}] HTTP 402 余额不足，禁用该 API Key。")
                    self.disabled = True
                    raise InsufficientBalanceError(f"{self.name_tag()} 余额不足：{e}")
                else:
                    log(f"[{self.name_tag()}] 请求异常：{e}，等待 {delay} 秒，重试第 {attempt} 次。")
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"[{self.name_tag()}] 超过最大重试次数 {self.max_retries}，操作失败。")

# =============================================================================
# 1. 直接调用模式处理器（Direct）
# =============================================================================
class DirectOpenAIHandler(BaseOpenAIHandler):
    """
    直接调用模式处理器：直接调用 OpenAI /v1/chat/completions 接口获取响应。
    """
    def __init__(self, api_key: str, key_index: int):
        super().__init__(api_key, key_index)

    def call_openai_api(self, payload: dict) -> dict:
        """
        调用 OpenAI /v1/chat/completions 接口，并返回响应的 JSON 数据。
        
        :param payload: 请求负载（包含 model、messages、max_tokens 等字段）。
        :return: API 返回的 JSON 数据。
        """
        def do_call():
            if self.disabled:
                raise InsufficientBalanceError(f"{self.name_tag()} 已禁用。")
            response = requests.post(API_URL, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        return self.robust_request(do_call)

    def process_task(self, payload: dict) -> dict:
        """
        处理单个任务，调用 chat completions 接口。
        
        :param payload: 请求负载。
        :return: API 响应结果（JSON 格式）。
        """
        return self.call_openai_api(payload)

class DirectOpenAIManager:
    """
    直接调用模式管理器，支持多 API Key 并发调用。
    
    采用线程安全的轮询调度（Round-Robin）方式，每次请求均均衡使用各 API Key。
    """
    def __init__(self, max_workers_per_key: int = 1):
        self.max_workers_per_key = max_workers_per_key
        self.handlers: List[DirectOpenAIHandler] = []
        self.results: Dict[str, dict] = {}
        # 轮询调度相关变量与线程锁
        self.rr_index = 0
        self.lock = threading.Lock()

    def load_handlers(self, api_keys: List[str]):
        """
        初始化所有 API Key 处理器实例。
        
        :param api_keys: API Key 列表。
        """
        self.handlers = [DirectOpenAIHandler(key, idx + 1) for idx, key in enumerate(api_keys)]
        log(f"[DirectOpenAIManager] 成功加载 {len(self.handlers)} 个 API Key 处理器。")

    def _select_handler(self) -> Optional[DirectOpenAIHandler]:
        """
        采用轮询调度，从所有处理器中选择一个未禁用的 API Key 处理器。
        
        :return: 可用的 DirectOpenAIHandler；若所有处理器均被禁用，则返回 None。
        """
        with self.lock:
            n = len(self.handlers)
            count = 0
            while count < n:
                handler = self.handlers[self.rr_index]
                self.rr_index = (self.rr_index + 1) % n
                if not handler.disabled:
                    return handler
                count += 1
        return None

    def process_tasks(self, tasks: List[Dict], resume_file: Optional[str] = None) -> Dict[str, dict]:
        """
        并发处理所有任务，并支持断点续传，避免重复处理已完成的任务。
        
        :param tasks: 任务列表，每个任务包含 custom_id 与 body 字段。
        :param resume_file: 已完成任务记录文件路径（可选）。
        :return: custom_id 到 API 响应 JSON 的映射字典。
        """
        completed_ids = set()
        if resume_file and os.path.isfile(resume_file):
            with open(resume_file, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        obj = json.loads(line)
                        cid = obj.get("custom_id")
                        if cid:
                            self.results[cid] = obj.get("response", {})
                            completed_ids.add(cid)
                    except Exception as e:
                        log(f"[DirectOpenAIManager] 读取断点文件异常：{e}")
            log(f"[DirectOpenAIManager] 断点续传：已完成 {len(completed_ids)} 个任务。")
        
        pending_tasks = [task for task in tasks if task.get("custom_id") not in completed_ids]
        log(f"[DirectOpenAIManager] 总任务数：{len(tasks)}，待处理任务：{len(pending_tasks)}。")
        
        total_workers = len(self.handlers) * self.max_workers_per_key
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            future_to_cid = {}
            for task in pending_tasks:
                future = executor.submit(self._process_single_task, task)
                future_to_cid[future] = task.get("custom_id")
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                try:
                    result = future.result()
                    self.results[cid] = result
                    log(f"[DirectOpenAIManager] 任务 {cid} 成功处理。")
                except Exception as e:
                    log(f"[DirectOpenAIManager] 任务 {cid} 处理失败：{e}")
        log("[DirectOpenAIManager] 所有任务处理完成。")
        return self.results

    def _process_single_task(self, task: Dict) -> dict:
        """
        处理单个任务，自动选择可用 API Key 并重试直至成功。
        
        :param task: 单个任务数据（包含 custom_id 和 body）。
        :return: API 响应结果（JSON 格式）。
        :raises RuntimeError: 若所有 API Key 均不可用，则抛出异常。
        """
        payload = task["body"]
        while True:
            handler = self._select_handler()
            if handler is None:
                raise RuntimeError("无可用 API Key 处理器，任务无法继续处理。")
            try:
                return handler.process_task(payload)
            except InsufficientBalanceError as ibe:
                log(f"[DirectOpenAIManager] {handler.name_tag()} 已失效，切换到其他 API Key。")
                continue
            except Exception as e:
                log(f"[DirectOpenAIManager] 任务 {task.get('custom_id')} 在 {handler.name_tag()} 上异常：{e}，稍后重试...")
                time.sleep(2)
                continue

# =============================================================================
# 2. 批处理模式处理器（Batch）
# =============================================================================
class OpenAIBatchHandler(BaseOpenAIHandler):
    """
    批处理模式处理器：用于批量任务的文件上传、批任务创建、状态查询和结果下载。
    
    注意：
      - 文件上传时不指定 Content-Type，由 requests 自动处理 multipart/form-data；
      - 轮询查询中，当任务状态为 completed 且返回 output_file_id 时，立即下载结果文件；
      - 若任务状态为 "failed"、"cancelled" 或 "expired" 时，需重建批任务；
      - 对于状态为 "validating"、"running"、"in_progress"、"finalizing" 等状态，继续等待。
    """
    def __init__(self, api_key: str, key_index: int):
        super().__init__(api_key, key_index)
        # 文件上传接口不指定 Content-Type
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def upload_batch_file(self, file_path: str) -> str:
        """
        上传 JSONL 文件到 /v1/files 接口，用于批处理任务。
        
        :param file_path: 本地 JSONL 文件路径（必须以 .jsonl 结尾）。
        :return: 返回上传后获得的 file_id。
        """
        if self.disabled:
            raise InsufficientBalanceError(f"{self.name_tag()} 已禁用。")
        if not file_path.endswith(".jsonl"):
            raise ValueError(f"{self.name_tag()} 仅支持上传 .jsonl 文件。")
        
        def do_upload():
            if self.disabled:
                raise InsufficientBalanceError(f"{self.name_tag()} 已禁用。")
            with open(file_path, "rb") as f:
                response = requests.post(
                    FILES_API_URL,
                    headers=self.headers,
                    files={"file": (os.path.basename(file_path), f, "application/json")},
                    data={"purpose": "batch"}
                )
            response.raise_for_status()
            result = response.json()
            if "id" not in result:
                raise RuntimeError(f"{self.name_tag()} 上传文件未返回 id。")
            return result["id"]
        file_id = self.robust_request(do_upload)
        log(f"[{self.name_tag()}] 成功上传文件，获得 file_id: {file_id}")
        return file_id

    def create_batch(self, file_id: str) -> dict:
        """
        根据上传的 file_id 创建批处理任务。
        
        :param file_id: 文件上传返回的 file_id。
        :return: 批任务创建成功后返回的任务信息（包含 batch_id）。
        """
        if self.disabled:
            raise InsufficientBalanceError(f"{self.name_tag()} 已禁用。")
        def do_create():
            payload = {
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h"
            }
            response = requests.post(BATCH_API_URL, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if "id" not in result:
                raise RuntimeError(f"{self.name_tag()} 批任务创建失败，无 batch_id。")
            return result
        batch_info = self.robust_request(do_create)
        log(f"[{self.name_tag()}] 批任务创建成功，batch_id: {batch_info.get('id')}")
        return batch_info

    def check_batch_status(self, batch_id: str) -> dict:
        """
        查询批任务状态信息。
        
        :param batch_id: 批任务 ID。
        :return: 返回任务状态信息的字典。
        """
        if self.disabled:
            raise InsufficientBalanceError(f"{self.name_tag()} 已禁用。")
        def do_check():
            url = f"{BATCH_API_URL}/{batch_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        status_info = self.robust_request(do_check)
        log(f"[{self.name_tag()}] 批任务 {batch_id} 状态查询结果: {status_info.get('status')}")
        return status_info

    def download_batch_results(self, output_file_id: str, save_path: str):
        """
        下载批任务结果文件，并保存到指定本地路径。
        
        :param output_file_id: 结果文件的 file_id。
        :param save_path: 结果文件的保存路径。
        """
        if self.disabled:
            raise InsufficientBalanceError(f"{self.name_tag()} 已禁用。")
        def do_download():
            url = f"{FILES_API_URL}/{output_file_id}/content"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        content = self.robust_request(do_download)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        log(f"[{self.name_tag()}] 成功下载批任务结果，保存至 {save_path}")

# =============================================================================
# 3. 批处理模式管理器（MultiOpenAIBatchManager）
# =============================================================================
# 定义 APIKey 在批任务重建过程中允许的最大连续失败次数（低于此阈值时允许重用当前 APIKey）
MAX_KEY_FAILURE_THRESHOLD = 99999

class MultiOpenAIBatchManager:
    """
    批处理模式管理器：负责管理多个 JSONL 子文件（chunk）的批任务处理，
    主要功能包括：
      1. 分块任务初始化（支持断点续传）；
      2. 采用线程池控制同时运行的批任务数量（由 MAX_CONCURRENT_BATCHES 限制）；
      3. 支持 API Key 间的轮询调度，同时对每个 API Key 维护累计预留的 token 数，
         确保单个 API Key 的预留 token 数不超过 ENQUEUED_TOKEN_LIMIT；
      4. 对每个子文件任务，进行文件上传、批任务创建、状态轮询，直至任务完成或失败；
      5. 当任务状态为异常（failed、cancelled、expired）时，自动尝试重新分配 API Key 并重建批任务；
         如果只有一个 API Key，则允许重用该 API Key；若多个 API Key可用，则在当前 API Key 的连续失败次数超过
         MAX_KEY_FAILURE_THRESHOLD 时，优先选择其他 API Key重建任务。
      6. 最终返回各子文件对应的结果文件列表，支持断点续传。
    """
    def __init__(self, max_failures_per_job: int = 99999, poll_interval: int = 10, resume_map: Optional[Dict[str, str]] = None):
        """
        :param max_failures_per_job: 单个任务允许的最大失败次数。
        :param poll_interval: 每次轮询间隔（秒）。
        :param resume_map: 已完成任务映射，格式为 {chunk_file: output_file}，用于断点续传。
        """
        self.max_failures_per_job = max_failures_per_job
        self.poll_interval = poll_interval
        self.resume_map = resume_map if resume_map else {}
        self.handlers: List[OpenAIBatchHandler] = []
        # 轮询调度相关变量与锁
        self.rr_index = 0
        self.lock = threading.Lock()
        # 记录每个 API Key 当前累计预留的 token 数（键为 handler.name_tag()）
        self.handler_token_counts: Dict[str, int] = {}
        # 记录每个 API Key 的连续失败次数（非网络异常，仅针对批任务重建失败）
        self.api_key_failures: Dict[str, int] = {}

    def load_handlers(self, api_keys: List[str]):
        """
        初始化所有批处理 API Key 处理器实例，并初始化各自的 token 计数和失败计数。
        
        :param api_keys: API Key 列表。
        """
        self.handlers = [OpenAIBatchHandler(key, idx + 1) for idx, key in enumerate(api_keys)]
        for handler in self.handlers:
            tag = handler.name_tag()
            self.handler_token_counts[tag] = 0
            self.api_key_failures[tag] = 0
        log(f"[MultiOpenAIBatchManager] 成功加载 {len(self.handlers)} 个批处理 API Key 处理器。")

    # ------------------------------
    # Token 管理相关方法
    # ------------------------------
    def reserve_tokens(self, handler: OpenAIBatchHandler, tokens: int) -> bool:
        """
        尝试在指定处理器上预留一定数量的 token。
        
        :param handler: 指定的批处理处理器。
        :param tokens: 预留的 token 数。
        :return: 预留成功返回 True，否则返回 False。
        """
        with self.lock:
            key = handler.name_tag()
            current = self.handler_token_counts.get(key, 0)
            if current + tokens <= ENQUEUED_TOKEN_LIMIT:
                self.handler_token_counts[key] = current + tokens
                log(f"[MultiOpenAIBatchManager] {key} 预留 token {tokens} 成功，当前累计 token 数为 {self.handler_token_counts[key]}。")
                return True
            else:
                log(f"[MultiOpenAIBatchManager] {key} 无法预留 token {tokens}（当前已预留 {current}，上限 {ENQUEUED_TOKEN_LIMIT}）。")
                return False

    def release_tokens(self, handler: OpenAIBatchHandler, tokens: int):
        """
        在指定处理器上释放预留的 token 数。
        
        :param handler: 指定的批处理处理器。
        :param tokens: 释放的 token 数。
        """
        with self.lock:
            key = handler.name_tag()
            current = self.handler_token_counts.get(key, 0)
            new_value = max(0, current - tokens)
            self.handler_token_counts[key] = new_value
            log(f"[MultiOpenAIBatchManager] {key} 释放 token {tokens}，当前累计 token 数更新为 {new_value}。")

    def _select_handler(self, job_token: int) -> Optional[OpenAIBatchHandler]:
        """
        采用轮询调度，从所有处理器中选择一个未禁用且当前累计预留 token 数加上本任务 token 数不超过 ENQUEUED_TOKEN_LIMIT 的处理器。
        
        :param job_token: 当前任务大致的 token 数。
        :return: 可用的 OpenAIBatchHandler；若所有处理器均不可用则返回 None。
        """
        with self.lock:
            n = len(self.handlers)
            count = 0
            while count < n:
                handler = self.handlers[self.rr_index]
                self.rr_index = (self.rr_index + 1) % n
                if handler.disabled:
                    count += 1
                    continue
                current_tokens = self.handler_token_counts.get(handler.name_tag(), 0)
                if current_tokens + job_token <= ENQUEUED_TOKEN_LIMIT:
                    return handler
                count += 1
        return None

    def _approximate_token_count(self, file_path: str) -> int:
        """
        近似计算文件的 token 数，粗略按照每 4 个字符计 1 个 token。
        
        :param file_path: 文件路径。
        :return: 近似 token 数。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return len(text) // 4

    # ------------------------------
    # 轮询任务状态与重建任务
    # ------------------------------
    def _poll_job(self, job: Dict[str, Any]) -> Optional[str]:
        """
        轮询单个批任务，直至任务完成、失败或超过最大重试次数／超时。
        
        当任务状态为 completed 且返回 output_file_id 时，下载结果文件并返回保存路径；
        当状态为异常（failed、cancelled、expired）时，尝试重建批任务（重新分配 API Key 并重新提交任务）。
        如果只有一个 API Key，则允许重用该 API Key；若多个 API Key 可用，则当当前 API Key连续失败次数超过
        MAX_KEY_FAILURE_THRESHOLD 时，优先选择其他 API Key重建任务。
        
        :param job: 任务字典，包含以下字段：
                    - chunk_file: 子文件路径；
                    - output_file: 结果文件保存路径；
                    - fail_count: 当前任务累计失败次数；
                    - handler: 当前分配的 API Key 处理器；
                    - batch_id: 当前批任务 ID；
                    - token_count: 本任务的 token 数。
        :return: 成功时返回结果文件保存路径，否则返回 None。
        """
        fail_count = job.get("fail_count", 0)
        start_time = time.time()
        while True:
            try:
                status_info = job["handler"].check_batch_status(job["batch_id"])
                status = status_info.get("status", "").lower()
                current_time = time.time()

                if status == "completed":
                    output_file_id = status_info.get("output_file_id", "")
                    if output_file_id:
                        # 下载结果文件
                        job["handler"].download_batch_results(output_file_id, job["output_file"])
                        log(f"[MultiOpenAIBatchManager] 任务 {job['chunk_file']} 已完成，结果文件: {job['output_file']}")
                        # 任务结束后释放预留 token，并重置当前 APIKey 的失败计数
                        self.release_tokens(job["handler"], job["token_count"])
                        self.api_key_failures[job["handler"].name_tag()] = 0
                        return job["output_file"]
                    else:
                        log(f"[MultiOpenAIBatchManager] 任务 {job['chunk_file']} 状态为 completed，但未返回 output_file_id。")
                        fail_count += 1

                elif status in ("failed", "cancelled", "expired"):
                    log(f"[MultiOpenAIBatchManager] 任务 {job['chunk_file']} 状态异常: {status}，准备重建批任务。")
                    fail_count += 1
                    # 累计当前 APIKey 对此任务的失败次数
                    current_key = job["handler"].name_tag()
                    self.api_key_failures[current_key] += 1
                    # 若只有一个 APIKey，或当前 APIKey连续失败次数低于阈值，则允许重用当前 APIKey
                    if len(self.handlers) == 1 or self.api_key_failures[current_key] < MAX_KEY_FAILURE_THRESHOLD:
                        log(f"[MultiOpenAIBatchManager] 当前 APIKey {current_key} 失败次数 {self.api_key_failures[current_key]} 未超过阈值，尝试重用。")
                        self.release_tokens(job["handler"], job["token_count"])
                        if not self.reserve_tokens(job["handler"], job["token_count"]):
                            log(f"[MultiOpenAIBatchManager] 重新预留当前 APIKey {current_key} token 失败。")
                            return None
                        # 继续使用当前 handler
                    else:
                        # 多个 APIKey情况下，当当前 APIKey连续失败次数达到阈值时，尝试选择其他 APIKey
                        log(f"[MultiOpenAIBatchManager] 当前 APIKey {current_key} 失败次数 {self.api_key_failures[current_key]} 达到阈值，尝试选择其他 APIKey。")
                        candidate = None
                        for _ in range(3):
                            cand = self._select_handler(job["token_count"])
                            if cand and cand.name_tag() != current_key:
                                # 候选 APIKey使用前尝试预留 token
                                if self.reserve_tokens(cand, job["token_count"]):
                                    candidate = cand
                                    break
                            time.sleep(self.poll_interval)
                        if candidate is None:
                            # 若未找到其他合适 APIKey，则重用当前 APIKey
                            log(f"[MultiOpenAIBatchManager] 未找到其他 APIKey，重用当前 APIKey {current_key}。")
                            self.release_tokens(job["handler"], job["token_count"])
                            if not self.reserve_tokens(job["handler"], job["token_count"]):
                                return None
                            candidate = job["handler"]
                        # 更新 job 使用新的 APIKey，并重置新 APIKey的失败计数（或保留已有值）
                        job["handler"] = candidate

                    try:
                        file_id = job["handler"].upload_batch_file(job["chunk_file"])
                        batch_info = job["handler"].create_batch(file_id)
                        job["batch_id"] = batch_info.get("id", "")
                        log(f"[MultiOpenAIBatchManager] 任务 {job['chunk_file']} 重建批任务成功，新 batch_id: {job['batch_id']}")
                        # 重建成功后，将新 APIKey的失败计数重置
                        self.api_key_failures[job["handler"].name_tag()] = 0
                    except Exception as e:
                        fail_count += 1
                        log(f"[MultiOpenAIBatchManager] 任务 {job['chunk_file']} 重建批任务异常：{e}")
                        continue

                else:
                    log(f"[MultiOpenAIBatchManager] 任务 {job['chunk_file']} 当前状态: {status}，继续轮询。")

                # 检查是否超过最大失败次数或超时
                if fail_count >= self.max_failures_per_job or (current_time - start_time) > 3600:
                    log(f"[MultiOpenAIBatchManager] 任务 {job['chunk_file']} 超过最大失败次数或超时，终止任务。")
                    self.release_tokens(job["handler"], job["token_count"])
                    return None

            except Exception as e:
                fail_count += 1
                log(f"[MultiOpenAIBatchManager] 任务 {job['chunk_file']} 轮询异常：{e}，累计失败次数: {fail_count}")
                if fail_count >= self.max_failures_per_job:
                    self.release_tokens(job["handler"], job["token_count"])
                    return None

            time.sleep(self.poll_interval)

    def process_chunk_files(self, chunk_files: List[str], out_dir: str) -> List[str]:
        """
        处理所有 JSONL 子文件（chunk），为每个文件分配批任务、轮询状态、下载结果文件。
        支持断点续传，并同时控制：
          ① 同时最多运行 MAX_CONCURRENT_BATCHES 个批任务；
          ② 每个 API Key 的累计预留 token 数不超过 ENQUEUED_TOKEN_LIMIT。
        如果某个 API Key 的 token 数超限，则暂停使用该 API Key 提交新任务，待部分任务完成后再重试。
        
        :param chunk_files: JSONL 子文件列表。
        :param out_dir: 结果文件输出目录。
        :return: 输出结果文件路径列表，顺序与输入文件顺序一致。
        """
        # 初始化任务字典，支持断点续传
        jobs = {}
        for cf in chunk_files:
            if cf in self.resume_map and os.path.isfile(self.resume_map[cf]):
                # 已完成的任务直接记录输出文件
                jobs[cf] = {"output": self.resume_map[cf], "token_count": 0}
            else:
                base = os.path.splitext(os.path.basename(cf))[0]
                output_file = os.path.join(out_dir, f"{base}_output.jsonl")
                token_count = self._approximate_token_count(cf)
                jobs[cf] = {
                    "chunk_file": cf,
                    "output_file": output_file,
                    "fail_count": 0,
                    "handler": None,
                    "batch_id": "",
                    "token_count": token_count
                }
        # 使用线程池控制最大并发批任务数量
        executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_BATCHES)
        futures = {}
        results = {}  # 存放各任务结果，键为 chunk 文件路径

        def submit_job(job_key, job) -> bool:
            """
            内部函数：尝试提交单个任务，成功则预留 token 并创建批任务。
            
            :param job_key: 任务对应的文件路径。
            :param job: 任务字典。
            :return: 提交成功返回 True，否则返回 False。
            """
            handler = self._select_handler(job["token_count"])
            if handler is None:
                log(f"[MultiOpenAIBatchManager] 任务 {job_key} 无可用 API Key（token 限制），暂不提交。")
                return False
            # 预留 token
            if not self.reserve_tokens(handler, job["token_count"]):
                log(f"[MultiOpenAIBatchManager] 任务 {job_key} 在 {handler.name_tag()} 预留 token 失败。")
                return False
            job["handler"] = handler
            try:
                file_id = handler.upload_batch_file(job["chunk_file"])
                batch_info = handler.create_batch(file_id)
                job["batch_id"] = batch_info.get("id", "")
                log(f"[MultiOpenAIBatchManager] 任务 {job_key} 创建批任务成功，batch_id: {job['batch_id']}")
            except Exception as e:
                job["fail_count"] += 1
                log(f"[MultiOpenAIBatchManager] 任务 {job_key} 创建批任务失败：{e}")
                # 失败时释放预留 token
                self.release_tokens(handler, job["token_count"])
                return False
            futures[executor.submit(self._poll_job, job)] = job_key
            log(f"[MultiOpenAIBatchManager] 提交任务 {job_key}，token 数 {job['token_count']}。")
            return True

        pending_keys = list(jobs.keys())
        idx = 0
        # 循环提交所有任务，若某任务因 token 限制无法提交则等待部分任务完成后再重试
        while idx < len(pending_keys):
            job_key = pending_keys[idx]
            job = jobs[job_key]
            if "output" in job:
                idx += 1
                continue
            submitted = submit_job(job_key, job)
            if submitted:
                idx += 1
            else:
                time.sleep(self.poll_interval)
        # 等待所有任务完成
        executor.shutdown(wait=True)
        # 收集所有任务结果
        for fut in futures:
            job_key = futures[fut]
            try:
                result = fut.result()
                results[job_key] = result if result else ""
            except Exception as e:
                log(f"[MultiOpenAIBatchManager] 任务 {job_key} 异常：{e}")
                results[job_key] = ""
        # 根据输入文件顺序返回结果文件列表
        output_files = [results.get(cf, jobs.get(cf, {}).get("output", "")) for cf in chunk_files]
        return output_files

# =============================================================================
# 本模块导出 DirectOpenAIManager 与 MultiOpenAIBatchManager 两个类，
# 分别对应直接调用模式与批处理模式，供其它模块调用。
# =============================================================================
