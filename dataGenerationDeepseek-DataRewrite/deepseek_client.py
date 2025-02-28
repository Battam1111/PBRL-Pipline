#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
deepseek_client.py

本模块封装了 DeepSeek API 的两种调用模式：
  1. 直接调用模式（Direct）：实时请求 /v1/chat/completions 接口；
  2. 批处理模式（Batch）：适用于大批量任务，通过文件上传、轮询任务状态、下载结果等完成批处理。

DeepSeek API 兼容 OpenAI API 格式，故本模块基于 OpenAI API 调用方式改造而成。
模块内部实现了多 API Key 协同调用、重试机制、指数退避以及断点续传功能，确保任务处理高效且鲁棒。
"""

import os
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
import requests
from config import API_URL, MODEL, BATCH_API_URL, FILES_API_URL, DEEPSEEK_API_KEYS
from utils import log

# --------------------------
# 自定义异常
# --------------------------
class InsufficientBalanceError(Exception):
    """当 API Key 余额不足或被禁用时抛出异常。"""
    pass

# --------------------------
# 基础处理类：封装重试机制和请求
# --------------------------
class BaseDeepSeekHandler:
    def __init__(self, api_key: str, key_index: int, max_retries: int = 5, initial_delay: int = 3):
        self.api_key = api_key
        self.key_index = key_index
        self.disabled = False
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def name_tag(self) -> str:
        return f"DeepSeekAPIKey#{self.key_index}"

    def robust_request(self, func: Callable, *args, **kwargs):
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

# --------------------------
# 1. 直接调用模式处理器
# --------------------------
class DeepSeekDirectHandler(BaseDeepSeekHandler):
    def __init__(self, api_key: str, key_index: int):
        super().__init__(api_key, key_index)

    def call_deepseek_api(self, payload: dict) -> dict:
        """
        调用 DeepSeek 的对话补全接口（POST /v1/chat/completions）
        """
        def do_call():
            if self.disabled:
                raise InsufficientBalanceError(f"{self.name_tag()} 已禁用。")
            response = requests.post(API_URL, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        return self.robust_request(do_call)

    def process_task(self, payload: dict) -> dict:
        return self.call_deepseek_api(payload)

class DeepSeekDirectManager:
    def __init__(self, max_workers_per_key: int = 1):
        self.max_workers_per_key = max_workers_per_key
        self.handlers: List[DeepSeekDirectHandler] = []
        self.results: Dict[str, dict] = {}
        self.rr_index = 0
        self.lock = threading.Lock()

    def load_handlers(self, api_keys: List[str]):
        self.handlers = [DeepSeekDirectHandler(key, idx+1) for idx, key in enumerate(api_keys)]
        log(f"[DeepSeekDirectManager] 成功加载 {len(self.handlers)} 个 API Key 处理器。")

    def _select_handler(self) -> Optional[DeepSeekDirectHandler]:
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
        处理任务列表，支持断点续传。
        每个任务包含 custom_id 和请求 payload（存储在 task["body"]）。
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
                        log(f"[DeepSeekDirectManager] 读取断点文件异常：{e}")
            log(f"[DeepSeekDirectManager] 断点续传：已完成 {len(completed_ids)} 个任务。")
        
        pending_tasks = [task for task in tasks if task.get("custom_id") not in completed_ids]
        log(f"[DeepSeekDirectManager] 总任务数：{len(tasks)}，待处理任务：{len(pending_tasks)}。")
        
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
                    log(f"[DeepSeekDirectManager] 任务 {cid} 成功处理。")
                except Exception as e:
                    log(f"[DeepSeekDirectManager] 任务 {cid} 处理失败：{e}")
        log("[DeepSeekDirectManager] 所有任务处理完成。")
        return self.results

    def _process_single_task(self, task: Dict) -> dict:
        payload = task["body"]
        while True:
            handler = self._select_handler()
            if handler is None:
                raise RuntimeError("无可用 API Key 处理器，任务无法继续处理。")
            try:
                return handler.process_task(payload)
            except InsufficientBalanceError as ibe:
                log(f"[DeepSeekDirectManager] {handler.name_tag()} 已失效，切换到其他 API Key。")
                continue
            except Exception as e:
                log(f"[DeepSeekDirectManager] 任务 {task.get('custom_id')} 在 {handler.name_tag()} 上异常：{e}，稍后重试...")
                time.sleep(2)
                continue

# --------------------------
# 2. 批处理模式处理器
# --------------------------
class DeepSeekBatchHandler(BaseDeepSeekHandler):
    def __init__(self, api_key: str, key_index: int):
        super().__init__(api_key, key_index)
        # 批处理接口通常不需要 Content-Type 头（依赖文件上传），但保留认证头
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def upload_batch_file(self, file_path: str) -> str:
        """
        上传 JSONL 文件用于批处理任务。
        DeepSeek API 兼容 OpenAI 文件上传接口。
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
        根据上传的文件创建批处理任务。
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
        查询批处理任务状态。
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
        下载批处理任务的结果文件内容，并保存到本地。
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

# --------------------------
# 3. 批处理模式管理器
# --------------------------
MAX_KEY_FAILURE_THRESHOLD = 5

class MultiDeepSeekBatchManager:
    def __init__(self, max_failures_per_job: int = 5, poll_interval: int = 10, resume_map: Optional[Dict[str, str]] = None):
        self.max_failures_per_job = max_failures_per_job
        self.poll_interval = poll_interval
        self.resume_map = resume_map if resume_map else {}
        self.handlers: List[DeepSeekBatchHandler] = []
        self.rr_index = 0
        self.lock = threading.Lock()
        self.handler_token_counts: Dict[str, int] = {}
        self.api_key_failures: Dict[str, int] = {}

    def load_handlers(self, api_keys: List[str]):
        self.handlers = [DeepSeekBatchHandler(key, idx+1) for idx, key in enumerate(api_keys)]
        for handler in self.handlers:
            tag = handler.name_tag()
            self.handler_token_counts[tag] = 0
            self.api_key_failures[tag] = 0
        log(f"[MultiDeepSeekBatchManager] 成功加载 {len(self.handlers)} 个批处理 API Key 处理器。")

    def reserve_tokens(self, handler: DeepSeekBatchHandler, tokens: int) -> bool:
        """
        简单的 token 预留机制，用于控制每个 API Key 当次请求的 token 累计，防止超出限制。
        这里采用固定阈值 10000（可根据实际情况调整）。
        """
        with self.lock:
            key = handler.name_tag()
            current = self.handler_token_counts.get(key, 0)
            if current + tokens <= 10000:
                self.handler_token_counts[key] = current + tokens
                log(f"[MultiDeepSeekBatchManager] {key} 预留 token {tokens} 成功，累计 token 数 {self.handler_token_counts[key]}。")
                return True
            else:
                log(f"[MultiDeepSeekBatchManager] {key} 无法预留 token {tokens}（当前累计 {current}）。")
                return False

    def release_tokens(self, handler: DeepSeekBatchHandler, tokens: int):
        """
        释放预留的 token 数。
        """
        with self.lock:
            key = handler.name_tag()
            current = self.handler_token_counts.get(key, 0)
            new_value = max(0, current - tokens)
            self.handler_token_counts[key] = new_value
            log(f"[MultiDeepSeekBatchManager] {key} 释放 token {tokens}，累计 token 数更新为 {new_value}。")

    def _select_handler(self, job_token: int) -> Optional[DeepSeekBatchHandler]:
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
                if current_tokens + job_token <= 10000:
                    return handler
                count += 1
        return None

    def _approximate_token_count(self, file_path: str) -> int:
        """
        粗略估计文件中 token 数，默认按 4 个字符 1 token 计算。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return len(text) // 4

    def _poll_job(self, job: Dict[str, Any]) -> Optional[str]:
        """
        轮询单个批处理任务状态，直至任务完成或失败超限。
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
                        job["handler"].download_batch_results(output_file_id, job["output_file"])
                        log(f"[MultiDeepSeekBatchManager] 任务 {job['chunk_file']} 已完成，结果文件: {job['output_file']}")
                        self.release_tokens(job["handler"], job["token_count"])
                        self.api_key_failures[job["handler"].name_tag()] = 0
                        return job["output_file"]
                    else:
                        log(f"[MultiDeepSeekBatchManager] 任务 {job['chunk_file']} 状态为 completed，但未返回 output_file_id。")
                        fail_count += 1
                elif status in ("failed", "cancelled", "expired"):
                    log(f"[MultiDeepSeekBatchManager] 任务 {job['chunk_file']} 状态异常: {status}，准备重建批任务。")
                    fail_count += 1
                    current_key = job["handler"].name_tag()
                    self.api_key_failures[current_key] += 1
                    if len(self.handlers) == 1 or self.api_key_failures[current_key] < MAX_KEY_FAILURE_THRESHOLD:
                        log(f"[MultiDeepSeekBatchManager] 当前 APIKey {current_key} 失败次数 {self.api_key_failures[current_key]}，尝试重用。")
                        self.release_tokens(job["handler"], job["token_count"])
                        if not self.reserve_tokens(job["handler"], job["token_count"]):
                            log(f"[MultiDeepSeekBatchManager] 重新预留当前 APIKey {current_key} token 失败。")
                            return None
                    else:
                        log(f"[MultiDeepSeekBatchManager] 当前 APIKey {current_key} 失败次数达到阈值，尝试选择其他 APIKey。")
                        candidate = None
                        for _ in range(3):
                            cand = self._select_handler(job["token_count"])
                            if cand and cand.name_tag() != current_key:
                                if self.reserve_tokens(cand, job["token_count"]):
                                    candidate = cand
                                    break
                            time.sleep(self.poll_interval)
                        if candidate is None:
                            log(f"[MultiDeepSeekBatchManager] 未找到其他 APIKey，重用当前 APIKey {current_key}。")
                            self.release_tokens(job["handler"], job["token_count"])
                            if not self.reserve_tokens(job["handler"], job["token_count"]):
                                return None
                            candidate = job["handler"]
                        job["handler"] = candidate
                    try:
                        file_id = job["handler"].upload_batch_file(job["chunk_file"])
                        batch_info = job["handler"].create_batch(file_id)
                        job["batch_id"] = batch_info.get("id", "")
                        log(f"[MultiDeepSeekBatchManager] 任务 {job['chunk_file']} 重建批任务成功，新 batch_id: {job['batch_id']}")
                        self.api_key_failures[job["handler"].name_tag()] = 0
                    except Exception as e:
                        fail_count += 1
                        log(f"[MultiDeepSeekBatchManager] 任务 {job['chunk_file']} 重建批任务异常：{e}")
                        continue
                else:
                    log(f"[MultiDeepSeekBatchManager] 任务 {job['chunk_file']} 当前状态: {status}，继续轮询。")
                if fail_count >= self.max_failures_per_job or (current_time - start_time) > 3600:
                    log(f"[MultiDeepSeekBatchManager] 任务 {job['chunk_file']} 超过最大失败次数或超时，终止任务。")
                    self.release_tokens(job["handler"], job["token_count"])
                    return None
            except Exception as e:
                fail_count += 1
                log(f"[MultiDeepSeekBatchManager] 任务 {job['chunk_file']} 轮询异常：{e}，累计失败次数: {fail_count}")
                if fail_count >= self.max_failures_per_job:
                    self.release_tokens(job["handler"], job["token_count"])
                    return None
            time.sleep(self.poll_interval)

    def process_chunk_files(self, chunk_files: List[str], out_dir: str) -> List[str]:
        """
        对每个分块文件启动批处理任务，等待所有任务完成后返回所有输出文件的路径列表。
        """
        jobs = {}
        for cf in chunk_files:
            if cf in self.resume_map and os.path.isfile(self.resume_map[cf]):
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
        executor = ThreadPoolExecutor(max_workers=5)
        futures = {}
        results = {}
        def submit_job(job_key, job) -> bool:
            handler = self._select_handler(job["token_count"])
            if handler is None:
                log(f"[MultiDeepSeekBatchManager] 任务 {job_key} 无可用 API Key（token 限制），暂不提交。")
                return False
            if not self.reserve_tokens(handler, job["token_count"]):
                log(f"[MultiDeepSeekBatchManager] 任务 {job_key} 在 {handler.name_tag()} 预留 token 失败。")
                return False
            job["handler"] = handler
            try:
                file_id = handler.upload_batch_file(job["chunk_file"])
                batch_info = handler.create_batch(file_id)
                job["batch_id"] = batch_info.get("id", "")
                log(f"[MultiDeepSeekBatchManager] 任务 {job_key} 创建批任务成功，batch_id: {job['batch_id']}")
            except Exception as e:
                job["fail_count"] += 1
                log(f"[MultiDeepSeekBatchManager] 任务 {job_key} 创建批任务失败：{e}")
                self.release_tokens(handler, job["token_count"])
                return False
            futures[executor.submit(self._poll_job, job)] = job_key
            log(f"[MultiDeepSeekBatchManager] 提交任务 {job_key}，token 数 {job['token_count']}。")
            return True
        pending_keys = list(jobs.keys())
        idx = 0
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
        executor.shutdown(wait=True)
        for fut in futures:
            job_key = futures[fut]
            try:
                result = fut.result()
                results[job_key] = result if result else ""
            except Exception as e:
                log(f"[MultiDeepSeekBatchManager] 任务 {job_key} 异常：{e}")
                results[job_key] = ""
        output_files = [results.get(cf, jobs.get(cf, {}).get("output", "")) for cf in chunk_files]
        return output_files
