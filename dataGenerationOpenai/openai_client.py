#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
openai_client.py

本模块整合了直连（Direct）与批处理（Batch）两种 OpenAI API 调用管理，提供以下功能：
  1. DirectOpenAIHandler：直接调用 OpenAI ChatCompletion 接口，内置重试及余额检测机制
  2. DirectOpenAIManager：管理多个 DirectOpenAIHandler，通过多线程并发调用直连接口
  3. OpenAIBatchHandler：与单个 OpenAI Batch API Key 交互，包括上传任务文件、创建批任务、轮询状态及下载结果
  4. MultiOpenAIBatchManager：管理多个 OpenAIBatchHandler，对每个 JSONL 子文件创建 Job 对象，
     在整个生命周期中仅创建一次批任务，并持续轮询状态直至完成，避免重复创建批任务

所有类均提供详细中文注释，确保代码逻辑清晰、异常处理完备。
"""

import os
import time
import json
import threading
from queue import Queue
from typing import List, Dict, Any, Optional, Callable
import requests
from config import API_URL, MODEL, SYSTEM_PROMPT, BATCH_API_URL, FILES_API_URL, MAX_CONCURRENT_BATCHES, ENQUEUED_TOKEN_LIMIT

# ==============================================================================
# 1. 直连 OpenAI API 模块
# ==============================================================================

class InsufficientBalanceError(Exception):
    """
    自定义异常，表示当前 API Key 余额不足或配额耗尽
    """
    pass

class DirectOpenAIHandler:
    def __init__(self, api_key: str, key_index: int):
        """
        初始化 DirectOpenAIHandler 实例
        :param api_key: OpenAI API Key
        :param key_index: Key 编号（仅用于日志标识）
        """
        self.api_key = api_key
        self.key_index = key_index
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.disabled = False

    def name_tag(self) -> str:
        """返回用于日志标识的名称"""
        return f"DirectKey#{self.key_index}"

    def robust_request(self, func: Callable, max_retries: int = 99999, backoff: int = 3, *args, **kwargs):
        """
        通用重试机制：捕获 requests.exceptions.RequestException（包括 HTTPError、ConnectionError、ProtocolError 等）。
        遇到限流（429）或连接异常时采用指数退避，直到达到最大重试次数。
        :param func: 请求函数
        :param max_retries: 最大重试次数
        :param backoff: 初始退避秒数
        :return: 请求成功的返回结果
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                # 尝试获取响应状态码
                if hasattr(e, 'response') and e.response is not None:
                    status = e.response.status_code
                    if status == 429:
                        print(f"[{self.name_tag()}] 遇到限流（429），等待 {backoff}s（重试 {attempt+1} 次）")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    elif status == 402:
                        print(f"[{self.name_tag()}] 检测到 HTTP 402（余额不足）")
                        raise InsufficientBalanceError(str(e))
                    else:
                        try:
                            err_json = e.response.json()
                            if "error" in err_json:
                                msg = err_json["error"].get("message", "").lower()
                                if "insufficient_quota" in msg or "payment" in msg:
                                    print(f"[{self.name_tag()}] 检测到不足额度错误")
                                    raise InsufficientBalanceError(msg)
                        except Exception:
                            pass
                else:
                    print(f"[{self.name_tag()}] 请求异常：{e}，等待 {backoff}s 后重试（重试 {attempt+1} 次）")
                time.sleep(backoff)
                backoff *= 2
                continue
            except Exception as e:
                print(f"[{self.name_tag()}] 请求失败（非 RequestException）：{e}")
                raise
        raise RuntimeError(f"[{self.name_tag()}] 超过最大重试次数，依然失败。")

    def call_openai_api(self, payload: dict) -> dict:
        """
        调用 OpenAI /v1/chat/completions 接口，返回响应 JSON
        :param payload: 请求负载
        :return: API 响应 JSON
        """
        def do_call():
            if self.disabled:
                raise InsufficientBalanceError(f"[{self.name_tag()}] Key 已禁用。")
            resp = requests.post(API_URL, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()
        return self.robust_request(do_call)

    def process_pair(self, user_content: str, max_tokens: int = 2000) -> dict:
        """
        发起一次 ChatCompletion 请求，返回响应结果
        :param user_content: 用户输入内容
        :param max_tokens: 最大生成 token 数量
        :return: API 响应结果 JSON
        """
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": max_tokens
        }
        return self.call_openai_api(payload)

class DirectOpenAIManager:
    def __init__(self, max_workers_per_key: int = 2):
        """
        初始化 DirectOpenAIManager 实例，用于并行管理多个 DirectOpenAIHandler
        :param max_workers_per_key: 每个 API Key 可并发线程数
        """
        self.max_workers_per_key = max_workers_per_key
        self.handlers: List[DirectOpenAIHandler] = []
        self.task_queue = Queue()
        self.results: Dict[str, dict] = {}

    def load_handlers(self, api_keys: List[str]):
        """
        加载多个 API Key，生成对应的 DirectOpenAIHandler 实例
        :param api_keys: API Key 列表
        """
        self.handlers = [DirectOpenAIHandler(key, idx+1) for idx, key in enumerate(api_keys)]
        print(f"[DirectManager] 加载了 {len(self.handlers)} 个直连处理器。")

    def add_tasks(self, tasks: List[Dict]):
        """
        将任务列表加入内部任务队列，每个任务需包含 custom_id 与 body 信息
        :param tasks: 任务列表
        """
        for task in tasks:
            self.task_queue.put(task)

    def worker(self, handler: DirectOpenAIHandler):
        """
        工作线程：不断从任务队列中取任务，通过 handler 调用 API，遇异常则重试
        :param handler: 当前使用的 DirectOpenAIHandler
        """
        while not self.task_queue.empty() and not handler.disabled:
            try:
                task = self.task_queue.get_nowait()
            except Exception:
                break
            custom_id = task.get("custom_id")
            user_content = task["body"]["messages"][-1]["content"]
            max_tokens = task["body"].get("max_tokens", 2000)
            try:
                response = handler.process_pair(user_content, max_tokens)
                self.results[custom_id] = response
                print(f"[{handler.name_tag()}] 成功处理任务 {custom_id}")
            except InsufficientBalanceError as ibe:
                handler.disabled = True
                print(f"[{handler.name_tag()}] 余额不足，停用该 Key：{ibe}")
                self.task_queue.put(task)
                break
            except Exception as e:
                print(f"[{handler.name_tag()}] 处理任务 {custom_id} 异常：{e}，稍后重试。")
                self.task_queue.put(task)
                time.sleep(5)
            finally:
                self.task_queue.task_done()

    def process_tasks(self, tasks: List[Dict], resume_file: str = None) -> Dict[str, dict]:
        """
        并行处理所有任务，支持从 resume_file 加载已完成任务避免重复处理
        :param tasks: 任务列表
        :param resume_file: 已完成任务记录文件路径（可选）
        :return: 处理结果字典：custom_id -> API 响应 JSON
        """
        skip_ids = set()
        if resume_file and os.path.isfile(resume_file):
            with open(resume_file, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        obj = json.loads(line)
                        cid = obj.get("custom_id")
                        if cid:
                            self.results[cid] = obj.get("response", {})
                            skip_ids.add(cid)
                    except Exception:
                        pass
            print(f"[DirectManager] 从 {resume_file} 加载 {len(skip_ids)} 条已完成任务。")
        pending_tasks = [t for t in tasks if t.get("custom_id") not in skip_ids]
        print(f"[DirectManager] 总任务数：{len(tasks)}，跳过：{len(skip_ids)}，待处理：{len(pending_tasks)}")
        self.add_tasks(pending_tasks)
        threads = []
        for handler in self.handlers:
            for _ in range(self.max_workers_per_key):
                t = threading.Thread(target=self.worker, args=(handler,))
                t.start()
                threads.append(t)
        for t in threads:
            t.join()
        print("[DirectManager] 所有任务处理完成。")
        return self.results

# ==============================================================================
# 2. 批处理 OpenAI API 模块
# ==============================================================================

class OpenAIBatchHandler:
    def __init__(self, api_key: str, key_index: int):
        """
        初始化 OpenAIBatchHandler 实例
        :param api_key: OpenAI API Key
        :param key_index: Key 编号，用于日志标识
        """
        self.api_key = api_key
        self.key_index = key_index
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.disabled = False

    def name_tag(self) -> str:
        return f"BatchKey#{self.key_index}"

    def robust_request(self, func: Callable, *args, max_retries: int = 99999, **kwargs):
        """
        通用请求重试函数：捕获 requests.exceptions.RequestException（包括连接重置等错误），
        遇到限流（429）时采用指数退避，直到达到最大重试次数。
        :param func: 请求函数
        :return: 请求返回结果
        """
        backoff = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                if hasattr(e, 'response') and e.response is not None:
                    status = e.response.status_code
                    if status == 429:
                        print(f"[{self.name_tag()}] 遇到限流（429），等待 {backoff}s（重试 {attempt+1} 次）")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    elif status == 402:
                        print(f"[{self.name_tag()}] 检测到 HTTP 402（余额不足）")
                        raise InsufficientBalanceError(str(e))
                    else:
                        try:
                            err_json = e.response.json()
                            if "error" in err_json:
                                msg = err_json["error"].get("message", "").lower()
                                if "insufficient_quota" in msg or "payment" in msg:
                                    print(f"[{self.name_tag()}] 检测到不足额度错误")
                                    raise InsufficientBalanceError(msg)
                        except Exception:
                            pass
                else:
                    print(f"[{self.name_tag()}] 请求异常：{e}，等待 {backoff}s 后重试（重试 {attempt+1} 次）")
                time.sleep(backoff)
                backoff *= 2
                continue
            except Exception as e:
                print(f"[{self.name_tag()}] 请求失败（非 RequestException）：{e}")
                raise
        raise RuntimeError(f"[{self.name_tag()}] 超过最大重试次数，操作失败。")

    def upload_batch_file(self, file_path: str) -> str:
        """
        上传 .jsonl 文件到 /v1/files 接口，返回 file_id
        :param file_path: 本地 JSONL 文件路径
        :return: 上传后返回的 file_id
        """
        if self.disabled:
            raise InsufficientBalanceError(f"[{self.name_tag()}] Key 已被禁用。")
        if not file_path.endswith(".jsonl"):
            raise ValueError(f"[{self.name_tag()}] 仅支持上传 .jsonl 文件：{file_path}")

        while True:
            print(f"[{self.name_tag()}] 上传文件：{file_path}")
            try:
                def do_upload():
                    if self.disabled:
                        raise InsufficientBalanceError(f"[{self.name_tag()}] Key 已禁用。")
                    with open(file_path, "rb") as f:
                        r = requests.post(
                            FILES_API_URL,
                            headers=self.headers,
                            files={"file": (file_path, f, "application/json")},
                            data={"purpose": "batch"}
                        )
                    r.raise_for_status()
                    return r.json()["id"]
                file_id = self.robust_request(do_upload)
                print(f"[{self.name_tag()}] 上传成功，file_id={file_id}")
                return file_id
            except InsufficientBalanceError:
                raise
            except Exception as e:
                print(f"[{self.name_tag()}] 上传异常：{e}，30秒后重试。")
                time.sleep(30)

    def create_batch(self, file_id: str) -> dict:
        """
        基于上传的 file_id 创建批任务，返回任务信息
        :param file_id: 上传文件的 file_id
        :return: 批任务详细信息字典
        """
        if self.disabled:
            raise InsufficientBalanceError(f"[{self.name_tag()}] Key 已禁用。")
        while True:
            print(f"[{self.name_tag()}] 创建批任务，file_id={file_id}")
            try:
                def do_create():
                    payload = {
                        "input_file_id": file_id,
                        "endpoint": "/v1/chat/completions",
                        "completion_window": "24h"
                    }
                    r = requests.post(BATCH_API_URL, headers=self.headers, json=payload)
                    r.raise_for_status()
                    return r.json()
                batch_info = self.robust_request(do_create)
                print(f"[{self.name_tag()}] 批任务创建成功，batch_id={batch_info.get('id')}")
                return batch_info
            except InsufficientBalanceError:
                raise
            except Exception as ex:
                print(f"[{self.name_tag()}] create_batch 异常：{ex}，30秒后重试。")
                time.sleep(30)

    def check_batch_status(self, batch_id: str) -> dict:
        """
        查询批任务状态，返回状态信息字典
        :param batch_id: 批任务 ID
        :return: 状态信息字典
        """
        if self.disabled:
            raise InsufficientBalanceError(f"[{self.name_tag()}] Key 已禁用。")
        while True:
            try:
                def do_check():
                    url = f"{BATCH_API_URL}/{batch_id}"
                    r = requests.get(url, headers=self.headers)
                    r.raise_for_status()
                    return r.json()
                return self.robust_request(do_check)
            except InsufficientBalanceError:
                raise
            except Exception as e:
                print(f"[{self.name_tag()}] check_batch_status 异常：{e}，30秒后重试。")
                time.sleep(30)

    def list_batches(self) -> List[dict]:
        """
        查询当前 Key 下的所有批任务，用于监控队列占用
        :return: 批任务列表
        """
        if self.disabled:
            raise InsufficientBalanceError(f"[{self.name_tag()}] Key 已禁用。")
        while True:
            try:
                def do_list():
                    r = requests.get(BATCH_API_URL, headers=self.headers)
                    r.raise_for_status()
                    return r.json()["data"]
                return self.robust_request(do_list)
            except InsufficientBalanceError:
                raise
            except Exception as e:
                print(f"[{self.name_tag()}] list_batches 异常：{e}，30秒后重试。")
                time.sleep(30)

    def download_batch_results(self, output_file_id: str, save_path: str):
        """
        下载批任务结果文件到本地指定路径
        :param output_file_id: 结果文件的 file_id
        :param save_path: 保存结果文件的本地路径
        """
        if self.disabled:
            raise InsufficientBalanceError(f"[{self.name_tag()}] Key 已禁用。")
        while True:
            try:
                print(f"[{self.name_tag()}] 正在下载结果文件（output_file_id={output_file_id}）到 {save_path}")
                def do_download():
                    url = f"{FILES_API_URL}/{output_file_id}/content"
                    r = requests.get(url, headers=self.headers)
                    r.raise_for_status()
                    return r.text
                content = self.robust_request(do_download)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"[{self.name_tag()}] 下载成功，文件保存在 {save_path}")
                return
            except InsufficientBalanceError:
                raise
            except Exception as e:
                print(f"[{self.name_tag()}] 下载异常：{e}，30秒后重试。")
                time.sleep(30)

    def wait_for_queue_space(self):
        """
        检查当前 Key 下活跃批任务及 token 使用情况，若超出限制则等待
        """
        if self.disabled:
            raise InsufficientBalanceError(f"[{self.name_tag()}] Key 已禁用。")
        while True:
            batches = self.list_batches()
            active_batches = [b for b in batches if b.get("status") in ("queued", "running")]
            total_tokens = sum(b.get("enqueued_tokens", 0) for b in active_batches)
            print(f"[{self.name_tag()}] 当前活跃批任务数：{len(active_batches)}，token 总数：{total_tokens}")
            if len(active_batches) >= MAX_CONCURRENT_BATCHES or total_tokens >= ENQUEUED_TOKEN_LIMIT:
                print(f"[{self.name_tag()}] 队列资源紧张，等待 30秒……")
                time.sleep(30)
            else:
                break

# ------------------------------------------------------------------------------
# MultiOpenAIBatchManager（重构版）
# ------------------------------------------------------------------------------
class MultiOpenAIBatchManager:
    """
    MultiOpenAIBatchManager 用于多 Key 并行处理多个 JSONL 子文件（chunk），
    针对每个 chunk 文件创建一个 Job 对象，在整个生命周期内仅创建一次批任务，
    并持续轮询任务状态直至完成，避免重复创建批任务。
    
    设计思路：
      1. 对每个 chunk 文件构造 Job 对象，记录文件路径、输出路径、分配的 handler、批任务 ID、状态及重试次数；
      2. Job 状态包括： "pending"（待处理）、"running"（任务运行中）、"completed"（任务完成）及 "failed"（任务失败）；
      3. 主流程不断检查各 Job 状态，对 "pending" 状态分配 handler 并创建批任务，对 "running" 状态轮询状态，
         若批任务完成则下载结果；如状态异常则进行重试，直至达到最大重试次数后标记为失败；
      4. 最终返回字典，键为 chunk 文件路径，值为对应输出结果文件路径。
    """
    def __init__(self,
                 max_tasks_per_key: int = 2,
                 max_failures_per_job: int = 99999,
                 poll_interval: int = 10,
                 resume_map: Optional[Dict[str, str]] = None):
        """
        初始化 MultiOpenAIBatchManager 实例
        :param max_tasks_per_key: 每个 Key 同时处理任务数
        :param max_failures_per_job: 单个 Job 允许的最大重试次数
        :param poll_interval: 轮询间隔（秒）
        :param resume_map: 已完成 chunk 文件与输出文件的映射（用于断点续跑）
        """
        self.max_tasks_per_key = max_tasks_per_key
        self.max_failures_per_job = max_failures_per_job
        self.poll_interval = poll_interval
        self.resume_map = resume_map if resume_map else {}
        
        self.jobs: Dict[str, Dict[str, Any]] = {}         # 存储所有 Job 对象，键为 chunk 文件路径
        self.results_map: Dict[str, str] = {}               # 最终结果映射：chunk 文件 -> 输出文件路径
        self.handlers: List[OpenAIBatchHandler] = []        # 存储使用的 handler 列表

    def load_handlers(self, api_keys: List[str]):
        """
        加载多个 API Key，生成对应的 OpenAIBatchHandler 实例
        :param api_keys: API Key 列表
        """
        from openai_client import OpenAIBatchHandler  # 避免循环依赖
        self.handlers = [OpenAIBatchHandler(key, idx+1) for idx, key in enumerate(api_keys)]
        print(f"[MultiBatchManager] 加载了 {len(self.handlers)} 个 Key。")

    def _create_job(self, chunk_file: str, out_dir: str) -> Dict[str, Any]:
        """
        根据 chunk 文件构造 Job 对象
        :param chunk_file: JSONL 子文件路径
        :param out_dir: 输出目录，用于生成输出文件路径
        :return: Job 对象，包含 chunk_file、output_file、status、fail_count、handler、batch_id 等信息
        """
        base = os.path.splitext(os.path.basename(chunk_file))[0]
        out_file = os.path.join(out_dir, f"{base}_output.jsonl")
        job = {
            "chunk_file": chunk_file,
            "output_file": out_file,
            "status": "pending",
            "fail_count": 0,
            "handler": None,
            "batch_id": ""
        }
        return job

    def process_chunk_files_official(self, chunk_files: List[str], out_dir: str) -> List[str]:
        """
        官方模式下处理所有 chunk 文件，直到全部成功（除非所有 Key 均不可用）
        :param chunk_files: JSONL 子文件列表
        :param out_dir: 输出目录
        :return: 输出结果文件路径列表，顺序与输入 chunk_files 一致
        """
        for cf in chunk_files:
            if cf in self.resume_map:
                self.results_map[cf] = self.resume_map[cf]
            else:
                self.jobs[cf] = self._create_job(cf, out_dir)
        
        while any(job["status"] != "completed" for job in self.jobs.values()):
            for cf, job in self.jobs.items():
                if job["status"] == "completed":
                    continue
                if job["status"] == "pending":
                    available = [h for h in self.handlers if not h.disabled]
                    if not available:
                        print("[MultiBatchManager] 无可用 Key，等待 10 秒重试……")
                        time.sleep(10)
                        continue
                    job["handler"] = available[0]
                    try:
                        fid = job["handler"].upload_batch_file(job["chunk_file"])
                        batch_info = job["handler"].create_batch(fid)
                        job["batch_id"] = batch_info.get("id", "")
                        if not job["batch_id"]:
                            raise RuntimeError("create_batch 返回空 batch_id")
                        job["status"] = "running"
                        print(f"[MultiBatchManager] 对文件 {os.path.basename(cf)} 创建批任务成功，batch_id={job['batch_id']}")
                    except Exception as e:
                        job["fail_count"] += 1
                        print(f"[MultiBatchManager] 对文件 {os.path.basename(cf)} 创建批任务失败，重试次数：{job['fail_count']}，错误：{e}")
                        if job["fail_count"] >= self.max_failures_per_job:
                            job["status"] = "failed"
                        else:
                            time.sleep(10)
                        continue
                if job["status"] == "running":
                    try:
                        status_info = job["handler"].check_batch_status(job["batch_id"])
                        st = status_info.get("status", "")
                        if st == "completed":
                            output_file_id = status_info.get("output_file_id", "")
                            if not output_file_id:
                                raise RuntimeError("completed 状态但无 output_file_id")
                            job["handler"].download_batch_results(output_file_id, job["output_file"])
                            job["status"] = "completed"
                            self.results_map[job["chunk_file"]] = job["output_file"]
                            print(f"[MultiBatchManager] 文件 {os.path.basename(cf)} 批任务已完成，结果保存至 {job['output_file']}")
                        elif st in ("queued", "running", "validating", "in_progress", "finalizing", "cancelling"):
                            print(f"[MultiBatchManager] 文件 {os.path.basename(cf)} 当前状态：{st}，继续等待。")
                        elif st in ("failed", "cancelled", "expired"):
                            job["fail_count"] += 1
                            print(f"[MultiBatchManager] 文件 {os.path.basename(cf)} 批任务状态异常（{st}），累计重试次数：{job['fail_count']}")
                            if job["fail_count"] >= self.max_failures_per_job:
                                job["status"] = "failed"
                                print(f"[MultiBatchManager] 文件 {os.path.basename(cf)} 重试次数过多，标记为失败。")
                            else:
                                time.sleep(10)
                                fid = job["handler"].upload_batch_file(job["chunk_file"])
                                batch_info = job["handler"].create_batch(fid)
                                job["batch_id"] = batch_info.get("id", "")
                                print(f"[MultiBatchManager] 文件 {os.path.basename(cf)} 重新创建批任务成功，新的 batch_id={job['batch_id']}")
                        else:
                            print(f"[MultiBatchManager] 文件 {os.path.basename(cf)} 返回未知状态：{st}，等待重试。")
                    except Exception as e:
                        job["fail_count"] += 1
                        print(f"[MultiBatchManager] 文件 {os.path.basename(cf)} 轮询异常：{e}，累计重试次数：{job['fail_count']}")
                        if job["fail_count"] >= self.max_failures_per_job:
                            job["status"] = "failed"
                    time.sleep(self.poll_interval)
            # 循环结束前再检查所有 Job 状态
        print("[MultiBatchManager] 所有 chunk 文件均处理完成。")
        return [self.results_map.get(cf, "") for cf in chunk_files]
