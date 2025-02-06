#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gemini_client.py

本模块封装了调用 Google Gemini API 的客户端管理，包含以下部分：
  1. DirectGeminiHandler：直接调用 Gemini API 接口，内置重试机制和异常处理。
  2. DirectGeminiManager：管理多个 DirectGeminiHandler，通过多线程并发调用 Gemini API。
  3. MultiGeminiBatchManager：模拟 Batch 模式，针对每个 JSONL 分块创建任务并并发处理所有任务。

所有 API 调用均采用 HTTP POST 请求，数据格式符合 Gemini API 要求，
请求体中使用 "contents" 数组描述对话，其中每个元素包含 "role" 与 "parts" 字段。

参考 Google 官方 Gemini API 文档：
  - 入门教程: https://ai.google.dev/gemini-api/docs/get-started/tutorial?hl=zh-cn :contentReference[oaicite:5]{index=5}
  - 模型参考: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference?hl=zh-cn :contentReference[oaicite:6]{index=6}
"""

import os
import time
import json
import threading
from queue import Queue
from typing import List, Dict, Any, Optional, Callable
import requests
from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_API_URL_TEMPLATE

class InsufficientBalanceError(Exception):
    """
    自定义异常，表示当前 API Key 配额或余额不足。
    """
    pass

class DirectGeminiHandler:
    def __init__(self, api_key: str, key_index: int):
        """
        初始化 DirectGeminiHandler 实例。

        参数：
          api_key: Gemini API 密钥。
          key_index: 用于日志标识的 Key 编号。
        """
        self.api_key = api_key
        self.key_index = key_index
        self.headers = {
            "Content-Type": "application/json"
        }
        self.disabled = False

    def name_tag(self) -> str:
        return f"DirectGeminiKey#{self.key_index}"

    def robust_request(self, func: Callable, max_retries: int = 99999, backoff: int = 3, *args, **kwargs):
        """
        通用请求重试机制：捕获 requests 异常，采用指数退避重试。

        参数：
          func: 要执行的请求函数。
          max_retries: 最大重试次数。
          backoff: 初始退避秒数。
        """
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
                        print(f"[{self.name_tag()}] HTTP 402（余额不足）")
                        raise InsufficientBalanceError(str(e))
                else:
                    print(f"[{self.name_tag()}] 请求异常：{e}，等待 {backoff}s 后重试（重试 {attempt+1} 次）")
                time.sleep(backoff)
                backoff *= 2
                continue
            except Exception as e:
                print(f"[{self.name_tag()}] 请求失败（非 RequestException）：{e}")
                raise
        raise RuntimeError(f"[{self.name_tag()}] 超过最大重试次数，操作失败。")

    def call_gemini_api(self, payload: dict) -> dict:
        """
        调用 Gemini API，返回响应 JSON。
        根据配置构造请求 URL，并附加 API Key 和模型名称。

        参数：
          payload: 请求体，符合 Gemini API 要求。
        返回：
          API 响应的 JSON 数据。
        """
        url = GEMINI_API_URL_TEMPLATE.format(model=GEMINI_MODEL, api_key=self.api_key)
        def do_call():
            if self.disabled:
                raise InsufficientBalanceError(f"[{self.name_tag()}] Key 已禁用。")
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        return self.robust_request(do_call)

    def process_pair(self, user_content: str, max_tokens: int = 2000) -> dict:
        """
        发起一次 Gemini API 请求，返回生成结果。
        请求体中包含 system 和 user 两部分内容，采用 Gemini API 所需的 "contents" 格式。

        参数：
          user_content: 用户输入文本。
          max_tokens: 最大输出 token 数量。
        返回：
          包含生成文本的字典。
        """
        payload = {
            "model": GEMINI_MODEL,
            "contents": [
                {"role": "system", "parts": [{"text": ""}]},  # 若需要固定提示，可在此填入；此处由 JSONL 已构造
                {"role": "user", "parts": [{"text": user_content}]}
            ],
            "maxOutputTokens": max_tokens
        }
        result = self.call_gemini_api(payload)
        try:
            candidate = result.get("candidates", [])[0]
            content_parts = candidate.get("content", {}).get("parts", [])
            text = content_parts[0].get("text", "")
        except Exception as e:
            text = ""
            print(f"[{self.name_tag()}] 解析响应失败：{e}")
        return {"text": text}

class DirectGeminiManager:
    def __init__(self, max_workers_per_key: int = 2):
        """
        初始化 DirectGeminiManager，用于并行管理多个 DirectGeminiHandler。

        参数：
          max_workers_per_key: 每个 API Key 可并发的线程数。
        """
        self.max_workers_per_key = max_workers_per_key
        self.handlers: List[DirectGeminiHandler] = []
        self.task_queue = Queue()
        self.results: Dict[str, dict] = {}

    def load_handlers(self, api_keys: List[str]):
        """
        根据 API Key 列表创建 DirectGeminiHandler 实例。

        参数：
          api_keys: API Key 列表。
        """
        self.handlers = [DirectGeminiHandler(key, idx+1) for idx, key in enumerate(api_keys)]
        print(f"[DirectGeminiManager] 加载了 {len(self.handlers)} 个直连处理器。")

    def add_tasks(self, tasks: List[Dict]):
        """
        将任务列表加入内部队列，每个任务必须包含 custom_id 和 body 信息。

        参数：
          tasks: 任务列表。
        """
        for task in tasks:
            self.task_queue.put(task)

    def worker(self, handler: DirectGeminiHandler):
        """
        工作线程：不断从队列中获取任务，并通过指定 handler 调用 Gemini API。

        参数：
          handler: DirectGeminiHandler 实例。
        """
        while not self.task_queue.empty() and not handler.disabled:
            try:
                task = self.task_queue.get_nowait()
            except Exception:
                break
            custom_id = task.get("custom_id")
            try:
                user_content = task["body"]["contents"][1]["parts"][0]["text"]
            except Exception:
                user_content = ""
            max_tokens = task["body"].get("maxOutputTokens", 2000)
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
        并行处理所有任务，支持加载 resume_file 中已完成的任务以避免重复处理。

        参数：
          tasks: 任务列表。
          resume_file: 已完成任务记录文件路径（可选）。
        返回：
          处理结果字典，键为 custom_id，值为 API 响应结果。
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
            print(f"[DirectGeminiManager] 从 {resume_file} 加载 {len(skip_ids)} 条已完成任务。")
        pending_tasks = [t for t in tasks if t.get("custom_id") not in skip_ids]
        print(f"[DirectGeminiManager] 总任务数：{len(tasks)}，跳过：{len(skip_ids)}，待处理：{len(pending_tasks)}")
        self.add_tasks(pending_tasks)
        threads = []
        for handler in self.handlers:
            for _ in range(self.max_workers_per_key):
                t = threading.Thread(target=self.worker, args=(handler,))
                t.start()
                threads.append(t)
        for t in threads:
            t.join()
        print("[DirectGeminiManager] 所有任务处理完成。")
        return self.results

class MultiGeminiBatchManager:
    """
    模拟 Batch 模式处理 Gemini 任务。
    由于 Gemini API 本身为同步调用，此处采用多线程并发处理 JSONL 分块中的任务，
    并统一等待所有任务完成后返回结果。
    """
    def __init__(self,
                 max_tasks_per_key: int = 2,
                 poll_interval: int = 5,
                 resume_map: Optional[Dict[str, str]] = None):
        self.max_tasks_per_key = max_tasks_per_key
        self.poll_interval = poll_interval
        self.resume_map = resume_map if resume_map else {}
        self.results_map: Dict[str, str] = {}
        self.handlers: List[DirectGeminiHandler] = []

    def load_handlers(self, api_keys: List[str]):
        """
        根据 API Key 列表加载 DirectGeminiHandler 实例。

        参数：
          api_keys: API Key 列表。
        """
        self.handlers = [DirectGeminiHandler(key, idx+1) for idx, key in enumerate(api_keys)]
        print(f"[MultiGeminiBatchManager] 加载了 {len(self.handlers)} 个 Key。")

    def _process_chunk(self, chunk_file: str, out_dir: str):
        """
        处理单个 JSONL 分块文件中的任务，调用 Gemini API 并将结果写入输出文件。

        参数：
          chunk_file: JSONL 分块文件路径。
          out_dir: 输出目录。
        """
        print(f"[MultiGeminiBatchManager] 正在处理文件 {os.path.basename(chunk_file)}")
        with open(chunk_file, "r", encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        direct_manager = DirectGeminiManager(max_workers_per_key=self.max_tasks_per_key)
        # 使用所有可用的 handler（已从本实例中加载）
        direct_manager.load_handlers([handler.api_key for handler in self.handlers if not handler.disabled])
        results = direct_manager.process_tasks(tasks)
        output_file = os.path.splitext(chunk_file)[0] + "_output.jsonl"
        with open(output_file, "w", encoding="utf-8") as wf:
            for cid, resp in results.items():
                wf.write(json.dumps({"custom_id": cid, "response": resp}, ensure_ascii=False) + "\n")
        self.results_map[chunk_file] = output_file
        print(f"[MultiGeminiBatchManager] 文件 {os.path.basename(chunk_file)} 处理完成，输出 {output_file}")

    def process_chunk_files_official(self, chunk_files: List[str], out_dir: str) -> List[str]:
        """
        并发处理所有 JSONL 分块文件，并返回每个分块对应的输出文件路径。

        参数：
          chunk_files: JSONL 分块文件列表。
          out_dir: 输出目录。
        返回：
          输出结果文件路径列表，顺序与输入文件一致。
        """
        threads = []
        for cf in chunk_files:
            if cf in self.resume_map:
                self.results_map[cf] = self.resume_map[cf]
            else:
                t = threading.Thread(target=self._process_chunk, args=(cf, out_dir))
                t.start()
                threads.append(t)
        # 等待所有线程完成
        for t in threads:
            t.join()
        print("[MultiGeminiBatchManager] 所有分块文件处理完成。")
        return [self.results_map.get(cf, "") for cf in chunk_files]
