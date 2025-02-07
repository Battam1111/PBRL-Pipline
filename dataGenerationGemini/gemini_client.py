#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gemini_client.py

本模块封装了调用 Google 官方 Gemini API 的客户端管理，用于直连 Direct 模式。
主要功能包括：
  1. GeminiAPIClient：基于官方 Python SDK（google-generativeai）的封装，
     在每次调用时通过 APIKeyManager 动态获取 API Key 与项目 ID，
     并配置官方 SDK 后调用 Gemini API 生成回复。
  2. DirectGeminiManager：采用多线程并发处理任务请求，内置重试机制，
     每个任务从 JSONL 中提取 prompt，并调用 GeminiAPIClient 得到回复。

参考文档：
  - Gemini API 入门教程：https://ai.google.dev/gemini-api/docs/get-started/tutorial?hl=zh-cn
"""

import time
import threading
from queue import Queue, Empty
import json
import google.generativeai as genai

from config import GEMINI_MODEL, SYSTEM_PROMPT, MAX_TASK_RETRIES, API_CALL_TIMEOUT
from api_key_manager import APIKeyManager

class GeminiAPIClient:
    def __init__(self, model_name: str = GEMINI_MODEL, system_prompt: str = SYSTEM_PROMPT,
                 max_output_tokens: int = 2000, max_retries: int = MAX_TASK_RETRIES,
                 key_manager: APIKeyManager = None):
        """
        初始化 GeminiAPIClient。

        参数：
          model_name: Gemini 模型名称。
          system_prompt: 系统身份提示，用于构建对话历史。
          max_output_tokens: 最大输出 token 数量。
          max_retries: 最大重试次数。
          key_manager: APIKeyManager 对象，用于管理多个 API Key 和项目 ID；
                       如果未提供，则会使用配置中的默认（但建议总是提供）。
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        if key_manager is None:
            raise ValueError("必须提供 APIKeyManager 实例以支持多 API KEY 轮询")
        self.key_manager = key_manager

    def process(self, user_content: str) -> str:
        """
        通过 Gemini API 处理单个任务请求。

        参数：
          user_content: 用户输入文本（通常是经过任务模板格式化后的 prompt）。
        返回：
          Gemini API 生成的回复文本。
        """
        # 在每次调用时动态获取 API Key 和项目 ID
        api_key, project_id = self.key_manager.get_next()
        # 使用该 API Key 配置官方 SDK（全局配置，但在多线程中建议每个请求前重新配置）
        genai.configure(api_key=api_key)
        # 初始化模型实例，此处假设模型名称不依赖于项目 ID
        model = genai.GenerativeModel(self.model_name)
        # 创建一个空的对话会话（不重复传入 system_prompt，因其已在 prompt 中包含）
        chat = model.start_chat(history=[])
        # 调用 send_message 发起请求，user_content 已包含系统提示和 prompt 信息
        response = chat.send_message(user_content)
        return response.text

class DirectGeminiManager:
    def __init__(self, client: GeminiAPIClient, max_workers: int = 4):
        """
        初始化 DirectGeminiManager。

        参数：
          client: GeminiAPIClient 实例。
          max_workers: 最大并发工作线程数。
        """
        self.client = client
        self.max_workers = max_workers
        self.results = {}  # 存储各任务 custom_id 与返回结果
        self.task_queue = Queue()

    def add_tasks(self, tasks: list):
        """
        将任务列表加入内部队列。

        参数：
          tasks: 任务列表，每个任务为 dict 格式（包含 custom_id 与 body）。
        """
        for task in tasks:
            task.setdefault("retry_count", 0)
            self.task_queue.put(task)

    def worker(self):
        """
        工作线程：不断从队列中获取任务，调用 GeminiAPIClient 进行处理，
        并内置重试机制。
        """
        while True:
            try:
                task = self.task_queue.get(timeout=5)
            except Empty:
                break
            custom_id = task.get("custom_id")
            try:
                # 此处确保能正确获取任务中用户 prompt（注意索引为 0）
                user_content = task["body"]["contents"][0]["parts"][0]["text"]
            except Exception as e:
                print(f"[DirectGeminiManager] 任务 {custom_id} 解析 user_content 失败：{e}")
                user_content = ""
            for attempt in range(self.client.max_retries):
                try:
                    result_text = self.client.process(user_content)
                    self.results[custom_id] = {"text": result_text}
                    break
                except Exception as e:
                    if attempt == self.client.max_retries - 1:
                        self.results[custom_id] = {"text": "", "error": str(e)}
                    else:
                        time.sleep(2 ** attempt)
            self.task_queue.task_done()

    def process_tasks(self, tasks: list) -> dict:
        """
        并发处理所有任务，并返回结果字典。

        参数：
          tasks: 任务列表。
        返回：
          结果字典，键为 custom_id，值为 Gemini API 返回结果。
        """
        self.add_tasks(tasks)
        threads = []
        for _ in range(self.max_workers):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return self.results
