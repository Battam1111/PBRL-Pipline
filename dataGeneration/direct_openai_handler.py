# /home/star/Yanjun/RL-VLM-F/dataGeneration/direct_openai_handler.py
# -------------------------------------------------------------------------------
"""
direct_openai_handler.py

定义 DirectOpenAIHandler 类，用于直接调用 OpenAI API，
而非使用 Batch API。提供上传任务、检查结果、下载结果的直接实现。
"""

import time
import requests
from config import API_URL, MODEL, SYSTEM_PROMPT

class DirectOpenAIHandler:
    """
    直接调用 OpenAI API 的处理器，不使用 Batch API。
    实现与 OpenAIBatchHandler 类似的接口，但通过同步请求处理单个任务。
    """

    def __init__(self, api_key: str, key_index: int):
        self.api_key = api_key
        self.key_index = key_index
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def name_tag(self) -> str:
        return f"DirectKey#{self.key_index}"

    def robust_request(self, func, max_retries=3, backoff=3, *args, **kwargs):
        """
        通用重试机制。针对 429 做指数退避，其它直接抛出或在重试上限后抛出。
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print(f"[{self.name_tag()}] 429 限流, 等待 {backoff}s (尝试 {attempt+1})")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"[{self.name_tag()}] 请求失败: {e}")
                    raise
        raise RuntimeError(f"[{self.name_tag()}] 多次重试后仍失败")

    def call_openai_api(self, payload: dict) -> dict:
        """
        直接调用 OpenAI API，返回响应 JSON
        """
        def do_call():
            response = requests.post(API_URL, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        return self.robust_request(do_call)

    def process_pair(self, user_content: str, max_tokens: int = 2000) -> dict:
        """
        处理单个对比任务的请求，返回 OpenAI API 响应结果。
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
