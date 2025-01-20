# /home/star/Yanjun/RL-VLM-F/dataGeneration/openai_batch.py
# -------------------------------------------------------------------------------
"""
openai_batch.py

定义 OpenAIBatchHandler 类, 与单个 OpenAI API Key 交互:
 - upload_batch_file -> file_id
 - create_batch -> batch_id
 - check_batch_status
 - download_batch_results
 - wait_for_queue_space
 不允许跳过任何任务，出现错误时无限重试.
"""

import time
import requests
from typing import Dict, List
from config import (
    FILES_API_URL, BATCH_API_URL,
    MAX_CONCURRENT_BATCHES, ENQUEUED_TOKEN_LIMIT
)

class OpenAIBatchHandler:
    """
    与 '单个' OpenAI API Key 的 Batch API 交互.
    负责:
     1. upload_batch_file -> file_id
     2. create_batch -> batch_id
     3. check_batch_status
     4. download_batch_results
     5. wait_for_queue_space
    在日志中用 "Key#?" 的形式显示, 避免泄漏真正的 Key.
    """

    def __init__(self, api_key: str, key_index: int):
        """
        :param api_key: 该 handler 使用的OpenAI API Key
        :param key_index: 在多Key列表中的序号, 用于日志识别
        """
        self.api_key = api_key
        self.key_index = key_index
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def name_tag(self) -> str:
        """返回用于日志标识的 Key 名称(不含敏感信息)."""
        return f"Key#{self.key_index}"

    def robust_request(self, func, *args, max_retries=3, **kwargs):
        """
        通用重试装置：对 429 做指数退避，其他错误直接抛。
        """
        backoff = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print(f"[{self.name_tag()}] 429, 等待 {backoff}s (尝试 {attempt+1})")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"[{self.name_tag()}] 请求失败: {e}")
                    raise
        raise RuntimeError(f"[{self.name_tag()}] 多次重试后仍失败")

    def upload_batch_file(self, file_path: str) -> str:
        """
        上传 .jsonl 文件到 /v1/files, 返回 file_id.
        如果网络失败, 不允许跳过, 将无限重试.
        """
        if not file_path.endswith(".jsonl"):
            raise ValueError(f"只能上传 .jsonl: {file_path}")

        while True:
            print(f"[{self.name_tag()}] [上传] {file_path}")
            try:
                def do_upload():
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
                print(f"[{self.name_tag()}] [上传成功] file_id={file_id}")
                return file_id
            except Exception as e:
                print(f"[{self.name_tag()}] 上传异常: {e}, 等待30s后重试.")
                time.sleep(30)
                continue

    def create_batch(self, file_id: str) -> Dict[str, any]:
        """
        基于 file_id 创建 batch. 
        若 token_limit_exceeded => 无限重试; 其他错误 => 也无限重试.
        """
        while True:
            print(f"[{self.name_tag()}] [Batch] create_batch for {file_id}")
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
                print(f"[{self.name_tag()}] [Batch] 创建成功, batch_id={batch_info.get('id')}")
                return batch_info
            except requests.exceptions.HTTPError as ex:
                can_retry = False
                if ex.response is not None:
                    try:
                        err_json = ex.response.json()
                        # 这里可根据 API 返回结构判断是否 token_limit_exceeded
                        if "errors" in err_json:
                            for sub in err_json["errors"].get("data", []):
                                if sub.get("code") == "token_limit_exceeded":
                                    print(f"[{self.name_tag()}] token_limit_exceeded, 等待30s再重试.")
                                    can_retry = True
                    except:
                        pass
                if can_retry:
                    time.sleep(30)
                    continue
                else:
                    print(f"[{self.name_tag()}] create_batch失败: {ex}, 继续重试30s后.")
                    time.sleep(30)
                    continue
            except Exception as ex2:
                print(f"[{self.name_tag()}] create_batch异常: {ex2}, 等待30s再试.")
                time.sleep(30)
                continue

    def check_batch_status(self, batch_id: str) -> Dict[str, any]:
        """
        查询指定 batch_id 的状态. 遇到网络错误则无限重试.
        """
        while True:
            try:
                def do_check():
                    r = requests.get(f"{BATCH_API_URL}/{batch_id}", headers=self.headers)
                    r.raise_for_status()
                    return r.json()
                return self.robust_request(do_check)
            except Exception as e:
                print(f"[{self.name_tag()}] check_batch_status异常: {e}, 等待30s后重试.")
                time.sleep(30)
                continue

    def list_batches(self) -> List[Dict[str, any]]:
        """
        拉取该 Key 下所有 batches 的信息. 遇到错误则无限重试.
        """
        while True:
            try:
                def do_list():
                    r = requests.get(BATCH_API_URL, headers=self.headers)
                    r.raise_for_status()
                    return r.json()["data"]
                return self.robust_request(do_list)
            except Exception as e:
                print(f"[{self.name_tag()}] list_batches异常: {e}, 等待30s后重试.")
                time.sleep(30)
                continue

    def download_batch_results(self, output_file_id: str, save_path: str):
        """
        下载结果到本地. 遇到错误则无限重试, 不允许放弃.
        """
        while True:
            try:
                print(f"[{self.name_tag()}] [下载] output_file_id={output_file_id} -> {save_path}")
                def do_download():
                    r = requests.get(f"{FILES_API_URL}/{output_file_id}/content", headers=self.headers)
                    r.raise_for_status()
                    return r.text

                content = self.robust_request(do_download)
                with open(save_path, "w", encoding="utf-8") as ff:
                    ff.write(content)
                print(f"[{self.name_tag()}] [下载成功] -> {save_path}")
                return
            except Exception as e:
                print(f"[{self.name_tag()}] 下载异常: {e}, 30s后重试.")
                time.sleep(30)
                continue

    def wait_for_queue_space(self):
        """
        检查当前 Key 对应的队列是否超量(运行/排队中的 batch 数量, enqueued token数).
        超量则等待. 不放弃.
        """
        while True:
            batches = self.list_batches()
            active_batches = [b for b in batches if b.get("status") in ("queued", "running")]
            num_active = len(active_batches)

            total_enqueued_tokens = 0
            for b in active_batches:
                total_enqueued_tokens += b.get("enqueued_tokens", 0)

            print(
                f"[{self.name_tag()}] [队列检查] running/queued={num_active}, "
                f"排队token={total_enqueued_tokens}, "
                f"阈值={ENQUEUED_TOKEN_LIMIT}"
            )

            if num_active >= MAX_CONCURRENT_BATCHES or total_enqueued_tokens >= ENQUEUED_TOKEN_LIMIT:
                print(f"[{self.name_tag()}] 并发/Token超限, 等待30s后重试...")
                time.sleep(30)
            else:
                break
