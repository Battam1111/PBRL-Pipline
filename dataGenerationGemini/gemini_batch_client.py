#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gemini_batch_client.py

本模块实现了基于 Google 官方 Gemini API 批量预测功能的封装。
采用官方推荐的批量预测 API 调用方式，通过将任务请求文件上传至 Cloud Storage，
构造批量预测作业，并轮询作业状态，最终下载并解析预测结果。
同时支持多 API KEY 与多个项目 ID 的轮询调用。

主要功能：
    1. 提交批量预测作业：根据任务请求 JSONL 文件的 GCS 路径，构造并提交批量预测作业。
    2. 轮询批量作业状态：采用指数退避策略轮询作业状态，直至作业完成或超时。
    3. 下载并解析预测结果：从指定的 Cloud Storage 输出位置下载结果文件，并解析为 JSON 对象列表。

使用示例：
    from gemini_batch_client import GeminiBatchClient
    from api_key_manager import APIKeyManager
    key_manager = APIKeyManager(api_keys=GEMINI_API_KEY, project_ids=GEMINI_PROJECT_ID)
    client = GeminiBatchClient(
                key_manager=key_manager,
                location="us-central1",
                model="publishers/google/models/gemini-1.0-pro-002"
             )
    job_name = client.submit_batch_job(
                display_name="BatchJob_Example_20250206",
                input_uri="gs://your_bucket/path/to/input.jsonl",
                output_uri="gs://your_bucket/path/to/output"
             )
    job_info = client.poll_job(job_name, poll_interval=30, timeout=3600)
    results = client.download_results(
                output_uri_prefix="gs://your_bucket/path/to/output",
                local_output_dir="./gemini_results"
             )
    print(results)

注意：
    1. 运行本模块前，请确保环境中已正确配置 Google Cloud 认证信息。
    2. 需要安装 google-cloud-storage 包：pip install google-cloud-storage
"""

import os
import time
import json
import requests
import logging
from typing import List, Dict
from google.cloud import storage
from api_key_manager import APIKeyManager

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class GeminiBatchClient:
    def __init__(self, key_manager: APIKeyManager, location: str, model: str, max_retries: int = 5):
        """
        初始化 GeminiBatchClient 实例。

        参数：
            key_manager: APIKeyManager 对象，用于管理多个 API Key 和项目 ID。
            location: 地域（例如 "us-central1"），应与 Gemini 模型所在区域一致。
            model: 使用的 Gemini 模型标识，如 "publishers/google/models/gemini-1.0-pro-002"。
            max_retries: 网络请求的最大重试次数（默认 5 次）。
        """
        self.key_manager = key_manager
        self.location = location
        self.model = model
        self.max_retries = max_retries
        # 从 key_manager 中取出一个初始项目 ID（后续每次调用时都会轮询获取）
        _, project_id = self.key_manager.get_next()
        self.project_id = project_id
        # 构造基础 URL
        self.base_url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}"
        # 获取访问令牌（建议使用 gcloud auth application-default login）
        self.token = self._get_access_token()
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json; charset=utf-8"
        }

    def _get_access_token(self) -> str:
        """
        获取 Google Cloud 访问令牌。依赖环境中已配置的默认凭据。

        返回：
            访问令牌字符串。
        """
        try:
            import google.auth
            import google.auth.transport.requests
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            return credentials.token
        except Exception as e:
            logging.error(f"获取访问令牌失败，请确保已正确配置 Google Cloud 凭据: {e}")
            raise

    def _robust_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        通用的网络请求方法，支持重试和指数退避。

        参数：
            method: 请求方法，如 "GET" 或 "POST"。
            url: 请求 URL。
            **kwargs: 传递给 requests.request 的其他参数。
        返回：
            requests.Response 对象。
        """
        attempt = 0
        backoff = 2
        while attempt < self.max_retries:
            try:
                response = requests.request(method, url, headers=self.headers, **kwargs)
                if response.status_code in [429, 500, 502, 503, 504]:
                    logging.warning(f"请求 {url} 返回状态码 {response.status_code}，等待 {backoff} 秒后重试（第 {attempt+1} 次）")
                    time.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue
                return response
            except requests.RequestException as e:
                logging.warning(f"请求异常：{e}，等待 {backoff} 秒后重试（第 {attempt+1} 次）")
                time.sleep(backoff)
                backoff *= 2
                attempt += 1
        raise Exception(f"请求 {url} 失败，超过最大重试次数。")

    def submit_batch_job(self, display_name: str, input_uri: str, output_uri: str,
                         instances_format: str = "jsonl", predictions_format: str = "jsonl") -> str:
        """
        提交批量预测作业，调用官方 Batch Prediction API。

        参数：
            display_name: 作业显示名称，用于标识此次批量预测任务。
            input_uri: 输入任务请求文件的 GCS 路径，例如 "gs://your_bucket/path/to/input.jsonl"。
            output_uri: 输出结果存储的 GCS 路径前缀，例如 "gs://your_bucket/path/to/output"。
            instances_format: 输入实例的格式，默认为 "jsonl"。
            predictions_format: 预测结果的输出格式，默认为 "jsonl"。
        返回：
            批量预测作业的唯一标识符（job_name）。
        """
        # 每次调用前，动态获取 API Key 和项目 ID，并更新基础 URL
        api_key, project_id = self.key_manager.get_next()
        # 更新项目 ID和基础 URL
        self.project_id = project_id
        self.base_url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}"
        # 更新全局访问令牌（注意：此处不更换 token，依赖环境凭据）
        url = f"{self.base_url}/batchPredictionJobs?key={api_key}"
        payload = {
            "displayName": display_name,
            "model": self.model,
            "inputConfig": {
                "instancesFormat": instances_format,
                "gcsSource": {
                    "uris": [input_uri]
                }
            },
            "outputConfig": {
                "predictionsFormat": predictions_format,
                "gcsDestination": {
                    "outputUriPrefix": output_uri
                }
            }
        }
        logging.info(f"提交批量预测作业，显示名称：{display_name}，输入 URI：{input_uri}，输出 URI：{output_uri}")
        response = self._robust_request("POST", url, data=json.dumps(payload))
        if response.status_code != 200:
            logging.error(f"批量预测作业提交失败：状态码 {response.status_code}，响应内容：{response.text}")
            raise Exception(f"批量预测作业提交失败：{response.status_code} {response.text}")
        job_info = response.json()
        job_name = job_info.get("name")
        if not job_name:
            raise Exception(f"未能获取批量预测作业的名称，响应内容：{job_info}")
        logging.info(f"批量预测作业提交成功，作业名称：{job_name}")
        return job_name

    def poll_job(self, job_name: str, poll_interval: int = 30, timeout: int = 3600) -> Dict:
        """
        轮询批量预测作业状态，直到作业成功完成或超时。

        参数：
            job_name: 批量预测作业的唯一标识符。
            poll_interval: 轮询间隔时间（秒），默认为 30 秒。
            timeout: 总超时时间（秒），默认为 3600 秒。
        返回：
            作业完成后的作业信息字典。
        """
        url = f"https://{self.location}-aiplatform.googleapis.com/v1/{job_name}?key={self.api_key}"
        start_time = time.time()
        logging.info(f"开始轮询作业状态：{job_name}")
        while True:
            response = self._robust_request("GET", url)
            if response.status_code != 200:
                logging.error(f"获取作业状态失败：状态码 {response.status_code}，响应内容：{response.text}")
                raise Exception(f"获取作业状态失败：{response.status_code} {response.text}")
            job_info = response.json()
            state = job_info.get("state", "")
            logging.info(f"当前作业状态：{state}")
            if state == "JOB_STATE_SUCCEEDED":
                logging.info("批量预测作业已成功完成。")
                return job_info
            elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                logging.error(f"批量预测作业失败或已取消，状态：{state}，详细信息：{job_info}")
                raise Exception(f"批量预测作业失败或取消，状态：{state}")
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logging.error("轮询作业状态超时。")
                raise Exception("轮询作业状态超时。")
            logging.info(f"等待 {poll_interval} 秒后继续轮询...")
            time.sleep(poll_interval)

    def download_results(self, output_uri_prefix: str, local_output_dir: str) -> List[Dict]:
        """
        从指定的 Cloud Storage 输出路径下载批量预测结果，并解析为 JSON 对象列表。

        参数：
            output_uri_prefix: 输出结果存储的 GCS 路径前缀，例如 "gs://your_bucket/path/to/output"。
            local_output_dir: 本地存储下载结果文件的目录。
        返回：
            包含所有预测结果的 JSON 对象列表。
        """
        logging.info(f"开始从 Cloud Storage 下载预测结果，输出 URI 前缀：{output_uri_prefix}")
        if not output_uri_prefix.startswith("gs://"):
            raise Exception("输出 URI 必须以 gs:// 开头。")
        # 解析 GCS bucket 名称和前缀路径
        parts = output_uri_prefix[5:].split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        if not os.path.exists(local_output_dir):
            os.makedirs(local_output_dir)
        results = []
        for blob in blobs:
            local_file = os.path.join(local_output_dir, os.path.basename(blob.name))
            logging.info(f"下载文件：{blob.name} 到本地：{local_file}")
            blob.download_to_filename(local_file)
            # 假设下载的文件为 JSONL 格式，每行一个 JSON 对象
            with open(local_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except Exception as e:
                            logging.warning(f"解析文件 {local_file} 中的行失败：{e}")
        logging.info(f"下载并解析完成，共获得 {len(results)} 条预测结果。")
        return results
