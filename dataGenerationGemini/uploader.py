#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uploader.py

本模块定义 HuggingFaceUploader 类，用于将本地图像文件批量上传至 Hugging Face 仓库，
或者直接返回图像的 Base64 编码数据。

主要功能：
  - 缓冲多个上传操作，达到一定数量后一次性提交上传。
  - 上传前检查远程仓库中是否存在相同文件，避免重复上传。
  - 使用指数退避机制处理网络限流及连接异常。
  - 内部维护上传缓存，防止重复上传同一文件。
  - 根据仓库类型构造正确的图像访问 URL。
  - 当 IMAGE_UPLOAD_MODE 为 "base64" 时，直接返回 Base64 编码字符串。

所有关键步骤均附有详细中文注释。
"""

import os
import time
import requests
import base64
import sys
from typing import Dict, List
from huggingface_hub import HfApi, CommitOperationAdd
from config import HF_TOKEN, HF_REPO_ID, HF_REPO_TYPE, BATCH_COMMIT_SIZE, SLEEP_BETWEEN_COMMITS, CHECK_EXISTS_BEFORE_UPLOAD, IMAGE_UPLOAD_MODE

class HuggingFaceUploader:
    def __init__(self,
                 hf_token: str = HF_TOKEN,
                 repo_id: str = HF_REPO_ID,
                 repo_type: str = HF_REPO_TYPE,
                 batch_commit_size: int = BATCH_COMMIT_SIZE,
                 sleep_between_commits: int = SLEEP_BETWEEN_COMMITS,
                 check_exists_before_upload: bool = CHECK_EXISTS_BEFORE_UPLOAD):
        """
        初始化 HuggingFaceUploader 实例。

        参数：
          hf_token: Hugging Face 访问令牌。
          repo_id: 仓库 ID（格式：用户名/仓库名）。
          repo_type: 仓库类型，如 "dataset"。
          batch_commit_size: 达到该数量后触发批量提交。
          sleep_between_commits: 每次提交后休眠秒数。
          check_exists_before_upload: 是否上传前检查仓库是否已有相同文件。
        """
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.batch_commit_size = batch_commit_size
        self.sleep_between_commits = sleep_between_commits
        self.check_exists_before_upload = check_exists_before_upload
        self.upload_mode = IMAGE_UPLOAD_MODE.lower()  # "url" 或 "base64"

        self.api = HfApi()
        self.operations_buffer: List[CommitOperationAdd] = []
        self.local_paths_buffer: List[str] = []
        self.upload_cache: Dict[str, str] = {}
        self.commit_counter = 0

        if self.upload_mode == "url" and self.check_exists_before_upload:
            self._load_repo_file_list()

    def _load_repo_file_list(self):
        """
        从远程仓库加载文件列表，避免重复上传。
        """
        sys.stdout.flush()
        print(f"[HFUploader] 正在加载仓库 {self.repo_id} 文件列表……")
        try:
            all_files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.hf_token
            )
            self.repo_files_set = set(all_files)
            print(f"[HFUploader] 仓库中已存在 {len(self.repo_files_set)} 个文件。")
        except Exception as e:
            print(f"[HFUploader] 加载仓库文件列表失败: {e}")
            self.repo_files_set = set()

    def _build_hf_url(self, path_in_repo: str) -> str:
        """
        根据仓库类型构造图像访问 URL。

        参数：
          path_in_repo: 文件在仓库中的路径。
        返回：
          远程访问 URL。
        """
        if self.repo_type.lower() == "dataset":
            return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{path_in_repo}"
        else:
            return f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"

    def _encode_image_base64(self, local_path: str) -> str:
        """
        读取图像文件并返回 Base64 编码字符串（包含 MIME 前缀）。

        参数：
          local_path: 图像文件本地路径。
        返回：
          Base64 编码后的图像字符串。
        """
        ext = os.path.splitext(local_path)[1].lower()
        if ext in [".png"]:
            mime = "image/png"
        elif ext in [".gif"]:
            mime = "image/gif"
        elif ext in [".webp"]:
            mime = "image/webp"
        else:
            mime = "image/jpeg"
        try:
            with open(local_path, "rb") as f:
                img_bytes = f.read()
            encoded = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
        except Exception as e:
            raise RuntimeError(f"[HFUploader] Base64 编码失败：{e}")

    def robust_request(self, func, max_retries=99999, backoff=3, *args, **kwargs):
        """
        通用请求重试机制：采用指数退避方式重试请求。

        参数：
          func: 请求函数。
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
                        print(f"[robust_request] {e} - 遇到限流（429），等待 {backoff}s（重试 {attempt+1} 次）")
                        sys.stdout.flush()
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    elif status == 402:
                        print(f"[robust_request] {e} - HTTP 402（余额不足）")
                        raise Exception("InsufficientBalanceError: " + str(e))
                else:
                    print(f"[robust_request] 请求异常：{e}，等待 {backoff}s 后重试（重试 {attempt+1} 次）")
                sys.stdout.flush()
                time.sleep(backoff)
                backoff *= 2
                continue
            except Exception as e:
                print(f"[robust_request] 请求失败（非 RequestException）：{e}")
                raise
        raise RuntimeError("[robust_request] 超过最大重试次数，依然失败。")

    def _flush_commit(self, env_name: str):
        """
        将缓冲区内的所有上传操作一次性提交至 Hugging Face 仓库。

        参数：
          env_name: 当前环境名称，用于提交日志信息。
        """
        if not self.operations_buffer:
            return

        commit_msg = f"[Auto Commit] 提交 {len(self.operations_buffer)} 张图片，来源：{env_name}"
        max_retries = 99999
        backoff = 3

        for attempt in range(max_retries):
            try:
                print(f"[HFUploader][{env_name}] 正在提交 {len(self.operations_buffer)} 张图片（重试 {attempt+1} 次）")
                sys.stdout.flush()
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    operations=self.operations_buffer,
                    commit_message=commit_msg,
                    token=self.hf_token
                )
                for lp in self.local_paths_buffer:
                    fname = os.path.basename(lp)
                    path_in_repo = f"images/{env_name}/{fname}"
                    hf_url = self._build_hf_url(path_in_repo)
                    self.upload_cache[lp] = hf_url
                    if self.repo_files_set is not None:
                        self.repo_files_set.add(path_in_repo)
                break
            except requests.exceptions.RequestException as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    print(f"[HFUploader][{env_name}] 限流（429），等待 {backoff}s 后重试……")
                else:
                    print(f"[HFUploader][{env_name}] 提交异常：{e}，等待 {backoff}s 后重试……")
                sys.stdout.flush()
                time.sleep(backoff)
                backoff *= 2
                continue
        else:
            raise RuntimeError(f"[HFUploader][{env_name}] 超过最大重试次数，无法提交图片。")

        self.operations_buffer.clear()
        self.local_paths_buffer.clear()
        self.commit_counter += 1
        if self.sleep_between_commits > 0:
            time.sleep(self.sleep_between_commits)

    def upload_image(self, env_name: str, local_path: str) -> str:
        """
        上传指定本地图像，并返回图像的访问数据（URL 或 Base64 编码）。

        参数：
          env_name: 当前环境名称。
          local_path: 本地图像文件路径。
        返回：
          图像的远程访问 URL 或 Base64 编码字符串。
        """
        if local_path in self.upload_cache:
            return self.upload_cache[local_path]

        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"[HFUploader][{env_name}] 文件不存在：{local_path}")

        if self.upload_mode == "base64":
            try:
                encoded = self._encode_image_base64(local_path)
                self.upload_cache[local_path] = encoded
                print(f"[HFUploader][{env_name}] Base64 编码成功：{local_path}")
                sys.stdout.flush()
                return encoded
            except Exception as e:
                raise RuntimeError(f"[HFUploader][{env_name}] Base64 编码失败：{e}")

        # "url" 模式下采用仓库上传流程
        fname = os.path.basename(local_path)
        path_in_repo = f"images/{env_name}/{fname}"
        if self.check_exists_before_upload and self.repo_files_set is not None:
            if path_in_repo in self.repo_files_set:
                cached_url = self._build_hf_url(path_in_repo)
                self.upload_cache[local_path] = cached_url
                print(f"[HFUploader][{env_name}] 文件 {fname} 已存在于仓库中，跳过上传。")
                sys.stdout.flush()
                return cached_url

        op = CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local_path)
        self.operations_buffer.append(op)
        self.local_paths_buffer.append(local_path)
        if len(self.operations_buffer) >= self.batch_commit_size:
            self._flush_commit(env_name)
        return self._build_hf_url(path_in_repo)

    def finalize(self, env_name: str):
        """
        强制提交所有缓冲中的上传操作，确保所有图像均已上传。

        参数：
          env_name: 当前环境名称。
        """
        if self.upload_mode == "url":
            self._flush_commit(env_name)
