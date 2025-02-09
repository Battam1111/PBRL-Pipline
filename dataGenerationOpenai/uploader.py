#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uploader.py

定义 HuggingFaceUploader 类，用于将本地图像上传至 Hugging Face 仓库，或直接返回 Base64 编码。
支持：
  - "url" 模式：上传图像后返回远程 URL，支持批量提交和重复检查
  - "base64" 模式：直接读取并返回 Base64 编码字符串
本模块使用 utils.py 中的通用重试函数，保证异常情况能自动退避重试。
"""

import os
import time
import requests
import base64
from typing import Dict, List
from huggingface_hub import HfApi, CommitOperationAdd
from config import HF_TOKEN, HF_REPO_ID, HF_REPO_TYPE, BATCH_COMMIT_SIZE, SLEEP_BETWEEN_COMMITS, CHECK_EXISTS_BEFORE_UPLOAD, IMAGE_UPLOAD_MODE
from utils import robust_request, log

class HuggingFaceUploader:
    def __init__(self,
                 hf_token: str = HF_TOKEN,
                 repo_id: str = HF_REPO_ID,
                 repo_type: str = HF_REPO_TYPE,
                 batch_commit_size: int = BATCH_COMMIT_SIZE,
                 sleep_between_commits: int = SLEEP_BETWEEN_COMMITS,
                 check_exists_before_upload: bool = CHECK_EXISTS_BEFORE_UPLOAD):
        """
        初始化 HuggingFaceUploader 并设置上传模式 (url / base64)。
        如果是 url 模式，且配置了 check_exists_before_upload，则会预先载入仓库文件列表用于去重。
        """
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.batch_commit_size = batch_commit_size
        self.sleep_between_commits = sleep_between_commits
        self.check_exists_before_upload = check_exists_before_upload
        self.upload_mode = IMAGE_UPLOAD_MODE.lower()

        self.api = HfApi()
        self.operations_buffer: List[CommitOperationAdd] = []
        self.local_paths_buffer: List[str] = []
        self.upload_cache: Dict[str, str] = {}  # 映射：本地路径 -> 远程 URL / Base64
        self.commit_counter = 0
        self.repo_files_set = None

        if self.upload_mode == "url" and self.check_exists_before_upload:
            self._load_repo_file_list()

    def _load_repo_file_list(self):
        """预加载仓库文件列表，用于去重判断。"""
        log(f"[HFUploader] 正在加载仓库 {self.repo_id} 的文件列表……")
        try:
            all_files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.hf_token
            )
            self.repo_files_set = set(all_files)
            log(f"[HFUploader] 仓库中已有文件数：{len(self.repo_files_set)}")
        except Exception as e:
            log(f"[HFUploader] 加载仓库文件列表失败：{e}")
            self.repo_files_set = set()

    def _build_hf_url(self, path_in_repo: str) -> str:
        """构造文件在 Hugging Face 仓库中的访问 URL。"""
        if self.repo_type.lower() == "dataset":
            return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{path_in_repo}"
        else:
            return f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"

    def _encode_image_base64(self, local_path: str) -> str:
        """
        对图像文件进行 Base64 编码，返回 data URI 格式的字符串。
        根据扩展名自动判断 MIME 类型。
        """
        ext = os.path.splitext(local_path)[1].lower()
        if ext == ".png":
            mime = "image/png"
        elif ext == ".gif":
            mime = "image/gif"
        elif ext == ".webp":
            mime = "image/webp"
        else:
            mime = "image/jpeg"
        with open(local_path, "rb") as f:
            img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def _flush_commit(self, env_name: str):
        """
        将缓存中的提交操作一次性提交至 Hugging Face 仓库，
        并更新上传缓存与仓库文件列表。
        """
        if not self.operations_buffer:
            return
        commit_msg = f"[Auto Commit] 提交 {len(self.operations_buffer)} 张图片，来源：{env_name}"
        for attempt in range(50):
            try:
                log(f"[HFUploader][{env_name}] 正在提交 {len(self.operations_buffer)} 张图片（第 {attempt+1} 次尝试）")
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
                log(f"[HFUploader][{env_name}] 提交异常：{e}，等待重试……")
                time.sleep(3 * (2 ** attempt))
        else:
            raise RuntimeError(f"[HFUploader][{env_name}] 超过最大重试次数，无法提交图片。")

        self.operations_buffer.clear()
        self.local_paths_buffer.clear()
        self.commit_counter += 1
        if self.sleep_between_commits > 0:
            time.sleep(self.sleep_between_commits)

    def upload_image(self, env_name: str, local_path: str) -> str:
        """
        上传图像并返回其远程 URL 或 Base64 编码。
        如果文件已上传则直接返回缓存结果。
        """
        if local_path in self.upload_cache:
            return self.upload_cache[local_path]

        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"[HFUploader][{env_name}] 文件不存在：{local_path}")

        # 如果是 base64 模式，直接读取并返回编码
        if self.upload_mode == "base64":
            encoded = self._encode_image_base64(local_path)
            self.upload_cache[local_path] = encoded
            log(f"[HFUploader][{env_name}] Base64 编码成功：{local_path}")
            return encoded

        # 否则使用 url 模式，先检查是否已存在
        fname = os.path.basename(local_path)
        path_in_repo = f"images/{env_name}/{fname}"
        if self.check_exists_before_upload and self.repo_files_set is not None:
            if path_in_repo in self.repo_files_set:
                cached_url = self._build_hf_url(path_in_repo)
                self.upload_cache[local_path] = cached_url
                log(f"[HFUploader][{env_name}] 文件已存在，跳过上传：{fname}")
                return cached_url

        # 缓存提交操作，批量提交
        op = CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local_path)
        self.operations_buffer.append(op)
        self.local_paths_buffer.append(local_path)
        if len(self.operations_buffer) >= self.batch_commit_size:
            self._flush_commit(env_name)
        return self._build_hf_url(path_in_repo)

    def finalize(self, env_name: str):
        """强制将剩余缓存中的图像提交至仓库（仅适用于 url 模式）"""
        if self.upload_mode == "url":
            self._flush_commit(env_name)
