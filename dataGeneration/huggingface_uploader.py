# /home/star/Yanjun/RL-VLM-F/dataGeneration/huggingface_uploader.py
# -------------------------------------------------------------------------------
"""
huggingface_uploader.py

定义 HuggingFaceUploader 类, 用于批量上传本地图像到 Hugging Face Repo.
包含自动缓存、跳过已存在、以及自动 commit 提交.
"""

import os
import time
import requests
from typing import Dict, List, Optional
from huggingface_hub import HfApi, CommitOperationAdd

# 从 config 中读取相关配置常量
from config import (
    HF_TOKEN, HF_REPO_ID, HF_REPO_TYPE,
    BATCH_COMMIT_SIZE, SLEEP_BETWEEN_COMMITS,
    CHECK_EXISTS_BEFORE_UPLOAD
)

class HuggingFaceUploader:
    """
    用于将本地图像批量上传到 Hugging Face Repo。
    1. 缓冲 + create_commit()
    2. 遇到429进行指数退避重试
    3. 可选地跳过已存在文件
    4. upload_cache 避免重复上传同一张图
    """

    def __init__(
        self,
        hf_token: str = HF_TOKEN,
        repo_id: str = HF_REPO_ID,
        repo_type: str = HF_REPO_TYPE,
        batch_commit_size: int = BATCH_COMMIT_SIZE,
        sleep_between_commits: int = SLEEP_BETWEEN_COMMITS,
        check_exists_before_upload: bool = CHECK_EXISTS_BEFORE_UPLOAD
    ):
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.batch_commit_size = batch_commit_size
        self.sleep_between_commits = sleep_between_commits
        self.check_exists_before_upload = check_exists_before_upload

        self.api = HfApi()
        self.operations_buffer: List[CommitOperationAdd] = []
        self.local_paths_buffer: List[str] = []
        self.upload_cache: Dict[str, str] = {}  # local_path -> HF url

        self.commit_counter = 0

        # 若需要检查文件是否已存在
        self.repo_files_set: Optional[set] = None
        if self.check_exists_before_upload:
            self._load_repo_file_list()

    def _load_repo_file_list(self):
        """
        加载 HF repo 的文件列表为 set，方便判断是否存在。
        """
        print(f"[HFUploader] 加载仓库 {self.repo_id} 文件列表...")
        all_files = self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.hf_token
        )
        self.repo_files_set = set(all_files)
        print(f"[HFUploader] 已存在 {len(self.repo_files_set)} 个文件.")

    def _flush_commit(self, env_name: str):
        """
        一次性提交缓冲中的文件到 HF。带 429 重试。
        如遇429, 指数退避后重试.
        """
        if not self.operations_buffer:
            return
        commit_msg = f"[Auto Commit] {len(self.operations_buffer)} from {env_name}"
        max_retries = 3
        backoff = 3

        for attempt in range(max_retries):
            try:
                print(f"[HFUploader] [{env_name}] 提交 {len(self.operations_buffer)} 张图 (尝试 {attempt+1})")
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    operations=self.operations_buffer,
                    commit_message=commit_msg,
                    token=self.hf_token
                )
                # 提交成功后，更新缓存
                for lp in self.local_paths_buffer:
                    fname = os.path.basename(lp)
                    path_in_repo = f"images/{env_name}/{fname}"
                    url = f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
                    self.upload_cache[lp] = url
                    if self.repo_files_set is not None:
                        self.repo_files_set.add(path_in_repo)
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print(f"[HFUploader] [{env_name}] 429 限流, 等待 {backoff}s 重试...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"[HFUploader] [{env_name}] 提交失败: {e}")
                    raise
        else:
            raise RuntimeError(f"[HFUploader] [{env_name}] 多次重试后仍失败, 终止")

        self.operations_buffer.clear()
        self.local_paths_buffer.clear()
        self.commit_counter += 1

        if self.sleep_between_commits > 0:
            time.sleep(self.sleep_between_commits)

    def upload_image(self, env_name: str, local_path: str) -> str:
        """
        将 local_path 所指的图像放进缓冲；如到达批量提交上限则 flush。
        返回在 Hugging Face 上对应的 URL。
        如果已上传过 (upload_cache)，则直接返回。
        """
        if local_path in self.upload_cache:
            return self.upload_cache[local_path]
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"[HFUploader] [{env_name}] 文件不存在: {local_path}")

        fname = os.path.basename(local_path)
        p_repo = f"images/{env_name}/{fname}"

        # 检查是否已存在
        if self.check_exists_before_upload and self.repo_files_set is not None:
            if p_repo in self.repo_files_set:
                skip_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{p_repo}"
                self.upload_cache[local_path] = skip_url
                print(f"[HFUploader] [{env_name}] 跳过 {fname}, 已存在.")
                return skip_url

        # 放入缓冲
        op = CommitOperationAdd(path_in_repo=p_repo, path_or_fileobj=local_path)
        self.operations_buffer.append(op)
        self.local_paths_buffer.append(local_path)

        # 若已达到批量提交上限 -> flush
        if len(self.operations_buffer) >= self.batch_commit_size:
            self._flush_commit(env_name)

        raw_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{p_repo}"
        return raw_url

    def finalize(self, env_name: str):
        """
        将缓冲区中剩余文件全部提交。
        """
        self._flush_commit(env_name)
