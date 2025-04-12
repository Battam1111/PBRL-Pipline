#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hf_uploader_single_file.py

该脚本独立实现 HuggingFaceUploader 功能，用于将指定的单个本地文件上传到 Hugging Face dataset 仓库中。

主要功能特性：
  1. 用户在代码中直接指定上传的单个文件路径和目标环境名称，无需命令行传入参数；
  2. 可配置 Hugging Face 的相关访问信息，如 HF_TOKEN、REPO_ID、REPO_TYPE 等；
  3. 内置文件存在检查、批量提交机制（即使只有单文件也走统一提交流程）；
  4. 内置 429 限流处理，采用指数退避重试策略，确保网络异常情况下依然能上传；
  5. 注释和代码说明全面详细，便于后续维护和扩展。

使用说明：
  - 在代码中修改 HF_TOKEN、REPO_ID、REPO_TYPE 等全局变量为你实际使用的值；
  - 在 main() 函数中直接指定待上传的单个文件路径 file_path 和目标环境名称 env_name；
  - 运行脚本后，文件将会上传到 Hugging Face dataset 仓库中，
    上传后的文件路径为："{env_name}/{文件名}"，可在 HF 仓库中查看上传结果。

作者: 你的姓名或说明
日期: 2025-01-23
"""

import os
import time
import requests
from typing import Dict, List, Optional

# 从 huggingface_hub 库导入与 HF 交互的 API 类以及提交操作类
from huggingface_hub import HfApi, CommitOperationAdd

###############################################################################
# 1) 配置 Hugging Face 访问信息及相关参数
###############################################################################
HF_TOKEN = "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"  # 请替换为你的 Hugging Face 访问 Token，必须具有写权限
REPO_ID = "Battam/3D-CoT"  # 目标 Hugging Face 仓库名称，例如 "Battam/PLM-Finetune"
REPO_TYPE = "dataset"            # 仓库类型，通常为 "dataset"，若为模型则可修改为 "model"
BATCH_COMMIT_SIZE = 1          # 累计多少个文件后自动提交，本次上传单文件依然使用此参数
SLEEP_BETWEEN_COMMITS = 1        # 每次提交后休眠的秒数，避免过快提交导致问题
CHECK_EXISTS_BEFORE_UPLOAD = False  # 是否在上传前检查仓库中是否已存在同名文件（True: 检查并跳过上传；False: 不检查）

###############################################################################
# 2) 定义 HuggingFaceUploader 类，用于实现文件上传功能
###############################################################################
class HuggingFaceUploader:
    """
    HuggingFaceUploader 类用于将本地文件批量（或单个）上传到 Hugging Face dataset 仓库中。
    主要特性：
      - 支持批量提交：文件达到一定数量后自动提交，降低性能消耗；
      - 内置 429 (限流) 错误处理：采用指数退避重试机制，保证上传的鲁棒性；
      - 可选检查仓库中是否已存在相同文件，避免重复上传；
      - 提供 upload_file() 方法用于上传单个文件，finalize() 方法用于提交剩余文件。
    使用示例:
      uploader = HuggingFaceUploader(...)
      url = uploader.upload_file(env_name="example_env", local_path="/path/to/file.jpg")
      uploader.finalize(env_name="example_env")
    """

    def __init__(
        self,
        hf_token: str = HF_TOKEN,
        repo_id: str = REPO_ID,
        repo_type: str = REPO_TYPE,
        batch_commit_size: int = BATCH_COMMIT_SIZE,
        sleep_between_commits: int = SLEEP_BETWEEN_COMMITS,
        check_exists_before_upload: bool = CHECK_EXISTS_BEFORE_UPLOAD
    ):
        """
        构造函数，初始化上传器的各项参数。

        :param hf_token: Hugging Face 的访问 Token（必须具备写权限）。
        :param repo_id: 目标 Hugging Face 仓库名称，如 "username/RepoName"。
        :param repo_type: 仓库类型，通常为 "dataset"，若为模型则为 "model"。
        :param batch_commit_size: 累计多少文件后自动提交一次，减少频繁提交带来的性能消耗。
        :param sleep_between_commits: 每次提交后等待的秒数，防止提交过快。
        :param check_exists_before_upload: 是否在上传前检查仓库中是否已存在同名文件，
                                            若为 True，则已存在的文件会跳过上传，避免重复。
        """
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.batch_commit_size = batch_commit_size
        self.sleep_between_commits = sleep_between_commits
        self.check_exists_before_upload = check_exists_before_upload

        # 创建 Hugging Face API 实例，用于与 HF 进行交互
        self.api = HfApi()

        # 用于存储待提交的操作列表（CommitOperationAdd 对象）
        self.operations_buffer: List[CommitOperationAdd] = []
        # 记录与操作缓冲区对应的本地文件路径
        self.local_paths_buffer: List[str] = []
        # 缓存：本地文件路径 -> 上传后在 HF 上的 URL
        self.upload_cache: Dict[str, str] = {}

        # 记录已经执行的提交次数（用于日志显示）
        self.commit_counter = 0

        # 若需要检查文件是否已存在，则加载仓库中已有的文件列表
        self.repo_files_set: Optional[set] = None
        if self.check_exists_before_upload:
            self._load_repo_file_list()

    def _load_repo_file_list(self):
        """
        加载目标仓库中的所有文件列表，并存储在 self.repo_files_set 中，
        以便后续判断是否需要跳过已存在的文件上传。
        """
        print(f"[HFUploader] 正在加载仓库 {self.repo_id} 的文件列表...")
        all_files = self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.hf_token
        )
        self.repo_files_set = set(all_files)
        print(f"[HFUploader] 仓库中已有 {len(self.repo_files_set)} 个文件。")

    def _flush_commit(self, env_name: str):
        """
        将缓冲区中的文件操作统一提交到 Hugging Face 仓库。
        如果遇到 429 限流错误，则采用指数退避策略重试；重试超过设定次数后将抛出异常。

        :param env_name: 上传环境名称，用于构造提交信息及日志打印。
        """
        if not self.operations_buffer:
            # 若缓冲区为空，则无需提交
            return
        commit_msg = f"[Auto Commit] 提交 {len(self.operations_buffer)} 个文件，环境: {env_name}"
        max_retries = 3  # 最大重试次数
        backoff = 3      # 初始等待时间（秒）

        for attempt in range(max_retries):
            try:
                print(f"[HFUploader] [{env_name}] 正在提交 {len(self.operations_buffer)} 个文件（尝试 {attempt + 1} 次）...")
                # 调用 Hugging Face API 进行提交操作
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    operations=self.operations_buffer,
                    commit_message=commit_msg,
                    token=self.hf_token
                )

                # 提交成功后，将本次提交的文件记录到上传缓存中，同时更新已存在文件集合（如果需要检查重复上传）
                for lp in self.local_paths_buffer:
                    fname = os.path.basename(lp)
                    path_in_repo = f"{env_name}/{fname}" if env_name else fname
                    file_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
                    self.upload_cache[lp] = file_url
                    if self.repo_files_set is not None:
                        self.repo_files_set.add(path_in_repo)
                break  # 成功提交则退出重试循环
            except requests.exceptions.HTTPError as e:
                # 针对 HTTPError 判断是否为 429 限流错误
                if e.response is not None and e.response.status_code == 429:
                    print(f"[HFUploader] [{env_name}] 遇到 429 限流，等待 {backoff} 秒后重试...")
                    time.sleep(backoff)
                    backoff *= 2  # 指数退避：等待时间加倍
                else:
                    # 非限流错误直接抛出异常
                    print(f"[HFUploader] [{env_name}] 提交失败，错误信息: {e}")
                    raise
        else:
            # 如果多次重试后仍然失败，则抛出运行时异常
            raise RuntimeError(f"[HFUploader] [{env_name}] 多次重试后提交仍失败，终止上传。")

        # 提交完成后，清空操作缓冲区和本地路径记录，并增加提交计数器
        self.operations_buffer.clear()
        self.local_paths_buffer.clear()
        self.commit_counter += 1

        # 可选：每次提交后暂停一段时间，防止过快提交
        if self.sleep_between_commits > 0:
            time.sleep(self.sleep_between_commits)

    def upload_file(self, env_name: str, local_path: str) -> str:
        """
        上传单个文件到 Hugging Face 仓库中指定的路径。
        文件将被上传到路径 "{env_name}/{文件名}"，如果 env_name 为空，则直接在根目录下上传。
        - 如果之前已经上传过该文件，则直接返回对应的 URL；
        - 如果设置了检查重复上传，则在上传前检查仓库中是否已存在同名文件，
          若存在则跳过上传并直接返回对应的 URL；
        - 上传过程中，若缓冲区中累计的操作达到设定阈值，则自动执行提交操作。

        :param env_name: 用于指定文件在 HF 仓库中的子目录（可理解为上传环境名称），例如 "example_env"；
                         若设为空字符串，则文件直接上传至仓库根目录。
        :param local_path: 本地待上传文件的绝对或相对路径。
        :return: 文件上传后在 Hugging Face 上的访问 URL。
        :raises: FileNotFoundError 当指定文件不存在时抛出异常，
                  以及 requests.exceptions 相关的网络异常。
        """
        # 如果该文件之前已上传过，则直接返回缓存中的 URL
        if local_path in self.upload_cache:
            return self.upload_cache[local_path]

        # 检查本地文件是否存在，若不存在则抛出异常
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"[HFUploader] [{env_name}] 指定文件不存在: {local_path}")

        fname = os.path.basename(local_path)
        # 根据是否指定 env_name 来确定上传到 HF 仓库中的文件路径
        path_in_repo = f"{env_name}/{fname}" if env_name else fname

        # 如果需要检查重复上传，则判断仓库中是否已存在相同路径的文件
        if self.check_exists_before_upload and self.repo_files_set is not None:
            if path_in_repo in self.repo_files_set:
                skip_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
                self.upload_cache[local_path] = skip_url
                print(f"[HFUploader] [{env_name}] 文件 {fname} 已存在于 HF 仓库，跳过上传。")
                return skip_url

        # 构造一个提交操作，将本地文件与目标路径关联
        operation = CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local_path)
        self.operations_buffer.append(operation)
        self.local_paths_buffer.append(local_path)

        # 若缓冲区达到批量提交阈值，则自动提交
        if len(self.operations_buffer) >= self.batch_commit_size:
            self._flush_commit(env_name)

        # 返回文件在 HF 仓库中的访问 URL（注意：可能还未提交，但 URL 格式固定）
        raw_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
        return raw_url

    def finalize(self, env_name: str):
        """
        用于在所有上传操作完成后，确保缓冲区中剩余的文件操作得以提交。
        在单文件上传场景下，也应调用此方法以确保最后的提交。
        
        :param env_name: 上传环境名称，用于日志显示。
        """
        self._flush_commit(env_name)

###############################################################################
# 3) 定义单文件上传流程函数
###############################################################################
def run_upload_single_file():
    """
    单文件上传流程：
      - 在代码中直接指定待上传的文件路径及目标上传环境（子目录）；
      - 创建 HuggingFaceUploader 实例，并调用 upload_file() 方法上传单个文件；
      - 调用 finalize() 方法提交剩余操作；
      - 输出上传后的文件在 Hugging Face 上的访问 URL。
    """
    # -------------------------------------------------------------------------
    # 配置待上传的单个文件的本地路径（请修改为实际文件路径）
    # 例如：file_path = "/home/username/images/sample.jpg"
    # file_path = "dataGenerationOpenai-DataRewrite/rewriteOutput/20250301-192944/gapartnet_sft_27k_openai_output_merged.jsonl"
    # file_path = "dataGenerationOpenai-DataRewrite/rewriteOutput/20250301-230548/cap3d_existed_output_merged.jsonl"
    file_path = "dataGenerationOpenai-DataRewrite/rewriteOutput/20250304-114301/cap3d_objaverse_sft_45k_output_merged.jsonl"
    # -------------------------------------------------------------------------
    # 配置目标上传的环境名称，此名称将作为 HF 仓库中文件存储的子目录；
    # 如果不需要子目录可将 env_name 设置为空字符串 ""
    env_name = ""

    # 输出提示信息，说明开始上传
    print(f"[INFO] 准备上传单个文件: {file_path}")
    print(f"[INFO] 上传到 Hugging Face 仓库 {REPO_ID} 中的目录: '{env_name}'")

    try:
        # 创建上传器实例，参数均可在全局变量中配置
        uploader = HuggingFaceUploader(
            hf_token=HF_TOKEN,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            batch_commit_size=BATCH_COMMIT_SIZE,
            sleep_between_commits=SLEEP_BETWEEN_COMMITS,
            check_exists_before_upload=CHECK_EXISTS_BEFORE_UPLOAD
        )

        # 调用 upload_file() 方法上传单个文件，并获取上传后在 HF 上的 URL
        uploaded_url = uploader.upload_file(env_name, file_path)
        # 调用 finalize() 方法提交所有剩余的操作（在单文件场景下用于确保文件真正提交）
        uploader.finalize(env_name)

        # 输出上传成功信息及文件的 URL
        print(f"[DONE] 文件上传成功！文件在 Hugging Face 上的访问 URL 为:\n{uploaded_url}")
    except Exception as e:
        # 捕获上传过程中可能出现的异常，并输出错误信息
        print(f"[ERROR] 上传过程中出现异常: {e}")

###############################################################################
# 4) 主函数入口
###############################################################################
def main():
    """
    主函数入口：
      - 直接调用 run_upload_single_file() 函数执行单文件上传流程；
      - 该脚本不再使用命令行参数，所有参数均在代码中直接指定。
    """
    run_upload_single_file()

# 如果直接运行该脚本，则调用 main() 函数
if __name__ == "__main__":
    main()
