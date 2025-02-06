#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hf_uploader_standalone.py

该脚本/模块独立实现 HuggingFaceUploader 功能，支持从任意本地文件夹中
批量上传文件(例如 .jpg 图像)到一个 Hugging Face dataset 仓库中。

功能特性:
  1. 用户可在脚本中配置 Hugging Face 相关访问信息 (HF_TOKEN, REPO_ID, REPO_TYPE等)。
  2. 提供 batch 提交机制, 避免提交过多小文件时浪费性能。
  3. 如已存在文件, 可选跳过, 以免重复上传。
  4. 内置 429(限流)处理, 使用指数退避重试。
  5. 提供命令行接口, 也可被其他Python脚本直接import和调用.

使用示例:
  # (1) 在脚本内写好 TOKEN, REPO_ID 等,
  # (2) 终端调用, 指定需上传的本地文件夹 folder(可包含任意数量的文件):
    python hf_uploader_standalone.py \
        --folder ./my_local_images \
        --env_name soccer_env

  # 该脚本会将 ./my_local_images/*.jpg 上传至 HF 仓库中:
  # images/soccer_env/xxx.jpg
  # images/soccer_env/yyy.jpg
  # ...
  # 成功后在HF dataset中即可看到 "images/soccer_env/xxx.jpg"文件.

注意:
  - 默认只上传 .jpg 文件; 如需上传更多类型, 请在 run_upload() 里自行调整文件后缀筛选.
  - 遇到网络错误时, 会进行最多3次重试; 若3次均失败则抛异常.
  - 如果想修改 batch 提交大小, 在 BATCH_COMMIT_SIZE 处修改即可.

作者: (你自己的名称/说明)
日期: 2025-01-23
"""

import os
import sys
import time
import argparse
import requests
from typing import Dict, List, Optional

# huggingface_hub 提供与HF交互的API
from huggingface_hub import HfApi, CommitOperationAdd

###############################################################################
# 1) 在此处配置 Hugging Face 访问信息
###############################################################################
HF_TOKEN = "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"  # 请填入你的真实 Hugging Face Token
REPO_ID = "Battam/PLM-Finetune" # 你的HF仓库名, e.g. "Battam/PLM-Images"
REPO_TYPE = "dataset"                # 如果是dataset类型, 否则可改 "model" ...
BATCH_COMMIT_SIZE = 100               # 累计多少文件后自动commit
SLEEP_BETWEEN_COMMITS = 1            # 每次commit后休眠秒数
CHECK_EXISTS_BEFORE_UPLOAD = False    # 是否在上传前检查HF仓库中是否已存在同名文件

###############################################################################
# 2) 定义 HuggingFaceUploader 类 (与原先类似, 但改为可独立使用)
###############################################################################
class HuggingFaceUploader:
    """
    HuggingFaceUploader 实现将本地文件批量上传到 Hugging Face dataset仓库。
    - 支持批量提交, 例如攒到 BATCH_COMMIT_SIZE=50 后统一 create_commit().
    - 遇到 429 (限流)则指数退避重试.
    - 可检查HF仓库已存在文件, 跳过重复上传.
    - 提供upload_file(...), finalize(...)等API.

    使用方法:
      uploader = HuggingFaceUploader()
      uploader.upload_file(env_name="soccer", local_path="./my_images/img1.jpg")
      ...
      uploader.finalize(env_name="soccer")
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
        构造函数, 初始化各项参数.

        :param hf_token: Hugging Face 的访问token (带写权限).
        :param repo_id: 目标 Hugging Face 仓库名称, 形如 "username/RepoName".
        :param repo_type: 通常 "dataset" 或 "model".
        :param batch_commit_size: 累计多少文件后自动提交.
        :param sleep_between_commits: 每次commit后休眠的秒数(避免过快提交).
        :param check_exists_before_upload: 是否检查重复文件, True则跳过已存在的相同文件.
        """
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.batch_commit_size = batch_commit_size
        self.sleep_between_commits = sleep_between_commits
        self.check_exists_before_upload = check_exists_before_upload

        # huggingface_hub 的 HfApi 实例
        self.api = HfApi()

        # 暂存的 commit operations
        self.operations_buffer: List[CommitOperationAdd] = []
        self.local_paths_buffer: List[str] = []
        # 缓存: local_path -> HF url
        self.upload_cache: Dict[str, str] = {}

        # 已经提交过多少次commit
        self.commit_counter = 0

        # 若要检查是否已存在 => 加载repo文件列表
        self.repo_files_set: Optional[set] = None
        if self.check_exists_before_upload:
            self._load_repo_file_list()

    def _load_repo_file_list(self):
        """
        加载HF仓库文件列表(仅文件路径), 存入self.repo_files_set, 用于检查重复.
        """
        print(f"[HFUploader] 正在加载仓库 {self.repo_id} 的文件列表...")
        all_files = self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.hf_token
        )
        self.repo_files_set = set(all_files)
        print(f"[HFUploader] 仓库中已有 {len(self.repo_files_set)} 个文件.")

    def _flush_commit(self, env_name: str):
        """
        将 operations_buffer 中的文件统一提交到 HF.
        遇到 429 => 指数退避. 超过三次则报错.

        :param env_name: 用于在提交信息/打印日志时标识
        """
        if not self.operations_buffer:
            return
        commit_msg = f"[Auto Commit] {len(self.operations_buffer)} from {env_name}"
        max_retries=3
        backoff=3

        for attempt in range(max_retries):
            try:
                print(f"[HFUploader] [{env_name}] 提交 {len(self.operations_buffer)} 个文件 (尝试 {attempt+1})")
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    operations=self.operations_buffer,
                    commit_message=commit_msg,
                    token=self.hf_token
                )

                # 提交成功 => 将本批次文件记录到 upload_cache, 并在 repo_files_set 中标记
                for lp in self.local_paths_buffer:
                    fname=os.path.basename(lp)
                    path_in_repo = f"{env_name}/{fname}"
                    file_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
                    self.upload_cache[lp] = file_url
                    if self.repo_files_set is not None:
                        self.repo_files_set.add(path_in_repo)
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code==429:
                    print(f"[HFUploader] [{env_name}] 遇到429限流, 等待 {backoff}秒后重试...")
                    time.sleep(backoff)
                    backoff*=2
                else:
                    print(f"[HFUploader] [{env_name}] 提交失败: {e}")
                    raise
        else:
            raise RuntimeError(f"[HFUploader] [{env_name}] 多次重试后仍提交失败,终止.")

        # 清空缓冲
        self.operations_buffer.clear()
        self.local_paths_buffer.clear()
        self.commit_counter+=1

        # 可选休眠
        if self.sleep_between_commits>0:
            time.sleep(self.sleep_between_commits)

    def upload_file(self, env_name: str, local_path: str)->str:
        """
        上传单个文件(例如图片)到 huggingface 仓库的 "images/{env_name}/" 目录下。
        - 如果 check_exists_before_upload=True, 并检测到已存在, 则跳过.
        - 若缓冲已达 batch_commit_size, 则自动 flush.

        :param env_name: 用于区分/在HF端的文件路径前缀. 也可指定为 "soccer"等.
        :param local_path: 本地文件路径
        :return: 上传后在HF上的URL (或已存在时的URL)
        :raises: FileNotFoundError, requests.exceptions等
        """
        # 若之前已经传过 => 直接返回
        if local_path in self.upload_cache:
            return self.upload_cache[local_path]

        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"[HFUploader] [{env_name}] 文件不存在: {local_path}")

        fname = os.path.basename(local_path)
        path_in_repo = f"{env_name}/{fname}"

        # 检查仓库是否已存在
        if self.check_exists_before_upload and self.repo_files_set is not None:
            if path_in_repo in self.repo_files_set:
                skip_url=f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
                self.upload_cache[local_path]=skip_url
                print(f"[HFUploader] [{env_name}] 跳过 {fname}, 已存在于HF.")
                return skip_url

        # 放入缓冲
        operation=CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local_path)
        self.operations_buffer.append(operation)
        self.local_paths_buffer.append(local_path)

        # 若超批量 => flush
        if len(self.operations_buffer)>=self.batch_commit_size:
            self._flush_commit(env_name)

        raw_url=f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
        return raw_url

    def finalize(self, env_name: str):
        """
        最后可调用一次, 确保将缓冲剩余文件也提交.
        """
        self._flush_commit(env_name)


###############################################################################
# 3) 提供一个命令行函数 run_upload(...)，递归/遍历指定文件夹, 上传所有jpg等
###############################################################################
def run_upload(
    folder: str,
    env_name: str,
    pattern: str = ".jpg"
):
    """
    遍历 folder 下所有 (pattern结尾) 的文件, 使用HuggingFaceUploader 上传到HF.

    :param folder: 本地文件夹路径
    :param env_name: 目标 HF 路径前缀(相当于子文件夹), e.g. "soccer"
    :param pattern: 用于筛选文件后缀(默认".jpg"), 
                    如果想包括png等可自己修改/或加判断.

    使用示例:
      run_upload(folder="./my_images", env_name="soccer")
      => 上传 "./my_images/*.jpg" 文件
    """
    uploader = HuggingFaceUploader(
        hf_token=HF_TOKEN,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        batch_commit_size=BATCH_COMMIT_SIZE,
        sleep_between_commits=SLEEP_BETWEEN_COMMITS,
        check_exists_before_upload=CHECK_EXISTS_BEFORE_UPLOAD
    )

    if not os.path.isdir(folder):
        raise ValueError(f"指定的folder不存在: {folder}")

    # 遍历文件夹,找所有后缀=pattern的文件(大小写都算)
    # 你也可以做更复杂的递归
    file_list = []
    for f in os.listdir(folder):
        fullp = os.path.join(folder, f)
        if os.path.isfile(fullp) and f.lower().endswith(pattern.lower()):
            file_list.append(fullp)

    if not file_list:
        print(f"[INFO] 文件夹 {folder} 中未找到 *{pattern} 文件, 结束.")
        return

    print(f"[INFO] 在 {folder} 中共找到 {len(file_list)} 个匹配'{pattern}'的文件, 准备上传 => env_name={env_name}")
    # 逐个上传
    for fp in file_list:
        url = uploader.upload_file(env_name, fp)
        print(f" - Uploaded: {fp} => {url}")

    # 最后 flush
    uploader.finalize(env_name)
    print("[DONE] 全部上传完成.")


###############################################################################
# 4) main函数, 支持命令行参数
###############################################################################
def main():
    """
    若你直接运行本文件, 可通过命令行传入参数:
      --folder  指定本地文件夹
      --env_name 指定 HF上的文件夹前缀(子路径)
    示例:
      python hf_uploader_standalone.py --folder ./my_images --env_name soccer
    """
    # parser = argparse.ArgumentParser(
    #     description="HuggingFace Uploader - 上传指定folder下的.jpg文件到HF"
    # )
    # parser.add_argument("--folder", type=str, required=True, help="本地文件夹路径")
    # parser.add_argument("--env_name", type=str, required=True, help="HF子文件夹名称,例如'soccer_env'")
    # parser.add_argument("--ext", type=str, default=".jpg", help="文件后缀(默认.jpg)")

    # args=parser.parse_args()
    # run_upload(folder=args.folder, env_name=args.env_name, pattern=args.ext)

    folder = f"dataCollection/Dataset"
    env_name = f"Jsonl"
    ext = f".jsonl"

    # folder = f"data/pointclouds/metaworld_drawer-open-v2"
    # env_name = f"PointClouds/metaworld_drawer-open-v2"
    # ext = f".npy"

    # folder = f"data/pointclouds/metaworld_handle-pull-side-v2"
    # env_name = f"PointClouds/metaworld_handle-pull-side-v2"
    # ext = f".npy"

    # folder = f"data/pointclouds/metaworld_peg-insert-side-v2"
    # env_name = f"PointClouds/metaworld_peg-insert-side-v2"
    # ext = f".npy"

    # folder = f"data/pointclouds/metaworld_soccer-v2"
    # env_name = f"PointClouds/metaworld_soccer-v2"
    # ext = f".npy"

    run_upload(folder=folder, env_name=env_name, pattern=ext)

if __name__=="__main__":
    main()
