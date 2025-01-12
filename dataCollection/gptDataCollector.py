import os
import re
import csv
import json
import time
import random
import requests
from itertools import combinations
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------------
# huggingface_hub 依赖
# ----------------------------------------------------
# 若版本较老，无法导入 HfHubHTTPError，则统一使用 requests.exceptions.HTTPError 处理
from huggingface_hub import HfApi, CommitOperationAdd

# =========================
# 一、全局配置
# =========================

# ---------- 1) 路径与文件命名 ----------
RENDER_ROOT_DIR = "data/renderPointCloud"   
FILENAME_REGEX = r"^pointcloud_(\d+)_view(\w+)\.jpg$"  

# ---------- 2) Hugging Face 配置 ----------
HF_TOKEN = "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"  
HF_REPO_ID = "Battam/PLM-Images"     
HF_REPO_TYPE = "dataset"             

BATCH_COMMIT_SIZE = 50               
SLEEP_BETWEEN_COMMITS = 1            

# 是否在上传前检查 Hugging Face Repo 中是否已存在相同文件
CHECK_EXISTS_BEFORE_UPLOAD = True

# ---------- 3) OpenAI 接口配置 ----------
API_KEY = "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"  # 替换为你的 OpenAI API Key
API_URL = "https://api.openai.com/v1/chat/completions"
BATCH_API_URL = "https://api.openai.com/v1/batches"
FILES_API_URL = "https://api.openai.com/v1/files"
MODEL = "gpt-4o-2024-11-20"

# ---------- 4) 对话系统 Prompt ----------
SYSTEM_PROMPT = (
    "You are an AI assistant designed to analyze and compare 3D point clouds (Already rendered as images). "
    "You must strictly follow instructions and adhere to the defined response formats: "
    "1. Provide concise descriptions of image features relevant to the given task. "
    "2. Focus on the objective when performing comparisons. Avoid adding speculative or irrelevant details. "
    "3. Follow the response rules provided in the task. "
    "4. Pay special attention to the presence of robotic grippers or mechanical claws in the images, "
    "   as they may be crucial for understanding task execution."
)

# ---------- 5) 任务目标提示 -----------
objective_env_prompts = {
    "metaworld_drawer-open-v2": "to open the drawer",
    "metaworld_door-open-v2": "to open the safe door",
    "metaworld_soccer-v2": "to move the soccer ball into the goal",
    "metaworld_disassemble-v2": "The peg is disassembled.",
    "metaworld_handle-pull-side-v2": "The handle is pulled to the side.",
    "metaworld_peg-insert-side-v2": "The peg is inserted into the hole.",
}

# ---------- 6) 单轮对话模板 ----------
single_round_template = (
    "Consider the following two images with URLs:\n"
    "Image 1 URL: {img1_url}\n"
    "Image 2 URL: {img2_url}\n"
    "Objective: {objective}\n\n"
    "Please answer the following questions one by one:\n"
    "1. What is shown in Image 1?\n"
    "2. What is shown in Image 2?\n"
    "3. Is there any difference between Image 1 and Image 2 in terms of achieving the objective?\n\n"
    "After answering these questions, based on your answers, conclude by replying with a single line:\n"
    "- Reply '0' if the objective is better achieved in Image 1.\n"
    "- Reply '1' if the objective is better achieved in Image 2.\n"
    "- Reply '-1' if you are unsure or there is no difference."
)


# =========================
# 二、辅助函数：生成图像对
# =========================

def parse_filename(filename: str) -> Tuple[int, str]:
    """
    解析类似 pointcloud_000001_view1.jpg -> (1, '1')
    如果不符合正则 FILENAME_REGEX，抛出 ValueError
    """
    match = re.match(FILENAME_REGEX, filename)
    if not match:
        raise ValueError(f"文件名 {filename} 不符合正则 {FILENAME_REGEX}")
    index_val = int(match.group(1))
    view_str = match.group(2)
    return index_val, view_str


def gather_images_by_view(env_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    扫描 env_dir 中所有 .jpg 文件，解析 (index, view)，按 view 分组。
    返回: {view: [(idx, filename), (idx2, filename2), ...]}
    """
    view_dict: Dict[str, List[Tuple[int, str]]] = {}
    if not os.path.isdir(env_dir):
        return view_dict

    all_files = os.listdir(env_dir)
    for fname in all_files:
        if not fname.lower().endswith(".jpg"):
            continue
        full_path = os.path.join(env_dir, fname)
        if not os.path.isfile(full_path):
            continue
        try:
            idx, v = parse_filename(fname)
        except ValueError:
            continue
        view_dict.setdefault(v, []).append((idx, fname))

    # 每个视角按 index 排序
    for v in view_dict:
        view_dict[v].sort(key=lambda x: x[0])

    return view_dict


def generate_pairs_for_env(
    env_name: str,
    selected_views: Optional[List[str]] = None,
    max_pairs: Optional[int] = None
) -> List[Tuple[str, str, str]]:
    """
    对指定环境 env_name:
      1. 收集同一 view 下的所有图像 (index, filename)
      2. 生成 C(n,2) 对 (filename1, filename2, view)
      3. 如果 selected_views 不为空，则只保留这些 view；否则全用
      4. 如果 max_pairs 不为空，则随机抽取 max_pairs 对返回
    """
    env_dir = os.path.join(RENDER_ROOT_DIR, env_name)
    if not os.path.isdir(env_dir):
        print(f"[警告] 环境 {env_name} 不存在目录 {env_dir}，跳过。")
        return []

    # 1. 按 view 分组
    view_dict = gather_images_by_view(env_dir)
    if not view_dict:
        print(f"[{env_name}] 未找到符合命名规则的 JPG 文件。")
        return []

    # 2. 若 selected_views 不为空，则只保留相关 view
    if selected_views:
        used_views = [v for v in selected_views if v in view_dict]
    else:
        used_views = list(view_dict.keys())

    # 3. 逐个 view 生成 C(n,2)
    from itertools import combinations
    all_pairs = []
    for v in used_views:
        items = view_dict[v]  # [(idx, filename), ...]
        idx_list = [it[0] for it in items]
        idx_to_fname = {it[0]: it[1] for it in items}
        combis = list(combinations(idx_list, 2))
        for (i1, i2) in combis:
            f1, f2 = idx_to_fname[i1], idx_to_fname[i2]
            # 确保顺序一致
            if f1 > f2:
                f1, f2 = f2, f1
            all_pairs.append((f1, f2, v))

    # 4. 随机抽取 max_pairs
    if all_pairs and max_pairs is not None:
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:max_pairs]

    print(f"[{env_name}] 最终生成 {len(all_pairs)} 个图像对 (视角过滤={selected_views}, max_pairs={max_pairs})")
    return all_pairs


# =========================
# 三、批量上传到 Hugging Face
# =========================

class BatchedImageCommitter:
    """
    实现批量上传功能，减少频繁请求导致的 429:
      1. 有一个 buffer (operations_buffer + local_paths_buffer)
      2. 当 buffer 满时，使用 create_commit 提交
      3. 提交后记录缓存 local_path -> HF URL，避免重复上传
      4. 若遇到 429 则重试 + 指数退避

    同时提供可选功能: check_exists_before_upload
      - 若为 True, 则在上传前检查仓库中文件列表, 如果已存在相同 path_in_repo 则直接跳过
      - 若为 False, 则不做检查, 直接提交 (覆盖)
    """
    def __init__(
        self,
        hf_token: str,
        repo_id: str,
        repo_type: str,
        batch_commit_size: int = 50,
        sleep_between_commits: int = 1,
        check_exists_before_upload: bool = True
    ):
        # 基本信息
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi()

        # 缓冲区
        self.operations_buffer: List[CommitOperationAdd] = []
        self.local_paths_buffer: List[str] = []

        # 本地缓存: local_path -> HF raw_url
        self.upload_cache: Dict[str, str] = {}

        # Commit 计数器
        self.commit_counter = 0

        # 配置项
        self.batch_commit_size = batch_commit_size
        self.sleep_between_commits = sleep_between_commits
        self.check_exists_before_upload = check_exists_before_upload

        # 如果需要检查是否已存在文件, 可以在初始化时加载一次 repo 文件列表
        self.repo_files_set: Optional[set] = None
        if self.check_exists_before_upload:
            self._load_repo_file_list()

    def _load_repo_file_list(self):
        """
        加载当前 Repo (self.repo_id) 中已有文件列表, 生成一个 set().
        例如: { 'images/env_name/file1.jpg', ... }
        """
        print(f"[信息] 正在获取 Hugging Face 仓库文件列表 (repo_id={self.repo_id})...")
        all_files = self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.hf_token
        )
        # 转成 set, 便于快速判断是否存在
        self.repo_files_set = set(all_files)
        print(f"[信息] 已获取 {len(self.repo_files_set)} 个文件在仓库中.")

    def _flush_commit(self, env_name: str):
        """
        批量执行一次 commit (若 buffer 不空)。
        包含简单的重试 & 指数退避来应对 429。
        """
        if not self.operations_buffer:
            return

        commit_message = f"[Auto Commit] {len(self.operations_buffer)} images from {env_name}"
        max_retries = 3
        backoff = 3  # 初始退避时间(秒)

        for attempt in range(max_retries):
            try:
                print(f"[{env_name}] [提交] 一次性提交 {len(self.operations_buffer)} 张图片 (尝试 {attempt+1})...")
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    operations=self.operations_buffer,
                    commit_message=commit_message,
                    token=self.hf_token
                )
                # 提交成功后, 更新 upload_cache
                for local_path in self.local_paths_buffer:
                    filename = os.path.basename(local_path)
                    path_in_repo = f"images/{env_name}/{filename}"
                    raw_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
                    self.upload_cache[local_path] = raw_url

                    # 若我们在本次 commit 中覆盖了文件, 就要在 repo_files_set 中也标记它已存在
                    if self.repo_files_set is not None:
                        self.repo_files_set.add(path_in_repo)

                break  # 提交成功, 跳出重试循环

            except requests.exceptions.HTTPError as e:
                # 如果是 429, 做指数退避
                if e.response is not None and e.response.status_code == 429:
                    print(f"[{env_name}] 429 Too Many Requests, 等待 {backoff}s 后重试...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"[{env_name}] 提交时出现 HTTPError: {e}")
                    raise

        else:
            # max_retries 后依然失败, 抛出异常
            raise RuntimeError(f"[{env_name}] 多次重试后仍无法提交至 HF.")

        # 提交成功, 清空 buffer
        self.operations_buffer.clear()
        self.local_paths_buffer.clear()
        self.commit_counter += 1

        if self.sleep_between_commits > 0:
            time.sleep(self.sleep_between_commits)

    def upload_image(self, env_name: str, local_path: str) -> str:
        """
        将本地文件 local_path 放入缓冲区, 若已在 upload_cache 则直接返回.
        若 check_exists_before_upload=True 且文件在 repo_files_set 中, 也直接跳过提交.
        当 buffer 达到 self.batch_commit_size, flush.
        """
        # 如果已经上传过, 直接用缓存 URL
        if local_path in self.upload_cache:
            return self.upload_cache[local_path]

        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"[{env_name}] 找不到文件: {local_path}")

        filename = os.path.basename(local_path)
        path_in_repo = f"images/{env_name}/{filename}"

        # 如果启用了 "检查是否已存在" 功能, 而且 repo_files_set 中包含 path_in_repo, 说明 HF 已有同名文件
        if self.check_exists_before_upload and self.repo_files_set is not None:
            if path_in_repo in self.repo_files_set:
                raw_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
                self.upload_cache[local_path] = raw_url
                print(f"[{env_name}] [跳过上传] {filename} 已在仓库里存在, 使用已有 URL.")
                return raw_url

        # 否则, 我们需要提交
        op = CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local_path)
        self.operations_buffer.append(op)
        self.local_paths_buffer.append(local_path)

        # 如果 buffer 满了就 flush
        if len(self.operations_buffer) >= self.batch_commit_size:
            self._flush_commit(env_name)

        # 预期 URL
        raw_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{path_in_repo}"
        return raw_url

    def finalize(self, env_name: str):
        """处理完环境后, flush 剩余的文件."""
        self._flush_commit(env_name)


# =========================
# 四、OpenAI Batch 相关
# =========================

class GPT4OHandler:
    """
    封装与 GPT-4O 和 Batch API 交互的相关方法, 包括:
      - 上传 .jsonl
      - 创建 batch
      - 查询 batch 状态
      - 下载 batch 结果
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def upload_batch_file(self, file_path: str) -> str:
        """
        上传 .jsonl 格式文件到 OpenAI /v1/files 接口, 返回 file_id.
        """
        if not file_path.endswith(".jsonl"):
            raise ValueError("上传的文件必须是 .jsonl 格式。")
        print(f"[信息] 正在上传 JSONL 文件: {file_path}")

        with open(file_path, "rb") as f:
            resp = requests.post(
                FILES_API_URL,
                headers=self.headers,
                files={"file": (os.path.basename(file_path), f, "application/json")},
                data={"purpose": "batch"}   # purpose="batch" 非常重要
            )
        resp.raise_for_status()
        file_id = resp.json().get("id")
        print(f"[信息] JSONL 文件上传成功, file_id: {file_id}")
        return file_id

    def create_batch(self, file_id: str) -> Dict[str, any]:
        """
        基于已上传的 file_id, 创建 Batch 任务, 返回 batch_info
        """
        print("[信息] 正在创建批处理任务...")
        payload = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }
        resp = requests.post(BATCH_API_URL, headers=self.headers, json=payload)
        resp.raise_for_status()
        batch_info = resp.json()
        print(f"[信息] 批处理任务创建成功, batch_id: {batch_info.get('id')}")
        return batch_info

    def check_batch_status(self, batch_id: str) -> Dict[str, any]:
        """查询指定 batch_id 的任务状态"""
        resp = requests.get(f"{BATCH_API_URL}/{batch_id}", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def download_batch_results(self, output_file_id: str, save_path: str):
        """下载 batch 结果文件, 保存到本地"""
        print(f"[信息] 正在下载批处理结果文件, file_id: {output_file_id}")
        resp = requests.get(f"{FILES_API_URL}/{output_file_id}/content", headers=self.headers)
        resp.raise_for_status()
        with open(save_path, "w", encoding="utf-8") as out_f:
            out_f.write(resp.text)
        print(f"[信息] 批处理结果文件已保存到: {save_path}")


# =========================
# 五、生成 JSONL 并调用 Batch
# =========================

def generate_jsonl_for_pairs(
    env_name: str,
    pairs: List[Tuple[str, str, str]],
    objective: str,
    committer: BatchedImageCommitter,
    output_file: str
):
    """
    针对 (filename1, filename2, view) 列表:
    1) 逐条上传到 Hugging Face(或直接缓存),
    2) 组装 single_round_template,
    3) 写入 .jsonl

    为避免 "duplicate_custom_id" 错误, 每条记录要有唯一 custom_id, 
    比如 custom_id = f"{env_name}_{line_idx}".
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for line_idx, (fname1, fname2, _) in enumerate(pairs, start=1):
            local_path1 = os.path.join(RENDER_ROOT_DIR, env_name, fname1)
            local_path2 = os.path.join(RENDER_ROOT_DIR, env_name, fname2)

            # 上传(或缓存) -> 获取对应的公开 URL
            url1 = committer.upload_image(env_name, local_path1)
            url2 = committer.upload_image(env_name, local_path2)

            # 组合用户 prompt
            user_content = single_round_template.format(
                img1_url=url1,
                img2_url=url2,
                objective=objective
            )

            # custom_id 保持唯一
            custom_id = f"{env_name}_{line_idx}"

            data = {
                "custom_id": custom_id,  # 在这里 unique
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content},
                    ],
                    "max_tokens": 2000
                }
            }
            out_f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[{env_name}] 已生成 JSONL 文件: {output_file}, 共 {len(pairs)} 条记录")


def process_one_environment(
    env_name: str,
    objective: str,
    committer: BatchedImageCommitter,
    gpt_handler: GPT4OHandler,
    selected_views: Optional[List[str]] = None,
    max_pairs: Optional[int] = None
):
    """
    处理单个环境的完整流程:
      1) 生成图像对
      2) 在生成 JSONL 时批量上传到 HF
      3) finalize buffer -> 提交剩余文件
      4) 上传 .jsonl 到 OpenAI + create_batch
      5) 轮询, 如果成功, 下载结果
    """
    print(f"\n=== 开始处理环境 {env_name} ===")

    # Step 1: 生成对
    pairs = generate_pairs_for_env(env_name, selected_views, max_pairs)
    if not pairs:
        print(f"[{env_name}] 无可用图像对, 跳过。")
        return

    # 创建目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join("dataCollection", "Dataset", env_name)
    timestamp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)

    # Step 2: 生成 JSONL (内部批量缓存/上传)
    batch_input_file = os.path.join(timestamp_dir, "batch_input.jsonl")
    generate_jsonl_for_pairs(env_name, pairs, objective, committer, batch_input_file)

    # 提交剩余文件
    committer.finalize(env_name)

    # Step 3: 上传 JSONL, 创建 Batch
    try:
        file_id = gpt_handler.upload_batch_file(batch_input_file)
        batch_info = gpt_handler.create_batch(file_id)
        batch_id = batch_info.get("id")
        if not batch_id:
            print(f"[{env_name}] [错误] 未获取 batch_id, 放弃。")
            return
    except Exception as e:
        print(f"[{env_name}] [错误] 创建批处理失败: {e}")
        return

    # Step 4: 轮询
    while True:
        try:
            st = gpt_handler.check_batch_status(batch_id)
            status = st.get("status", "")
            print(f"[{env_name}] 当前批处理状态: {status}")
        except Exception as e:
            print(f"[{env_name}] [错误] 检查批处理状态失败: {e}")
            return

        if status == "completed":
            print(f"[{env_name}] [完成] 批处理任务成功完成！")
            break
        elif status == "failed":
            print(f"[{env_name}] [失败] 批处理任务失败: {st}")
            return
        else:
            print(f"[{env_name}] 进行中, 等待 60 秒后再次检查...")
            time.sleep(60)

    # Step 5: 下载结果
    out_file_id = st.get("output_file_id")
    if out_file_id:
        result_path = os.path.join(timestamp_dir, "batch_output.jsonl")
        try:
            gpt_handler.download_batch_results(out_file_id, result_path)
        except Exception as e:
            print(f"[{env_name}] [错误] 下载结果失败: {e}")
    else:
        print(f"[{env_name}] [警告] 未找到 output_file_id, 无法下载结果。")

    print(f"=== 环境 {env_name} 处理完毕 ===")


def main():
    """
    主函数: 你可以在这里灵活配置:
      - selected_views: 需要处理的视角列表 (为空则全部)
      - max_pairs: 限制最大对数 (None 表示全部)
      - check_exists_before_upload: True/False

    其中：为避免 “duplicate_custom_id”，我们在 generate_jsonl_for_pairs() 
    中为每条记录生成独一无二的 custom_id。
    """
    selected_views = ["1"]  
    max_pairs = 10          

    # 初始化 BatchedImageCommitter
    committer = BatchedImageCommitter(
        hf_token=HF_TOKEN,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        batch_commit_size=BATCH_COMMIT_SIZE,
        sleep_between_commits=SLEEP_BETWEEN_COMMITS,
        check_exists_before_upload=CHECK_EXISTS_BEFORE_UPLOAD
    )

    gpt_handler = GPT4OHandler(API_KEY)

    for env_name, obj_prompt in objective_env_prompts.items():
        process_one_environment(
            env_name=env_name,
            objective=obj_prompt,
            committer=committer,
            gpt_handler=gpt_handler,
            selected_views=selected_views,
            max_pairs=max_pairs
        )
        # 根据需要，稍作延时，避免对 OpenAI API 频繁调用
        time.sleep(3)

    print("\n[完成] 所有环境处理结束！")


if __name__ == "__main__":
    main()
