# /home/star/Yanjun/RL-VLM-F/dataGeneration/environment_processor.py
# -------------------------------------------------------------------------------
"""
environment_processor.py

在原先基础上，加入了:
1) resume 机制: 若上次处理已有 chunk 输出( batch )或直接输出( direct )，则跳过已完成任务。
2) 处理多视角组合(比如view=1,2，则对pointcloud1与pointcloud2会产生4种视角组合，并随机选一种)。
3) 允许通过 config.USE_BATCH_API 决定使用哪种 API 调用。
4) 大量注释和防护性设计，确保代码更鲁棒。
"""

import os
import re
import time
import json
import random
from typing import Dict, List, Tuple, Optional

from config import (
    RENDER_ROOT_DIR, 
    FILENAME_REGEX,
    objective_env_prompts,
    SYSTEM_PROMPT,
    MODEL,
    single_round_template,
    OPENAI_API_KEYS,
    CHUNK_SIZE_MIN,
    CHUNK_SIZE_MAX,
    USE_BATCH_API
)
from huggingface_uploader import HuggingFaceUploader
from multi_openai_manager import MultiOpenAIBatchManager
from direct_openai_manager import DirectOpenAIManager


class EnvironmentProcessor:
    """
    EnvironmentProcessor 负责处理单个 environment 的数据生成流程：
      1. generate_pairs: 收集 pointcloud_***.jpg 并随机组合(跨点云、跨视角)。
      2. create_big_jsonl: 生成一个大 JSONL (包含所有对比对)。
      3. 若 USE_BATCH_API=True:
         - find_optimal_chunk_size(二分搜索 => test_chunk_size)
         - chunk_file(用最优 chunk_size)
         - multi_openai_manager 并行处理 => 不放弃任何chunk(但有resume机制)
         - merge_outputs => 得到最终合并结果
      4. 若 USE_BATCH_API=False:
         - 使用 DirectOpenAIManager 并行处理 => 同样支持 resume => 只处理尚未完成的 custom_id
    """

    def __init__(self, env_name: str, objective: str, hf_uploader: HuggingFaceUploader):
        """
        :param env_name: Environment 名称，如 "metaworld_soccer-v2"
        :param objective: 对应的任务目标提示，如 "to move the soccer ball into the goal"
        :param hf_uploader: HuggingFaceUploader 实例，用于上传图片到 HF
        """
        self.env_name = env_name
        self.objective = objective
        self.hf_uploader = hf_uploader

    def generate_pairs(
        self,
        selected_views: Optional[List[str]] = None,
        max_pairs: Optional[int] = None,
        random_seed: Optional[int] = 42
    ) -> List[Tuple[str,str,str]]:
        """
        收集本环境下所有图像，并按照点云对象组合后随机选择一个视角对。
        比如：点云1有view1,view2；点云2有view1,view2 => 4种组合 => 随机选1种。

        :param selected_views: 若指定，只处理这些视角；否则收集全部视角。
        :param max_pairs: 若指定，则在生成完所有组合后随机打乱只保留前 max_pairs 条
        :param random_seed: 控制随机选择的种子，以保证可复现性
        :return: 形如 [(fileA, fileB, chosen_views_info), ...]
        """
        random.seed(random_seed)

        env_dir = os.path.join(RENDER_ROOT_DIR, self.env_name)
        if not os.path.isdir(env_dir):
            print(f"[EnvProc] [{self.env_name}] 目录不存在：{env_dir}")
            return []

        # 字典: { 点云idx: [(view, filename), ...] }
        cloud_dict: Dict[int, List[Tuple[str, str]]] = {}

        # 收集符合条件的 jpg 文件
        for f in os.listdir(env_dir):
            if not f.lower().endswith(".jpg"):
                continue
            fullp = os.path.join(env_dir, f)
            if not os.path.isfile(fullp):
                continue
            match = re.match(FILENAME_REGEX, f)
            if match:
                idx = int(match.group(1))
                vw = match.group(2)
                if selected_views and vw not in selected_views:
                    continue
                cloud_dict.setdefault(idx, []).append((vw, f))

        all_indices = list(cloud_dict.keys())
        if len(all_indices) < 2:
            print(f"[EnvProc] [{self.env_name}] 点云对象不足以生成组合.")
            return []

        from itertools import combinations
        pairs: List[Tuple[str,str,str]] = []

        # 对每个 pair(i, j) (i<j)，随机选择一种视角组合
        for i, j in combinations(all_indices, 2):
            files_i = cloud_dict.get(i, [])
            files_j = cloud_dict.get(j, [])
            if not files_i or not files_j:
                continue

            possible_combinations = []
            for vw_i, file_i in files_i:
                for vw_j, file_j in files_j:
                    possible_combinations.append(((vw_i, file_i), (vw_j, file_j)))

            chosen = random.choice(possible_combinations)
            (vw_i, file_i), (vw_j, file_j) = chosen
            chosen_views_info = f"({vw_i}, {vw_j})"
            pairs.append((file_i, file_j, chosen_views_info))

        # 若指定 max_pairs => 随机洗牌后截取
        if max_pairs is not None and pairs:
            random.shuffle(pairs)
            pairs = pairs[:max_pairs]

        print(f"[EnvProc] [{self.env_name}] 生成 {len(pairs)} 对图像组合.")
        return pairs

    def create_big_jsonl(self, pairs: List[Tuple[str,str,str]], out_file: str):
        """
        将 pairs 写入 JSONL 文件，后续由 openai batch api 或 direct api 使用。
        :param pairs: [(f1, f2, view_info),...]
        :param out_file: 最终输出的 .jsonl 文件路径
        """
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as outf:
            for line_idx, (f1, f2, _) in enumerate(pairs, start=1):
                lp1 = os.path.join(RENDER_ROOT_DIR, self.env_name, f1)
                lp2 = os.path.join(RENDER_ROOT_DIR, self.env_name, f2)

                # 上传到 HF
                url1 = self.hf_uploader.upload_image(self.env_name, lp1)
                url2 = self.hf_uploader.upload_image(self.env_name, lp2)

                # 生成 user_content
                user_content = single_round_template.format(
                    img1_url=url1,
                    img2_url=url2,
                    objective=self.objective
                )
                custom_id = f"{self.env_name}-{line_idx}"

                data = {
                    "custom_id": custom_id,
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
                outf.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"[EnvProc] [{self.env_name}] 生成大 JSONL: {out_file}, 共 {len(pairs)} 条")

    def chunk_file(self, file_path: str, chunk_size: int) -> List[str]:
        """
        将 file_path (JSONL) 拆分为若干 chunk_size 行的小文件。
        若行数 <= chunk_size，返回 [file_path]。
        """
        with open(file_path, "r", encoding="utf-8") as ff:
            lines = ff.readlines()
        total_lines = len(lines)
        if total_lines <= chunk_size:
            return [file_path]

        base_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        chunk_files = []
        start=0
        idx=0
        while start<total_lines:
            end = min(start+chunk_size, total_lines)
            sub_lines = lines[start:end]
            chunk_path = os.path.join(base_dir, f"{base_name}_chunk{idx}.jsonl")
            with open(chunk_path, "w", encoding="utf-8") as subf:
                subf.writelines(sub_lines)
            chunk_files.append(chunk_path)
            idx+=1
            start+=chunk_size
        return chunk_files

    def merge_outputs(self, chunk_out_files: List[str], final_out: str):
        """
        将若干 chunk_out_files 的输出合并到 final_out. 
        按 custom_id 的末尾序号排序，以保持一定顺序。
        """
        all_data=[]
        for cof in chunk_out_files:
            if not os.path.isfile(cof):
                continue
            with open(cof, "r", encoding="utf-8") as ff:
                for line in ff:
                    obj = json.loads(line)
                    all_data.append(obj)

        def parse_line_idx(custom_id:str)->int:
            if "-" not in custom_id:
                return 999999
            try:
                return int(custom_id.rsplit("-",1)[-1])
            except:
                return 999999

        all_data.sort(key=lambda x: parse_line_idx(x.get("custom_id","")))
        os.makedirs(os.path.dirname(final_out), exist_ok=True)
        with open(final_out,"w",encoding="utf-8") as out_f:
            for obj in all_data:
                out_f.write(json.dumps(obj, ensure_ascii=False)+"\n")

        print(f"[EnvProc] [{self.env_name}] 合并输出: {final_out}, 共 {len(all_data)} 条")

    def test_chunk_size(self, big_json_file: str, chunk_size: int) -> bool:
        """
        测试给定 chunk_size 能否成功处理 (不触发 token_limit_exceeded 等无法继续的错误)。
        - 采用 test_mode=True 的 MultiOpenAIBatchManager:
          遇到失败后不会无限重试 => 直接返回 False。
        - 若全部 chunk 都成功 => True
        """
        cfiles = self.chunk_file(big_json_file, chunk_size)
        # 构建 resume_map (若已经有输出文件，则可以跳过，但测试阶段也可不做)
        # 这里为了简单，resume_map先为空
        manager = MultiOpenAIBatchManager(test_mode=True, resume_map={})
        manager.load_handlers(OPENAI_API_KEYS)

        temp_dir = os.path.join(os.path.dirname(big_json_file), "temp_test_dir")
        os.makedirs(temp_dir, exist_ok=True)

        success = manager.process_chunk_files_test(cfiles, temp_dir)
        return success

    def find_optimal_chunk_size(self, big_json_file:str) -> int:
        """
        对 chunk_size 在 [CHUNK_SIZE_MIN, CHUNK_SIZE_MAX] 做二分搜索,
        测试可行性(避免 token_limit_exceeded)。
        成功则尝试更大, 失败则减小。
        """
        low = CHUNK_SIZE_MIN
        high = CHUNK_SIZE_MAX
        best_ok = CHUNK_SIZE_MIN  # 默认至少 min 是可行

        if low != high: # 如果一样则说明不需要搜索
            while low<=high:
                mid = (low+high)//2
                print(f"[EnvProc] [{self.env_name}] 正在测试 chunk_size={mid}")
                ok = self.test_chunk_size(big_json_file, mid)
                if ok:
                    best_ok = mid
                    low = mid+1
                else:
                    high = mid-1

        print(f"[EnvProc] [{self.env_name}] 二分搜索结束, 最优 chunk_size={best_ok}")
        return best_ok

    def _build_resume_map(self, chunk_files: List[str]) -> Dict[str, str]:
        """
        针对 chunk_files，每个 chunk 若已经有对应 output.jsonl 则读取其中 data，
        将 chunk_file -> output_file 存于 map 以跳过重复提交。
        """
        resume_map = {}
        for cf in chunk_files:
            # 例如 chunk_file='..._chunk0.jsonl', output='..._chunk0_output.jsonl'
            outp = os.path.splitext(cf)[0] + "_output.jsonl"
            if os.path.isfile(outp):
                # 说明此前已完成下载 => 建立映射
                resume_map[cf] = outp
        return resume_map

    def process_environment(
        self,
        selected_views: Optional[List[str]] = None,
        max_pairs: Optional[int] = None
    ):
        """
        完整处理流程：
        1) generate_pairs => 生成 pairs
        2) create_big_jsonl => 生成大 JSONL
        3) 判断是 Batch 模式还是 Direct 模式 => 分别走后续逻辑(含 resume).
        """
        print(f"\n=== [EnvProc] 开始处理环境: {self.env_name} ===")

        # 1) 生成 pairs
        pairs = self.generate_pairs(selected_views, max_pairs)
        if not pairs:
            print(f"[EnvProc] [{self.env_name}] 无可用 pairs, 结束.")
            return

        # 2) 生成大 JSONL
        ts = time.strftime("%Y%m%d-%H%M%S")
        base_dir = os.path.join("dataCollection","Dataset", self.env_name)
        ts_dir = os.path.join(base_dir, ts)
        os.makedirs(ts_dir, exist_ok=True)

        big_json_file = os.path.join(ts_dir, "batch_input_all.jsonl")
        self.create_big_jsonl(pairs, big_json_file)

        # 3) 上传完图像后 flush 缓冲
        self.hf_uploader.finalize(self.env_name)

        # 4) 分模式进行处理
        if USE_BATCH_API:
            # --- Batch API 模式 ---
            best_chunk = self.find_optimal_chunk_size(big_json_file)
            chunk_files = self.chunk_file(big_json_file, best_chunk)
            if len(chunk_files)==1:
                print(f"[EnvProc] [{self.env_name}] 仅需1个 chunk, 不再细分.")
            else:
                print(f"[EnvProc] [{self.env_name}] 用 chunk_size={best_chunk}, 生成 {len(chunk_files)} 个 chunk.")

            # 构建 resume_map: 若之前部分 chunk 已成功下载, 则不再重复提交
            resume_map = self._build_resume_map(chunk_files)

            manager = MultiOpenAIBatchManager(test_mode=False, resume_map=resume_map)
            manager.load_handlers(OPENAI_API_KEYS)

            chunk_outs = manager.process_chunk_files_official(chunk_files, ts_dir)
            final_out = os.path.join(ts_dir, "batch_output_merged.jsonl")
            self.merge_outputs(chunk_outs, final_out)

            print(f"=== [EnvProc] [{self.env_name}] Batch API 处理完毕, 最终输出 => {final_out} ===")

        else:
            # --- Direct API 模式 ---
            direct_manager = DirectOpenAIManager()
            direct_manager.load_handlers(OPENAI_API_KEYS)

            # 读取大 JSONL => tasks
            tasks = []
            with open(big_json_file, "r", encoding="utf-8") as f:
                for line in f:
                    task = json.loads(line)
                    tasks.append(task)

            final_out = os.path.join(ts_dir, "direct_api_results.jsonl")

            # 并行处理 => 可从 final_out 进行 resume
            results = direct_manager.process_tasks(tasks, resume_file=final_out)

            # 将本轮结果写入 final_out(若里面原本已存在的会被保留+追加)
            # 注意：由于可以多次追加，需要合并去重
            old_records = {}
            if os.path.isfile(final_out):
                with open(final_out, "r", encoding="utf-8") as oldf:
                    for line in oldf:
                        try:
                            obj = json.loads(line)
                            cid = obj.get("custom_id","")
                            old_records[cid] = obj
                        except:
                            pass

            # 更新 old_records 中的数据
            for cid, resp in results.items():
                old_records[cid] = {"custom_id": cid, "response": resp}

            # 写回
            with open(final_out, "w", encoding="utf-8") as outf:
                for cid in sorted(old_records.keys()):
                    outf.write(json.dumps(old_records[cid], ensure_ascii=False)+"\n")

            print(f"=== [EnvProc] [{self.env_name}] Direct API 处理完毕, 结果已保存到 {final_out} ===")
