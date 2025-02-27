#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rewrite_jsonl_manager.py

本模块负责根据待改写文本任务构造 JSONL 请求文件，每行一个 JSON 对象，
用于调用 GPT‑4o 完成链式思考改写任务。每个任务的 payload 包含：
  - 系统提示（SYSTEM_PROMPT_TEXT），指示生成包含 <think>…</think> 的链式思考过程；
  - 用户消息模板（USER_MESSAGE_TEMPLATE），其中 {input_text} 占位符将替换为具体文本。

同时，本模块支持：
  - 将大 JSONL 文件按行分块（适用于批处理模式）
  - 构建断点续传映射
  - 合并输出文件（按 custom_id 排序）
  - Direct 模式结果的合并
  - 将改写结果合并回原始数据集，并输出新的 JSON 文件
"""

import os
import json
import copy
import re
from config import SYSTEM_PROMPT_TEXT, MODEL, USER_MESSAGE_TEMPLATE
from utils import log

class RewriteJSONLManager:
    def __init__(self, tasks):
        """
        :param tasks: 待改写文本任务列表，每个元素为 {"custom_id": ..., "input_text": ...}
        """
        self.tasks = tasks

    def create_jsonl(self, out_file: str) -> None:
        """
        根据任务构造 JSONL 请求文件，每行一个完整的请求 payload。
        """
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as outf:
            for task in self.tasks:
                custom_id = task["custom_id"]
                input_text = task["input_text"]
                user_message = []
                for item in USER_MESSAGE_TEMPLATE:
                    item_copy = copy.deepcopy(item)
                    if "text" in item_copy:
                        item_copy["text"] = item_copy["text"].replace("{input_text}", input_text)
                    user_message.append(item_copy)
                data = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": SYSTEM_PROMPT_TEXT}]
                            },
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ],
                        "max_tokens": 2000
                    }
                }
                outf.write(json.dumps(data, ensure_ascii=False) + "\n")
        log(f"[RewriteJSONLManager] 已生成 JSONL 文件：{out_file}，共 {len(self.tasks)} 条任务。")

    def chunk_file(self, file_path: str, chunk_size: int) -> list:
        """
        将大 JSONL 文件按指定行数分块，返回子文件路径列表。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        if total <= chunk_size:
            return [file_path]
        base_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_files = []
        idx = 0
        for start in range(0, total, chunk_size):
            chunk_lines = lines[start:start+chunk_size]
            chunk_file_path = os.path.join(base_dir, f"{base_name}_chunk{idx}.jsonl")
            with open(chunk_file_path, "w", encoding="utf-8") as cf:
                cf.writelines(chunk_lines)
            out_files.append(chunk_file_path)
            idx += 1
        log(f"[RewriteJSONLManager] 分块生成 {len(out_files)} 个子文件。")
        return out_files

    def build_resume_map(self, chunk_files: list) -> dict:
        """
        构建断点续传映射：对于每个子文件，若存在对应的输出文件则记录映射关系。
        """
        resume_map = {}
        for cf in chunk_files:
            outp = os.path.splitext(cf)[0] + "_output.jsonl"
            if os.path.isfile(outp):
                resume_map[cf] = outp
        return resume_map

    def load_tasks_from_file(self, file_path: str) -> list:
        """
        从 JSONL 文件中加载任务，每行一个 JSON 对象，返回任务列表。
        """
        tasks = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    tasks.append(json.loads(line))
                except Exception as e:
                    log(f"[RewriteJSONLManager] 解析任务行失败：{e}")
        return tasks

    def merge_outputs(self, chunk_out_files: list, final_out: str) -> None:
        """
        将多个输出文件合并为一个，并根据 custom_id 排序后写入 final_out 文件。
        """
        merged = []
        for cf in chunk_out_files:
            if not os.path.isfile(cf):
                continue
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        merged.append(json.loads(line))
                    except Exception as e:
                        log(f"[RewriteJSONLManager] 解析行失败：{e}")
        def parse_idx(custom_id: str) -> int:
            try:
                # 假设 custom_id 格式为 fileName_index_conversationIndex
                return int(custom_id.split("_")[-1])
            except:
                return 999999
        merged.sort(key=lambda x: parse_idx(x.get("custom_id", "")))
        os.makedirs(os.path.dirname(final_out), exist_ok=True)
        with open(final_out, "w", encoding="utf-8") as outf:
            for item in merged:
                outf.write(json.dumps(item, ensure_ascii=False) + "\n")
        log(f"[RewriteJSONLManager] 已合并输出文件：{final_out}，共 {len(merged)} 条记录。")

    def merge_direct_results(self, results_map: dict, final_out: str) -> None:
        """
        将 Direct 模式返回的结果合并后写入 final_out 文件。
        """
        merged_results = {}
        if final_out and os.path.isfile(final_out):
            with open(final_out, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        obj = json.loads(line)
                        cid = obj.get("custom_id")
                        if cid:
                            merged_results[cid] = obj
                    except:
                        pass
        for cid, resp in results_map.items():
            merged_results[cid] = {"custom_id": cid, "response": resp}
        def parse_idx(custom_id: str) -> int:
            try:
                return int(custom_id.split("_")[-1])
            except:
                return 999999
        sorted_ids = sorted(merged_results.keys(), key=lambda c: parse_idx(c))
        os.makedirs(os.path.dirname(final_out), exist_ok=True)
        with open(final_out, "w", encoding="utf-8") as wf:
            for cid in sorted_ids:
                wf.write(json.dumps(merged_results[cid], ensure_ascii=False) + "\n")
        log(f"[RewriteJSONLManager] Direct 模式合并结果完成，输出文件：{final_out}")

    def merge_results_into_dataset(self, result_file: str, original_data: list, output_json: str) -> None:
        """
        将改写结果合并回原始数据集。读取 result_file 中的结果，
        并将每个任务的改写结果（response 字段）映射到原始数据中对应的 "gpt" conversation，
       在该 conversation 中增加新字段 "rewritten" 保存改写后的文本。
        最后将合并后的数据输出为新的 JSON 文件。
        """
        # 构建 custom_id -> rewritten_text 的映射（假设返回结果中在 response 内的 choices[0].message.content）
        rewritten_map = {}
        with open(result_file, "r", encoding="utf-8") as rf:
            for line in rf:
                try:
                    obj = json.loads(line)
                    cid = obj.get("custom_id")
                    response = obj.get("response", {})
                    if response and "choices" in response and len(response["choices"]) > 0:
                        content = response["choices"][0]["message"]["content"]
                        rewritten_map[cid] = content
                except Exception as e:
                    log(f"[RewriteJSONLManager] 解析改写结果失败：{e}")
        # 将改写结果写入原始数据中
        for idx, item in enumerate(original_data):
            convs = item.get("conversations", [])
            for j, conv in enumerate(convs):
                if conv.get("from") == "gpt":
                    custom_id = f"{os.path.splitext(os.path.basename(result_file))[0].replace('_output_merged','')}_{idx}_{j}"
                    if custom_id in rewritten_map:
                        conv["rewritten"] = rewritten_map[custom_id]
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as wf:
            json.dump(original_data, wf, ensure_ascii=False, indent=2)
        log(f"[RewriteJSONLManager] 合并结果已保存至 {output_json}")
