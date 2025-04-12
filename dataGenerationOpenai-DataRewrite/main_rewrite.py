#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_rewrite.py

项目入口文件：
  1. 从 DATA_REWRITE_DIR 目录加载需要处理的 JSON 数据集，
     例如 "cap3d_objaverse_sft_45k.json" 或 "gapartnet_sft_27k_openai.json"。
  2. 对每个数据集，提取所有 {"from": "gpt", "value": ...} 中的文本数据作为待改写文本。
  3. 根据这些文本构造 JSONL 请求文件（每行一个任务 payload），要求 GPT‑4o 进行链式思考改写，
     输出格式为包含 <think> ... </think> 的链式思考文本，以及最终改写结果。
  4. 根据配置选择 Direct API 模式或 Batch API 模式调用 OpenAI 接口，
     并将结果合并输出到对应的结果文件中。
  5. 当 USE_BATCH_API=False 时，将改写结果合并回原始数据并生成新的 JSON 文件；
     当 USE_BATCH_API=True 时，则不自动生成最终的 rewritten JSON 文件（批处理模式下合并由外部工具处理）。
  6. 所有输出文件均存放于带有当前时间戳的子文件夹中，便于断点续传及区分测试批次。
     合并时使用原始 JSON 数据文件的基本名构造 custom_id（格式为 "{base_name}_{item_index}_{conversation_index}"），
     确保正确匹配改写结果。
"""
import os
import time
from config import DATA_REWRITE_DIR, DATA_FILES, OUTPUT_ROOT_DIR, USE_BATCH_API, CHUNK_SIZE, OPENAI_API_KEYS
from rewrite_data_loader import RewriteDataLoader
from rewrite_jsonl_manager import RewriteJSONLManager
from openai_client import DirectOpenAIManager, MultiOpenAIBatchManager
from utils import log

def get_timestamp_folder(base_dir):
    """
    根据当前时间生成时间戳字符串，并返回以时间戳为名的子目录路径。
    """
    # timestamp = time.strftime("%Y%m%d-%H%M%S")

    # gapartnet_sft_27k_openai断点续传
    # timestamp = "20250301-192944"

    # cap3d_objaverse_sft_45k断点续传
    timestamp = "20250304-114301"

    # cap3d_existed760k断点续传
    # timestamp = "20250301-230548"

    folder = os.path.join(base_dir, timestamp)
    os.makedirs(folder, exist_ok=True)
    return folder

def process_dataset(file_name, output_dir):
    log(f"处理数据集：{file_name}")
    file_path = os.path.join(DATA_REWRITE_DIR, file_name)
    loader = RewriteDataLoader(file_path)
    tasks, original_data = loader.load_rewrite_tasks()
    if not tasks:
        log(f"数据集 {file_name} 中无待处理文本，跳过。")
        return
    log(f"数据集 {file_name} 中共找到 {len(tasks)} 条待改写文本。")
    # 生成 JSONL 请求文件
    jsonl_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_input.jsonl")
    jsonl_mgr = RewriteJSONLManager(tasks)
    jsonl_mgr.create_jsonl(jsonl_file)
    
    if USE_BATCH_API:
        log("使用 Batch API 模式。")
        chunk_files = jsonl_mgr.chunk_file(jsonl_file, CHUNK_SIZE)
        resume_map = jsonl_mgr.build_resume_map(chunk_files)
        batch_manager = MultiOpenAIBatchManager(resume_map=resume_map)
        batch_manager.load_handlers(OPENAI_API_KEYS)
        out_files = batch_manager.process_chunk_files(chunk_files, output_dir)
        final_out = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_output_merged.jsonl")
        jsonl_mgr.merge_outputs(out_files, final_out)
        log(f"Batch 模式处理完成，输出文件：{final_out}")
        log("Batch 模式下不自动生成最终 rewritten JSON 文件，请在任务完成后单独执行合并。")
    else:
        log("使用 Direct API 模式。")
        direct_manager = DirectOpenAIManager()
        direct_manager.load_handlers(OPENAI_API_KEYS)
        tasks_payload = jsonl_mgr.load_tasks_from_file(jsonl_file)
        final_out = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_direct_results.jsonl")
        results_map = direct_manager.process_tasks(tasks_payload, resume_file=final_out)
        jsonl_mgr.merge_direct_results(results_map, final_out)
        log(f"Direct 模式处理完成，输出文件：{final_out}")
        output_json = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_rewritten.json")
        base_name = os.path.splitext(file_name)[0]
        jsonl_mgr.merge_results_into_dataset(final_out, original_data, output_json, base_name)
        log(f"数据集 {file_name} 改写结果合并完成，输出文件：{output_json}")

def main():
    try:
        log("开始处理重写数据集...")
        timestamp_dir = get_timestamp_folder(OUTPUT_ROOT_DIR)
        for file_name in DATA_FILES:
            process_dataset(file_name, timestamp_dir)
        log("所有数据集处理完毕。")
    except Exception as e:
        log(f"程序异常：{e}")

if __name__ == "__main__":
    main()
