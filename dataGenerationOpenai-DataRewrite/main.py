#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py

项目入口文件：
  1. 加载输入文本样本（每行一个文本）。
  2. 根据文本构造 JSONL 请求文件，要求 GPT‑4o 进行链式思考改写。
  3. 根据配置选择 Direct API 模式或 Batch API 模式调用 OpenAI 接口，
     并将结果合并输出到结果文件中。
"""

import os
import time
from config import TEXT_INPUT_FILE, OUTPUT_DIR, USE_BATCH_API, CHUNK_SIZE, OPENAI_API_KEYS
from text_loader import TextLoader
from jsonl_manager import JSONLManager
from openai_client import DirectOpenAIManager, MultiOpenAIBatchManager
from utils import log

def main():
    try:
        # 1. 加载输入文本样本
        log("加载输入文本样本...")
        loader = TextLoader(TEXT_INPUT_FILE)
        texts = loader.load_texts()
        if not texts:
            log("无输入文本样本，程序终止。")
            return
        log(f"共加载 {len(texts)} 个文本样本。")
        
        # 2. 生成 JSONL 请求文件（每行一个任务 payload）
        log("生成 JSONL 请求文件...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        jsonl_file = os.path.join(OUTPUT_DIR, "batch_input_all.jsonl")
        jsonl_mgr = JSONLManager(texts)
        jsonl_mgr.create_jsonl(jsonl_file)
        
        # 3. 根据调用模式选择处理逻辑
        if USE_BATCH_API:
            log("使用 Batch API 模式。")
            chunk_files = jsonl_mgr.chunk_file(jsonl_file, CHUNK_SIZE)
            resume_map = jsonl_mgr.build_resume_map(chunk_files)
            batch_manager = MultiOpenAIBatchManager(resume_map=resume_map)
            batch_manager.load_handlers(OPENAI_API_KEYS)
            out_files = batch_manager.process_chunk_files(chunk_files, OUTPUT_DIR)
            final_out = os.path.join(OUTPUT_DIR, "batch_output_merged.jsonl")
            jsonl_mgr.merge_outputs(out_files, final_out)
            log(f"Batch 模式完成，输出文件：{final_out}")
        else:
            log("使用 Direct API 模式。")
            direct_manager = DirectOpenAIManager()
            direct_manager.load_handlers(OPENAI_API_KEYS)
            tasks = jsonl_mgr.load_tasks_from_file(jsonl_file)
            final_out = os.path.join(OUTPUT_DIR, "direct_api_results.jsonl")
            results_map = direct_manager.process_tasks(tasks, resume_file=final_out)
            jsonl_mgr.merge_direct_results(results_map, final_out)
            log(f"Direct 模式完成，输出文件：{final_out}")
    except Exception as e:
        log(f"程序异常：{e}")

if __name__=="__main__":
    main()
