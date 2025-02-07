#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py

程序入口文件。执行流程：
  1. 初始化 HuggingFaceUploader（用于上传图像至 Hugging Face 仓库）。
  2. 初始化 APIKeyManager，用于管理多个 Gemini API KEY 及项目 ID。
  3. 针对配置中的各环境，依次调用 EnvironmentProcessor 进行样本数据加载、样本对生成、任务请求构造以及图像上传，
     并调用 Gemini API 处理任务（自动选择 Batch 或 Direct 模式）。
  4. 输出所有环境处理完毕后的提示信息。

注：本代码专为美国使用场景设计，全部采用 Google 官方 Gemini API 端点，并使用官方 Python SDK（google-generativeai）。
"""

import time
import logging
from config import objective_env_prompts, MAXPAIRS, GEMINI_API_KEY, GEMINI_PROJECT_ID
from uploader import HuggingFaceUploader
from environment_processor import EnvironmentProcessor
from api_key_manager import APIKeyManager

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def main():
    """
    主函数入口，遍历所有环境进行数据处理与任务构造。
    """
    logging.info("开始执行 main.py")
    # 初始化 Hugging Face 图像上传实例
    hf_uploader = HuggingFaceUploader()
    # 初始化 APIKeyManager（支持多个 API KEY 与项目 ID）
    key_manager = APIKeyManager(api_keys=GEMINI_API_KEY, project_ids=GEMINI_PROJECT_ID)
    # 针对每个环境，依次调用环境处理器
    for env_name, objective in objective_env_prompts.items():
        logging.info(f"开始处理环境：{env_name}，目标：{objective}")
        processor = EnvironmentProcessor(env_name, objective, hf_uploader, key_manager=key_manager, flip_pair_order=False)
        processor.process_environment(max_pairs=MAXPAIRS)
        time.sleep(3)
    logging.info("所有环境处理完毕。")

if __name__ == "__main__":
    main()
