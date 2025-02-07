#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py

程序入口文件，示例执行流程：
  1. 初始化 HuggingFaceUploader（用于批量上传图像至 Hugging Face 仓库）
  2. 对配置中的各环境，依次调用 EnvironmentProcessor 进行样本数据加载、样本对生成、任务构建及图像上传
  3. 根据配置选择使用 Batch API 或 Direct API 处理 OpenAI 任务，并最终合并所有任务结果
  4. 输出全部环境处理完成后的提示信息

注：每个环境处理完毕后会休眠一定时间，以避免服务器压力过大。
"""

import time
from config import objective_env_prompts, MAXPAIRS
from uploader import HuggingFaceUploader
from environment_processor import EnvironmentProcessor

def main():
    """
    主函数入口，遍历所有环境进行处理
    """
    # 初始化 Hugging Face Uploader 实例
    hf_uploader = HuggingFaceUploader()
    # 对每个环境进行处理
    for env_name, objective in objective_env_prompts.items():
        processor = EnvironmentProcessor(env_name, objective, hf_uploader, flip_pair_order=False)
        # 示例：限制最多生成100个样本对，可根据需要调整
        processor.process_environment(max_pairs=MAXPAIRS)
        time.sleep(3)
    print("\n[完成] 全部环境处理完毕。")

if __name__ == "__main__":
    main()
