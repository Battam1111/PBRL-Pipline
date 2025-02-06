#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py

程序入口文件。执行流程：
  1. 初始化 HuggingFaceUploader（用于上传图像至 Hugging Face 仓库）。
  2. 针对配置中的各环境，依次调用 EnvironmentProcessor 进行样本数据加载、样本对生成、任务请求构造以及图像上传。
  3. 根据配置选择使用 Batch 模式或直连 Direct 模式调用 Gemini API 处理任务，并最终合并所有任务结果。
  4. 输出全部环境处理完毕的提示信息。

注：本代码已删除任何针对中国境内代理的处理，全部使用 Google 官方端点（参考 :contentReference[oaicite:2]{index=2}）。
"""

import time
from config import objective_env_prompts
from uploader import HuggingFaceUploader
from environment_processor import EnvironmentProcessor

def main():
    """
    主函数入口，遍历配置中的所有环境进行数据处理与任务构建
    """
    # 初始化 Hugging Face 图像上传实例（上传逻辑保持不变）
    hf_uploader = HuggingFaceUploader()
    # 针对每个环境，依次调用环境处理器
    for env_name, objective in objective_env_prompts.items():
        processor = EnvironmentProcessor(env_name, objective, hf_uploader, flip_pair_order=False)
        # 例如：限制最多生成10个样本对
        processor.process_environment(max_pairs=10)
        time.sleep(3)
    print("\n[完成] 全部环境处理完毕。")

if __name__ == "__main__":
    main()
