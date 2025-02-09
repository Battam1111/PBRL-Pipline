#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py

项目入口文件。遍历 config 中配置的各环境，
对每个环境调用 EnvironmentProcessor 进行数据处理、JSONL 构造、图像上传以及 OpenAI API 调用。
"""

import time
from config import objective_env_prompts, MAXPAIRS
from uploader import HuggingFaceUploader
from environment_processor import EnvironmentProcessor

def main():
    try:
        hf_uploader = HuggingFaceUploader()
        for env_name, objective in objective_env_prompts.items():
            print(f"\n[Main] 正在处理环境：{env_name}，目标：{objective}")
            processor = EnvironmentProcessor(
                env_name=env_name,
                objective=objective,
                hf_uploader=hf_uploader,
                flip_pair_order=False  # 根据需要开启或关闭配对顺序翻转
            )
            processor.process_environment(max_pairs=MAXPAIRS)
            time.sleep(3)
        print("\n所有环境处理完毕。")
    except Exception as e:
        print(f"[Main] 报错：{e}")

if __name__=="__main__":
    main()
