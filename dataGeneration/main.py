# /home/star/Yanjun/RL-VLM-F/dataGeneration/main.py
# -------------------------------------------------------------------------------
"""
main.py

可执行入口:
1. 初始化HF Uploader
2. 对各环境 => EnvironmentProcessor
3. processor.process_environment()
"""

import time
from config import objective_env_prompts, USE_BATCH_API
from huggingface_uploader import HuggingFaceUploader
from environment_processor import EnvironmentProcessor

def main():
    """
    主函数入口。
    - 逐个 environment 进行处理，包括:
      1) 生成可比对的 pairs (多视角随机组合)
      2) 生成大JSONL
      3) 选择 Batch API 或 Direct API 工作模式
      4) 支持自动 resume(若上次已有部分结果，则跳过相应 chunk/对)
    """

    # 初始化 HuggingFace Uploader
    hf_uploader = HuggingFaceUploader()

    for env_name, objective in objective_env_prompts.items():
        processor = EnvironmentProcessor(env_name, objective, hf_uploader)

        # 这里示例使用所有可用视角 => selected_views=None
        # 同时设定 max_pairs=10000 (可根据实际需要调整/或不设)
        processor.process_environment(selected_views=None, max_pairs=10000)

        time.sleep(3)  # 防止连续处理时的速度过快, 做个间隔

    print("\n[完成] 全部环境处理完毕.")


if __name__=="__main__":
    main()
