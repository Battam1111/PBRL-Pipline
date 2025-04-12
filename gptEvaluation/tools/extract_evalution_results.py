#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_evaluation_results.py

本脚本用于提取评测结果 JSON 文件中的有效评测内容。
评测结果文件为 JSON 格式，包含一个列表，每个元素为字典，其中 "evaluation_result" 字段包含带有反引号包裹的 JSON 格式字符串。
该脚本将解析这些字符串并输出一个新的 JSON 文件，其中包含 question_id 与解析后的评测结果对象。

使用方法：
    直接在代码中设置 input_file 与 output_file 变量（可结合 data_file 与 evaluator 类型生成）。
    运行该脚本后，将在指定输出路径生成一个新的 JSON 文件，其中包含解析后的评测结果。

注意：
    - 代码设计高度模块化、低冗余，并添加了充分的异常处理与日志记录，确保鲁棒性与高效率。
"""

import os
import json
import re
import logging

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def extract_json_content(evaluation_str: str) -> dict:
    """
    提取包含在反引号中的 JSON 字符串，并将其解析为 Python 字典。

    参数:
        evaluation_str (str): 包含反引号的评测结果字符串。

    返回:
        dict: 解析后的 JSON 对象。

    如果提取失败，将抛出 ValueError。
    """
    # 使用正则表达式匹配 ```json ... ```
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, evaluation_str, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
    else:
        # 如果没有找到反引号包裹，则尝试直接解析整个字符串
        json_content = evaluation_str.strip()

    try:
        parsed = json.loads(json_content)
    except json.JSONDecodeError as e:
        logger.error("JSON解析失败: %s", e)
        raise ValueError(f"无法解析评测结果字符串: {evaluation_str}") from e

    return parsed

def process_evaluation_results(input_file: str) -> list:
    """
    读取评测结果文件，解析每个评测结果中的 JSON 字符串，并返回一个新的列表。
    
    参数:
        input_file (str): 输入评测结果文件路径。

    返回:
        list: 新的列表，每个元素为字典，包含 question_id 与解析后的 evaluation_result。
    """
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("加载输入文件失败: %s", e)
        raise

    processed_results = []
    for item in data:
        question_id = item.get("question_id", "N/A")
        raw_evaluation = item.get("evaluation_result", "")
        try:
            parsed_evaluation = extract_json_content(raw_evaluation)
        except ValueError as e:
            logger.error("解析 question_id %s 失败: %s", question_id, e)
            parsed_evaluation = None  # 解析失败时可以设为 None 或跳过该条记录

        processed_results.append({
            "question_id": question_id,
            "evaluation_result": parsed_evaluation
        })

    return processed_results

def save_processed_results(results: list, output_file: str):
    """
    将解析后的评测结果保存到指定的输出文件中。

    参数:
        results (list): 解析后的评测结果列表。
        output_file (str): 输出文件路径。
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"处理后的评测结果已保存至 {output_file}")
    except Exception as e:
        logger.error("保存处理后的评测结果失败: %s", e)
        raise

def main():
    """
    主函数，用于执行评测结果提取流程。
    """
    # 在此设置输入文件路径，示例文件名中包含完整的目录路径以及评估器类型信息
    # input_file = "gptEvaluation/result/llama31r1_8b_cot_think_cap3d_thinkmark_test_cot_withthinktag.json"
    # input_file = "gptEvaluation/result/llama31r1_8b_cot_think_cap3d_thinkmark_test_conclusion_withthinktag.json"
    # input_file = "gptEvaluation/result/llama31r1_8b_cot_unthink_cap3d_thinkmark_test_conclusion_withthinktag.json"
    # input_file = "gptEvaluation/result/llama31r1_8b_cot_unthink_cap3d_thinkmark_test_cot_withthinktag.json"
    # input_file = "gptEvaluation/result/llama31_8b_nocot_cap3d_thinkmark_test_conclusion_withthinktag.json"
    # input_file = "gptEvaluation/result/llama31_8b_cot_think_cap3d_thinkmark_test_conclusion_withthinktag.json"
    # input_file = "gptEvaluation/result/llama31_8b_cot_think_cap3d_thinkmark_test_cot_withthinktag.json"
    input_file = ""

    # 生成输出文件路径：
    # 1. 提取输入文件名（去掉目录及扩展名）
    # 2. 在基础文件名后添加 "_parsed"，再加上 .json 扩展名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(os.path.dirname(input_file), f"{base_name}_parsed.json")

    try:
        processed_results = process_evaluation_results(input_file)
    except Exception as e:
        logger.error("处理评测结果时出现错误: %s", e)
        return

    try:
        save_processed_results(processed_results, output_file)
    except Exception as e:
        logger.error("保存处理结果时出现错误: %s", e)

if __name__ == "__main__":
    main()
