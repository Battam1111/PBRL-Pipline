#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py

主入口文件，用于批量评估数据文件或运行示例评估任务，同时记录输入给 GPT 的信息日志（包含时间戳及数据文件名/标识信息）。

功能说明：
  1. 当 data_file 指定了数据文件路径时：
     - 加载 JSON 数据文件，每个评估实例包含 question_id、prompt、answer_gt、text 等字段；
     - 根据预先在代码中指定的评估器类型（变量 EVALUATOR_TYPE）创建对应评估器；
     - 对每个实例调用评估器，记录发送给 GPT 的完整 Prompt 信息，并将所有评估结果保存至 JSON 文件中；
     - 同时将所有发送给 GPT 的 Prompt 日志保存到本地，文件名中包含数据文件名与当前时间戳。

  2. 若 data_file 为空，则运行示例评估任务，展示各评估器调用方法及效果，并保存 demo 模式日志。

注意：
  - 所有日志均输出到控制台，评估过程中打印当前进度及 question_id；
  - 评估结果通常为模型返回的文本（建议为 JSON 格式字符串）。

重要提示：
  - 评估器类型由变量 EVALUATOR_TYPE 指定，不通过命令行参数传入，可在代码中直接修改其值。
    可选取值包括：
      "conclusion_withthinktag"
      "cot_withthinktag"
"""

import os
import sys
import json
import logging

from evaluator import (
    ConclusionEvaluatorWithThink,
    ConclusionEvaluatorWithoutThink,
    CotEvaluatorWithThink,
    CotEvaluatorWithoutThink
)
from prompt_logger import PromptLogger

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =====================【【 在此处直接指定评估器类型 】】=====================
# 请直接在代码中修改以下变量，不使用命令行参数！
# 可选取值：
#   "conclusion_withthinktag"
#   "cot_withthinktag"
EVALUATOR_TYPE = "conclusion_withthinktag"
# EVALUATOR_TYPE = "cot_withthinktag"
# =========================================================================

def create_evaluator(evaluator_type: str, model: str = "chatgpt-4o-latest"):
    """
    根据指定的 evaluator_type 创建对应的评估器实例。
    
    参数:
        evaluator_type (str): 评估器类型，直接在代码中指定。
        model (str): 使用的模型名称。
    
    返回:
        evaluator 对象：对应评估器实例。
    """
    if evaluator_type == "conclusion_withthinktag":
        return ConclusionEvaluatorWithThink(model=model)
    elif evaluator_type == "conclusion_withoutthinktag":
        return ConclusionEvaluatorWithoutThink(model=model)
    elif evaluator_type == "cot_withthinktag":
        return CotEvaluatorWithThink(model=model)
    elif evaluator_type == "cot_nothinktag":
        return CotEvaluatorWithoutThink(model=model)
    else:
        # 默认使用 CoT without thinktag 评估器
        return CotEvaluatorWithoutThink(model=model)

def process_evaluation_file(file_path: str, evaluator, prompt_logger: PromptLogger = None) -> list:
    """
    加载 JSON 数据文件，逐条执行评估，并收集结果，同时记录每次发送给 GPT 的完整 Prompt 信息。

    参数:
        file_path (str): 数据文件路径。
        evaluator: 评估器对象，必须提供 evaluate(textual_query, reference_answer, llm_response, question_id, prompt_logger) 方法。
        prompt_logger (PromptLogger): GPT 输入日志记录器，用于保存每次发送给 GPT 的 Prompt 信息。

    返回:
        list: 评估结果列表，每个元素为字典，包含 question_id 及评估结果。
    """
    results = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        sys.exit(1)
    
    total = len(data)
    logger.info(f"加载到 {total} 个评估实例。")
    
    for i, item in enumerate(data, start=1):
        question_id = item.get("question_id", f"unknown_{i}")
        textual_query = item.get("prompt", "")
        reference_answer = item.get("answer_gt", "")
        llm_response = item.get("text", "")
        logger.info(f"【{i}/{total}】评估实例，Question ID: {question_id}")
        try:
            evaluation_result = evaluator.evaluate(
                textual_query, reference_answer, llm_response,
                question_id=question_id, prompt_logger=prompt_logger
            )
        except Exception as e:
            logger.error(f"评估实例 {question_id} 失败: {e}")
            evaluation_result = "评估失败"
        results.append({
            "question_id": question_id,
            "evaluation_result": evaluation_result
        })
    return results

def save_results(results: list, output_path: str = "evaluation_results.json"):
    """
    将评估结果保存到 JSON 文件中。

    参数:
        results (list): 评估结果列表。
        output_path (str): 输出文件路径，默认为 "evaluation_results.json"。
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"评估结果已保存至 {output_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def run_demo():
    """
    运行示例评估任务，展示各评估器调用方法及效果，同时记录 demo 模式下的 GPT 输入日志。
    """
    textual_query = "请描述点云数据中的主要物体及其功能。"
    reference_answer = "参考答案描述了点云中存在一个桌子，并指出其主要用于支撑物品。"
    llm_response = "模型回答总结：点云中显示出一张桌子，其主要作用是放置物品。"
    
    # 使用 "demo" 作为源文件名标识，记录 demo 模式下的 GPT 输入日志
    demo_logger = PromptLogger(source_file="demo")
    
    logger.info("==== 带 <think> 标记的结论评估 ====")
    evaluator1 = ConclusionEvaluatorWithThink(model="gpt-3.5-turbo")
    result1 = evaluator1.evaluate(
        textual_query, reference_answer, llm_response,
        question_id="demo_1", prompt_logger=demo_logger
    )
    logger.info(result1)
    
    logger.info("==== 不带 <think> 标记的结论评估 ====")
    evaluator2 = ConclusionEvaluatorWithoutThink(model="gpt-3.5-turbo")
    result2 = evaluator2.evaluate(
        textual_query, reference_answer, llm_response,
        question_id="demo_2", prompt_logger=demo_logger
    )
    logger.info(result2)
    
    logger.info("==== 带 <think> 标记的 CoT 评估 ====")
    evaluator3 = CotEvaluatorWithThink(model="gpt-3.5-turbo")
    result3 = evaluator3.evaluate(
        textual_query, reference_answer, llm_response,
        question_id="demo_3", prompt_logger=demo_logger
    )
    logger.info(result3)
    
    logger.info("==== 不带 <think> 标记的 CoT 评估 ====")
    evaluator4 = CotEvaluatorWithoutThink(model="gpt-3.5-turbo")
    result4 = evaluator4.evaluate(
        textual_query, reference_answer, llm_response,
        question_id="demo_4", prompt_logger=demo_logger
    )
    logger.info(result4)
    
    # 保存 demo 模式下的 GPT 输入日志
    demo_logger.save_logs()

def main():
    """
    主入口函数：
      1. 若 data_file 指定了数据文件路径，则加载数据文件并批量评估；
      2. 否则运行示例评估任务。
    同时记录所有发送给 GPT 的 Prompt 信息（包含时间戳及数据文件名/标识）。
    """
    # 数据文件路径（请根据需要修改为实际路径；若不需要批量评估，可设为 None）
    # data_file = "gptEvaluation/data/llama31r1_8b_cot_think/cap3d_thinkmark_test.json"

    # data_file = "gptEvaluation/data/llama31r1_8b_cot_unthink/cap3d_thinkmark_test.json"

    # data_file = "gptEvaluation/data/llama31_8b_nocot/cap3d_thinkmark_test.json"

    # data_file = "gptEvaluation/data/llama31_8b_cot_think/cap3d_thinkmark_test.json"

    data_file = "gptEvaluation/data/llama31_8b_cot_unthink/cap3d_thinkmark_test.json"

    relative_path = os.path.relpath(data_file, "gptEvaluation/data")

    relative_name = os.path.splitext(relative_path)[0].replace(os.path.sep, "_")

    # 拼接输出目录和文件名（也可以在这里加上时间戳）
    output_file = os.path.join("gptEvaluation/result", f"{relative_name}_{EVALUATOR_TYPE}.json")

    if data_file and os.path.exists(data_file):
        logger.info(f"加载评估数据文件: {data_file}")
        evaluator = create_evaluator(EVALUATOR_TYPE, model="chatgpt-4o-latest")
        # 使用数据文件名作为 GPT 输入日志的标识
        file_logger = PromptLogger(source_file=os.path.basename(data_file))
        results = process_evaluation_file(data_file, evaluator, prompt_logger=file_logger)
        save_results(results, output_path=output_file)
        file_logger.save_logs()
    else:
        logger.info("未提供有效数据文件路径，运行示例评估任务。")
        run_demo()

if __name__ == "__main__":
    main()
