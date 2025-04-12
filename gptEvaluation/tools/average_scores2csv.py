#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
average_scores_to_csv.py

本脚本用于从指定文件夹中提取所有 parsed JSON 文件（这些文件中存放着评测结果，如各项指标的分数及说明），
计算每个文件中各项评测指标（例如 Truthfulness、Completeness、Object Recognition、Functional Reasoning、Interaction Prediction）的平均值与标准差，
并将所有结果汇总保存为 CSV 文件。

生成的输出包含两个 CSV 文件：
  1. "average_scores_raw.csv"：按 Model 与 Annotation 聚合后输出，每行记录唯一的 Model、Annotation、Category 及各指标以 "avg ± std" 格式显示；
  2. "average_scores_pivot.csv"：按 Category 和 Annotation 分组后，各指标均值和标准差汇总（以 "avg ± std" 格式显示）。

使用说明：
1. 在代码中指定存放 parsed JSON 文件的文件夹路径（例如：folder_path = "gptEvaluation/result/parsed"）。
2. 脚本会遍历该文件夹下所有 .json 文件，逐个解析文件中的评测结果，计算各项指标的统计数据，并从文件名中提取 evaluator_type，
   同时根据预定义映射规则将 file_name 转换为目标显示名称，并拆分为 Model 与 Annotation，再自动生成 Category 列。
3. 对于 raw CSV，将按 Model 与 Annotation 聚合（即合并同一模型和评注的数据），确保每种情况只输出一行；而 Pivot CSV 按 Category 与 Annotation 分组汇总。
4. 最终在同一文件夹下生成两个 CSV 文件："average_scores_raw.csv" 和 "average_scores_pivot.csv"。

要求：
    - Python 3.x
    - 依赖：pandas（用于生成 CSV 文件），请确保已安装 pandas（如 pip install pandas）。

作者：Your Name
日期：2025-03-07
"""

import os
import json
import math
import logging
from typing import Dict, Any, List, Tuple

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 更新后的文件名映射规则：
# 注意：更具体的映射（如 "llama31r1_8b_cot_think"）应排在前面
FILE_NAME_MAPPING = {
    "llama31r1_8b_cot_think": "DeepSeek-R1-Distill-Llama-8B Tagged CoT",
    "llama31r1_8b_cot_unthink": "DeepSeek-R1-Distill-Llama-8B Unmarked CoT",
    "llama31_8b_nocot": "Llama-3.1-8B-Instruct No CoT",
    "llama31_8b_cot_think": "Llama-3.1-8B-Instruct Tagged CoT"  # 新增，不含 r1 的情况
}

# 定义指标映射，目标列名与原指标名称对应（用于输出列顺序）
METRIC_MAPPING = {
    "Truthfulness": "TRU",
    "Completeness": "COMP",
    "Object Recognition": "OBJ",
    "Functional Reasoning": "FUNC",
    "Interaction Prediction": "INTER"
}

def map_file_name(original_name: str) -> str:
    """
    根据预定义的映射规则，将原始文件名转换为目标显示名称。
    若原始名称中包含某个映射键，则返回对应值；否则返回原始名称。

    参数:
        original_name (str): 原始文件名字符串。

    返回:
        str: 映射后的显示名称。
    """
    for key, mapped in FILE_NAME_MAPPING.items():
        if key in original_name:
            return mapped
    return original_name

def split_model_annotation(mapped_name: str) -> Tuple[str, str]:
    """
    将映射后的文件名拆分为 Model 和 Annotation 两部分。
    假定映射后的名称格式为 "Model Annotation"，其中 Annotation 为最后两个单词（例如 "No CoT"、"Tagged CoT" 或 "Unmarked CoT"）。

    参数:
        mapped_name (str): 映射后的名称。

    返回:
        Tuple[str, str]: (Model, Annotation)
    """
    parts = mapped_name.rsplit(" ", 2)
    if len(parts) >= 3:
        model = parts[0].strip()
        annotation = f"{parts[1].strip()} {parts[2].strip()}"
        return model, annotation
    else:
        return mapped_name, ""

def map_category(model: str) -> str:
    """
    根据 Model 字符串判断所属类别：
      - 若包含 "Llama-3.1-8B-Instruct" 则归为 LLM；
      - 若包含 "DeepSeek-R1-Distill-Llama-8B" 则归为 LRM；
      - 否则返回 "Unknown"。

    参数:
        model (str): 模型名称。

    返回:
        str: 类别 ("LLM", "LRM" 或 "Unknown")
    """
    if "Llama-3.1-8B-Instruct" in model:
        return "LLM"
    elif "DeepSeek-R1-Distill-Llama-8B" in model:
        return "LRM"
    else:
        return "Unknown"

def combine_metric(avg: float, std: float) -> str:
    """
    将平均值和标准差组合为一个字符串，格式为 "avg ± std"（均保留两位小数），
    若任一值为 None，则返回 "—"。

    参数:
        avg (float): 平均值
        std (float): 标准差

    返回:
        str: 组合后的字符串。
    """
    if avg is None or std is None:
        return "—"
    return f"{avg:.2f} ± {std:.2f}"

def compute_metric_stats(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算单个 JSON 文件中各项评测指标的平均值和标准差。

    参数:
        data (List[Dict[str, Any]]): 从 JSON 文件中加载的列表，
            每个元素为一个 dict，其中包含 "evaluation_result" 字段，
            此字段为一个 dict，键为指标名称，值为包含 "Score" 和 "Explanation" 的字典。

    返回:
        Dict[str, float]: 一个字典，其中每个指标生成两个键：
            - "{指标}_avg"：该指标的平均分
            - "{指标}_std"：该指标的标准差（总体标准差，四舍五入至后两位）
    """
    scores_dict: Dict[str, List[float]] = {}

    for item in data:
        evaluation = item.get("evaluation_result")
        if not evaluation:
            continue
        for metric, result in evaluation.items():
            score = result.get("Score")
            try:
                score_value = float(score)
            except (ValueError, TypeError):
                logger.warning("无法转换分数：%s，跳过该分数", score)
                continue
            scores_dict.setdefault(metric, []).append(score_value)

    stats = {}
    for metric, scores in scores_dict.items():
        if scores:
            avg = sum(scores) / len(scores)
            variance = sum((x - avg) ** 2 for x in scores) / len(scores)
            std = math.sqrt(variance)
            stats[f"{metric}_avg"] = avg
            stats[f"{metric}_std"] = round(std, 2)
        else:
            stats[f"{metric}_avg"] = None
            stats[f"{metric}_std"] = None
    return stats

def process_json_file(file_path: str) -> Dict[str, Any]:
    """
    处理单个 JSON 文件，计算其中各项指标的平均值和标准差，并返回包含文件名和结果的字典。

    参数:
        file_path (str): JSON 文件路径。

    返回:
        Dict[str, Any]: 结果字典，包含 "file_name" 及各项指标的统计数据。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("加载文件 %s 失败: %s", file_path, e)
        raise e

    stats = compute_metric_stats(data)
    result = {"file_name": os.path.basename(file_path)}
    result.update(stats)
    return result

def extract_evaluator_type(file_name: str) -> str:
    """
    从文件名中提取 evaluator_type。支持的类型包括：
      "conclusion_withthinktag", "conclusion_withoutthinktag", "cot_withthinktag", "cot_nothinktag"
      
    如果未能匹配，则返回 "unknown"。

    参数:
        file_name (str): 文件名字符串。

    返回:
        str: 提取到的 evaluator_type。
    """
    evaluator_types = ["conclusion_withthinktag", "conclusion_withoutthinktag", "cot_withthinktag", "cot_nothinktag"]
    for et in evaluator_types:
        if et in file_name:
            return et
    return "unknown"

def process_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    处理指定文件夹下所有 JSON 文件，计算每个文件的评测指标平均值和标准差，
    并添加从文件名中提取的 evaluator_type 字段。

    参数:
        folder_path (str): 存放 parsed JSON 文件的文件夹路径。

    返回:
        List[Dict[str, Any]]: 每个元素为一个文件的结果字典，
            包含文件名、evaluator_type 及各项指标的统计数据。
    """
    if not os.path.exists(folder_path):
        logger.error("文件夹不存在: %s", folder_path)
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    results = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".json"):
            file_path = os.path.join(folder_path, file)
            try:
                file_result = process_json_file(file_path)
                file_result["evaluator_type"] = extract_evaluator_type(file)
                results.append(file_result)
                logger.info("处理文件 %s 成功", file)
            except Exception as e:
                logger.error("处理文件 %s 失败: %s", file, e)
    return results

def save_to_csv(data: List[Dict[str, Any]], output_folder: str):
    """
    将结果数据保存到 CSV 文件中，生成两个文件：
      1. "average_scores_raw.csv"：按 Model 与 Annotation 聚合后输出，每行记录唯一的 Model、Annotation、Category 及各指标以 "avg ± std" 格式显示；
      2. "average_scores_pivot.csv"：按 Category 和 Annotation 分组后，各指标均值和标准差汇总（以 "avg ± std" 格式显示）。

    参数:
        data (List[Dict[str, Any]]): 每个元素为字典，包含文件名、evaluator_type 及各项指标的统计数据。
        output_folder (str): 输出 CSV 文件保存的文件夹路径。
    """
    try:
        import pandas as pd
    except ImportError as e:
        logger.error("pandas 库未安装，请先安装 pandas。")
        raise e

    # 将数据转换为 DataFrame（保留原始数值列）
    df = pd.DataFrame(data)
    # 对 file_name 列进行映射转换
    df["file_name"] = df["file_name"].apply(map_file_name)
    # 拆分 file_name 为 Model 与 Annotation，并新增 Category 列
    df[["Model", "Annotation"]] = df["file_name"].apply(lambda x: pd.Series(split_model_annotation(x)))
    df["Category"] = df["Model"].apply(map_category)
    
    # 将所有 *_avg 和 *_std 列转换为数值
    numeric_cols = [col for col in df.columns if col.endswith("_avg") or col.endswith("_std")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # 先对相同 Model 与 Annotation 的数据进行聚合（raw），取各指标均值，Category取第一项
    agg_dict = {col: "mean" for col in numeric_cols}
    agg_dict["Category"] = "first"
    grouped = df.groupby(["Model", "Annotation"], as_index=False).agg(agg_dict)
    
    # 生成每个指标的 "avg ± std" 格式
    for orig, abbr in METRIC_MAPPING.items():
        avg_col = f"{orig}_avg"
        std_col = f"{orig}_std"
        if avg_col not in grouped.columns:
            grouped[avg_col] = None
        if std_col not in grouped.columns:
            grouped[std_col] = None
        grouped[abbr] = grouped.apply(lambda row: combine_metric(row[avg_col], row[std_col]), axis=1)
    
    raw_columns = ["Model", "Annotation"] + [METRIC_MAPPING[m] for m in METRIC_MAPPING]
    raw_df = grouped[raw_columns].copy()
    
    raw_csv = os.path.join(output_folder, "average_scores_raw.csv")
    try:
        os.makedirs(output_folder, exist_ok=True)
        raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
        logger.info("Raw Data CSV 文件已保存至：%s", raw_csv)
    except Exception as e:
        logger.error("保存 Raw Data CSV 文件失败: %s", e)
        raise e

    # Pivot 数据：按 Category 与 Annotation 聚合原始数值数据
    pivot_numeric_avg = [col for col in df.columns if col.endswith("_avg")]
    pivot_numeric_std = [col for col in df.columns if col.endswith("_std")]
    
    try:
        pivot_avg = df.groupby(["Category", "Annotation"])[pivot_numeric_avg].mean().reset_index()
        pivot_std = df.groupby(["Category", "Annotation"])[pivot_numeric_std].mean().reset_index()
    except Exception as e:
        logger.error("生成 Pivot 数据失败: %s", e)
        raise e

    pivot_df = pivot_avg.merge(pivot_std, on=["Category", "Annotation"], suffixes=("_avg", "_std"))
    for orig, abbr in METRIC_MAPPING.items():
        avg_col = f"{orig}_avg"
        std_col = f"{orig}_std"
        pivot_df[abbr] = pivot_df.apply(lambda row: combine_metric(row[avg_col], row[std_col]), axis=1)
    pivot_columns = ["Category", "Annotation"] + list(METRIC_MAPPING.values())
    pivot_df = pivot_df[pivot_columns].copy()
    
    pivot_csv = os.path.join(output_folder, "average_scores_pivot.csv")
    try:
        pivot_df.to_csv(pivot_csv, index=False, encoding="utf-8-sig")
        logger.info("Pivot CSV 文件已保存至：%s", pivot_csv)
    except Exception as e:
        logger.error("保存 Pivot CSV 文件失败: %s", e)
        raise e

def main():
    """
    主函数：
      1. 指定包含 parsed JSON 文件的文件夹路径；
      2. 处理该文件夹下所有 JSON 文件，计算各项指标的平均值和标准差，并提取 evaluator_type；
      3. 对结果进行后处理（映射文件名、拆分 Model/Annotation、生成 Category、合并重复行及格式化指标数据），
         最终生成 Raw Data 和 Pivot 两个 CSV 文件。
    """
    folder_path = "gptEvaluation/result/parsed"  # 指定包含 JSON 文件的文件夹
    output_folder = folder_path                  # 输出 CSV 文件保存的文件夹

    try:
        results = process_folder(folder_path)
    except Exception as e:
        logger.error("处理文件夹失败: %s", e)
        return

    try:
        save_to_csv(results, output_folder)
    except Exception as e:
        logger.error("保存 CSV 文件失败: %s", e)

if __name__ == "__main__":
    main()
