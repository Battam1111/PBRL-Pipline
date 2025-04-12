#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluator.py

本模块封装了针对不同评估任务的评估器类，共包含四种评估场景：
  1. ConclusionEvaluatorWithThink —— 针对带 <think> 标记的结论评估。
  2. ConclusionEvaluatorWithoutThink —— 针对不带 <think> 标记的结论评估。
  3. CotEvaluatorWithThink —— 针对带 <think> 标记的链式思考（CoT）评估。
  4. CotEvaluatorWithoutThink —— 针对不带 <think> 标记的链式思考（CoT）评估。

每个评估器执行流程：
  1. 从 prompt.py 中加载对应的原始 Prompt 模板（该文件保持不变）。
  2. 替换模板中占位符为实际的文本查询、参考答案和 LLM 生成的回答；
  3. 调用 GPTScorer.send_prompt() 发送完整 Prompt，并获取模型响应；
  4. 返回模型生成的评估结果（通常为 JSON 格式字符串）。

新增功能：
  - 在调用 GPTScorer 前记录完整的 Prompt 信息，通过可选参数传入的 PromptLogger 记录器实现日志存储，
    日志中包含 question_id、Prompt 内容及当前时间戳。
"""

from typing import Any, Optional
from gptScorer import GPTScorer
from prompt import (
    eval_conclusion_withthinktag,
    eval_conclusion_withoutthinktag,
    eval_cot_withthinktag,
    eval_cot_withoutthinktag
)
from prompt_logger import PromptLogger

class BaseEvaluator:
    """
    所有评估器的基类，提供公共方法：
      - format_prompt()：根据 textual_query、reference_answer、llm_response 替换模板占位符；
      - evaluate()：调用 GPTScorer 发送请求，返回评估结果，并可记录发送的 Prompt 日志。
    """
    def __init__(self, prompt_template: str, model: str = None):
        """
        初始化评估器。

        参数:
            prompt_template (str): 评估任务的原始 Prompt 模板。
            model (str): 指定调用的 OpenAI 模型名称，默认为 None，使用 GPTScorer 默认值。
        """
        self.prompt_template = prompt_template
        self.scorer = GPTScorer(model=model)

    def format_prompt(self, textual_query: str, reference_answer: str, llm_response: str) -> str:
        """
        根据模板顺序替换占位符，生成完整评估 Prompt。

        参数:
            textual_query (str): 文本查询。
            reference_answer (str): 参考答案文本。
            llm_response (str): LLM 生成的回答文本。

        返回:
            str: 完整的评估 Prompt。
        """
        prompt = self.prompt_template.replace("[Insert Query]", textual_query, 1)
        prompt = prompt.replace("[Insert Reference Answer]", reference_answer, 1)
        prompt = prompt.replace("[Insert LLM Response]", llm_response, 1)
        return prompt

    def evaluate(self, textual_query: str, reference_answer: str, llm_response: str,
                 question_id: str = "N/A", prompt_logger: Optional[PromptLogger] = None) -> Any:
        """
        执行评估流程：格式化 Prompt 并调用 GPTScorer 发送请求。
        如果提供了 prompt_logger，则记录发送给 GPT 的 Prompt 信息。

        参数:
            textual_query (str): 文本查询。
            reference_answer (str): 参考答案。
            llm_response (str): LLM 生成的回答。
            question_id (str): 当前评估实例的标识符。
            prompt_logger (Optional[PromptLogger]): 用于记录 GPT 输入信息的日志记录器。

        返回:
            Any: 模型返回的评估结果（通常为 JSON 格式字符串）。
        """
        prompt = self.format_prompt(textual_query, reference_answer, llm_response)
        if prompt_logger is not None:
            prompt_logger.log(question_id, prompt)
        result = self.scorer.send_prompt(prompt)
        return result

# -----------------------------------------------
# 结论评估 —— 带 <think> 标记
# -----------------------------------------------
class ConclusionEvaluatorWithThink(BaseEvaluator):
    """
    针对带 <think> 标记的结论评估。
    使用 eval_conclusion_withthinktag 模板。
    """
    def __init__(self, model: str = None):
        super().__init__(prompt_template=eval_conclusion_withthinktag, model=model)

# -----------------------------------------------
# 结论评估 —— 不带 <think> 标记
# -----------------------------------------------
class ConclusionEvaluatorWithoutThink(BaseEvaluator):
    """
    针对不带 <think> 标记的结论评估。
    使用 eval_conclusion_withoutthinktag 模板。
    """
    def __init__(self, model: str = None):
        super().__init__(prompt_template=eval_conclusion_withoutthinktag, model=model)

# -----------------------------------------------
# CoT 评估 —— 带 <think> 标记
# -----------------------------------------------
class CotEvaluatorWithThink(BaseEvaluator):
    """
    针对带 <think> 标记的链式思考评估。
    使用 eval_cot_withthinktag 模板。
    """
    def __init__(self, model: str = None):
        super().__init__(prompt_template=eval_cot_withthinktag, model=model)

# -----------------------------------------------
# CoT 评估 —— 不带 <think> 标记
# -----------------------------------------------
class CotEvaluatorWithoutThink(BaseEvaluator):
    """
    针对不带 <think> 标记的链式思考评估。
    使用 eval_cot_withoutthinktag 模板。
    """
    def __init__(self, model: str = None):
        super().__init__(prompt_template=eval_cot_withoutthinktag, model=model)
