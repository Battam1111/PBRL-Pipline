#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gptScorer.py

本模块封装了与 OpenAI ChatCompletion API 的交互逻辑，
提供 send_prompt() 接口以及针对评分任务的 get_score() 和 get_average_score() 方法。

此版本不使用 openai 包，而是直接通过 requests 调用 OpenAI API 端点。
"""

import time
import logging
import requests
from typing import Optional
import config

# 初始化日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GPTScorer:
    """
    GPTScorer 类封装了与 OpenAI ChatCompletion API 的调用逻辑，
    提供发送完整 Prompt 并返回响应文本的功能，同时支持单次评分和多次评分取平均值。
    """
    def __init__(self, model: Optional[str] = None):
        """
        初始化 GPTScorer 对象。

        参数:
          model (str): 指定调用的 OpenAI 模型名称，默认为 config.DEFAULT_MODEL。
        """
        self.model = model if model else config.DEFAULT_MODEL

    def send_prompt(self, prompt: str) -> str:
        """
        将完整 Prompt 发送至 OpenAI 接口，并返回模型响应文本。
        内置最大重试次数机制，防止无限重试。
        
        参数:
          prompt (str): 完整评估 Prompt 文本。
        
        返回:
          str: 模型响应中的文本内容。
        """
        url = f"{config.OPENAI_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model,
            "messages": messages
        }
        attempt = 0
        while attempt < config.MAX_RETRIES:
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                result = data["choices"][0]["message"]["content"]
                logger.info("请求成功返回结果。")
                return result
            except Exception as e:
                attempt += 1
                logger.error(f"请求出错，正在重试（{attempt}/{config.MAX_RETRIES}）：{e}")
                time.sleep(config.RETRY_INTERVAL)
        raise RuntimeError("超过最大重试次数，无法获得有效响应。")

    def get_score(self, question: str, question_type: str, answer_model: str, answer_label: str) -> str:
        """
        根据给定的问题和答案构造评分 Prompt，调用 send_prompt() 获取模型返回的置信度分数。
        
        参数:
          question (str): 问题内容。
          question_type (str): 问题类型。
          answer_model (str): 模型生成的答案。
          answer_label (str): 参考标签答案。
        
        返回:
          str: 模型返回的置信度分数字符串（仅包含分数）。
        """
        prompt = (
            "Now I will give you a question, the type of the question, an answer from model, and an answer from label.\n"
            "Focus on these two answers and figure out whether they are saying the same thing about the specific question type.\n"
            "Output only a confidence score between 0 and 100.\n"
            "Here are some examples:\n\n"
            "question1: How many oranges will there be if 1/3 are removed?\n"
            "question type: Knowledge\n"
            "answer from model: There will be 6 left.\n"
            "answer from label: With 9 oranges in total, 6 remain after removing 1/3.\n"
            "confidence score: 100\n\n"
            "question2: What is this object?\n"
            "question type: General Visual Recognition\n"
            "answer from model: This is a bathtub.\n"
            "answer from label: This is a dirty bathtub.\n"
            "confidence score: 80\n\n"
            "question3: What is this object?\n"
            "question type: General Visual Recognition\n"
            "answer from model: This is a bottle of water.\n"
            "answer from label: This is a bottle of oil.\n"
            "confidence score: 50\n\n"
            "question4: What is holding in this boy's right hand?\n"
            "question type: Spatial Recognition\n"
            "answer from model: He is holding a white cup.\n"
            "answer from label: He is holding a sword.\n"
            "confidence score: 0\n\n"
            "Now, here is the new data:\n"
            f"question: {question}\n"
            f"question type: {question_type}\n"
            f"answer from model: {answer_model}\n"
            f"answer from label: {answer_label}\n"
            "Output only the confidence score (number between 0 and 100)."
        )
        return self.send_prompt(prompt)

    def get_average_score(self, question: str, question_type: str, answer_model: str, answer_label: str, times: int = 5) -> float:
        """
        多次调用 get_score() 计算评分平均值，以降低偶然误差。
        
        参数:
          question (str): 问题内容。
          question_type (str): 问题类型。
          answer_model (str): 模型生成的答案。
          answer_label (str): 参考标签答案。
          times (int): 调用次数，默认为 5 次。
        
        返回:
          float: 多次评分后的平均置信度分数。
        """
        scores = []
        while len(scores) < times:
            score_str = self.get_score(question, question_type, answer_model, answer_label)
            try:
                score = float(score_str.strip())
                if 0 <= score <= 100:
                    scores.append(score)
            except ValueError:
                logger.warning("解析评分失败，跳过此次结果。")
                continue
        average = sum(scores) / len(scores)
        logger.info(f"多次评分平均值：{average}")
        return average
