#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件，用于集中管理所有全局配置参数，例如 OpenAI API Key、API Base URL、默认模型名称、重试参数等。
请确保在运行前设置环境变量：OPENAI_API_KEY（必须）、OPENAI_API_BASE（可选）以及 OPENAI_MODEL（可选）。
"""

import os

# 从环境变量中加载 OpenAI API 配置（生产中请勿在代码中硬编码敏感信息）
# OPENAI_API_KEY = "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"
OPENAI_API_KEY = "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"

if not OPENAI_API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY！")

# 如果有自定义 API 基地址，可通过环境变量 OPENAI_API_BASE 设置；否则保持默认空字符串
OPENAI_API_BASE = "https://api.openai.com/v1"

# 默认使用的 OpenAI 模型名称，推荐使用 "gpt-3.5-turbo" 或 "gpt-4"（如有权限）
DEFAULT_MODEL = "chatgpt-4o-latest"

# 重试机制相关参数
RETRY_INTERVAL = 1.0  # 每次重试的间隔（单位：秒）
MAX_RETRIES = 5       # 单次请求允许的最大重试次数（防止死循环）