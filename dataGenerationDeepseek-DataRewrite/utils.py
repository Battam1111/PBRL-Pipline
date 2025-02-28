#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py

通用工具函数模块，包含：
  - 日志输出函数（后续可替换为更专业的 logging 模块）
  - 通用重试装饰器与封装请求函数（支持指数退避重试）
"""

import time
import functools
import requests

def log(message: str):
    """
    统一日志输出函数，当前直接打印日志，
    后续可扩展为记录到日志文件或使用 logging 模块。
    """
    print(message)

def retry(exceptions, tries=5, delay=3, backoff=2):
    """
    通用重试装饰器。
    
    :param exceptions: 捕获的异常类型（单个或元组）
    :param tries: 最大重试次数
    :param delay: 初始等待秒数
    :param backoff: 等待时间倍数（每次失败后延迟时间乘以该值）
    """
    def deco_retry(f):
        @functools.wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    log(f"[retry] 异常：{e}，等待 {mdelay} 秒后重试...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

def robust_request(func, *args, **kwargs):
    """
    使用 retry 装饰器封装请求，遇到 requests 异常时自动重试。
    """
    wrapped = retry(requests.exceptions.RequestException)(func)
    return wrapped(*args, **kwargs)
