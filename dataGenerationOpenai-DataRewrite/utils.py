#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py

通用工具函数模块，包含：
  - 统一日志输出函数（可扩展为 logging 模块）
  - 通用重试装饰器与 robust_request 函数
"""

import time
import functools
import requests

def log(message: str):
    """统一日志输出函数，可根据需要扩展为 logging 模块。"""
    print(message)

def retry(exceptions, tries=5, delay=3, backoff=2):
    """
    装饰器：对被装饰函数实现通用重试机制。
    
    :param exceptions: 要捕获的异常类型或元组
    :param tries: 最大重试次数
    :param delay: 初始等待秒数
    :param backoff: 等待时间倍数
    """
    def deco_retry(f):
        @functools.wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    log(f"[retry] 异常：{e}，等待 {mdelay} 秒重试...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

def robust_request(func, *args, **kwargs):
    """
    使用 retry 装饰器封装请求，遇到请求异常时自动重试。
    """
    wrapped = retry(requests.exceptions.RequestException)(func)
    return wrapped(*args, **kwargs)
