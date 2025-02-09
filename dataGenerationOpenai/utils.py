#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py

本模块封装了项目中通用的工具函数，包括：
  - 通用重试机制（支持指数退避、详细日志记录）
  - 日志输出函数（可扩展为使用 logging 模块）
  
所有工具函数均提供详细的中文注释说明。
"""

import time
import functools
import requests

def log(message: str):
    """统一日志输出函数，未来可替换为 logging 模块。"""
    print(message)

def retry(exceptions, tries=99999, delay=3, backoff=2):
    """
    装饰器：对被装饰函数实现通用重试机制，遇到指定异常时等待后重试。
    
    :param exceptions: 要捕获的异常类型（可以是元组）
    :param tries: 最大重试次数
    :param delay: 初始等待秒数
    :param backoff: 等待时间的倍数（指数退避）
    """
    def deco_retry(f):
        @functools.wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            attempt = 1
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    status = None
                    if hasattr(e, 'response') and e.response:
                        status = e.response.status_code
                    if status == 429:
                        log(f"[retry] 限流（429），等待 {mdelay} 秒（重试 {attempt} 次）")
                    elif status == 402:
                        log(f"[retry] HTTP 402（余额不足），停止重试。")
                        raise
                    else:
                        log(f"[retry] 请求异常：{e}，等待 {mdelay} 秒（重试 {attempt} 次）")
                    time.sleep(mdelay)
                    mtries -= 1
                    attempt += 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

def robust_request(func, *args, **kwargs):
    """
    通用的 robust_request 封装，内部使用 retry 装饰器，
    适用于 requests 请求调用，若超过最大重试次数则抛出异常。
    """
    wrapped = retry(requests.exceptions.RequestException)(func)
    return wrapped(*args, **kwargs)
