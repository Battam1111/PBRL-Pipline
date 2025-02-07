#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
api_key_manager.py

本模块提供 APIKeyManager 类，用于管理多个 API Key 以及多个 GEMINI 项目 ID，
并实现线程安全的轮询分配机制，以最大化 API 调用效率。

特点：
  - 接受多个 API Key 和多个项目 ID。
  - 提供线程安全的轮询分配方法，确保每次请求均使用不同的 API Key 和项目 ID（轮询模式）。
  - 支持在 API 调用过程中自动切换 API Key 与项目 ID 以应对限流情况。
"""

import threading

class APIKeyManager:
    def __init__(self, api_keys, project_ids):
        """
        初始化 APIKeyManager。

        参数：
          api_keys: API Key 列表，至少包含一个。
          project_ids: GEMINI 项目 ID 列表，至少包含一个。
        """
        if not isinstance(api_keys, list) or len(api_keys) == 0:
            raise ValueError("必须提供至少一个 API Key")
        if not isinstance(project_ids, list) or len(project_ids) == 0:
            raise ValueError("必须提供至少一个项目 ID")
        self.api_keys = api_keys
        self.project_ids = project_ids
        self.api_key_index = 0
        self.project_id_index = 0
        self.lock = threading.Lock()
    
    def get_next(self):
        """
        获取下一个可用的 API Key 和项目 ID。

        返回：
          (api_key, project_id) 的元组。
        """
        with self.lock:
            api_key = self.api_keys[self.api_key_index]
            project_id = self.project_ids[self.project_id_index]
            self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)
            self.project_id_index = (self.project_id_index + 1) % len(self.project_ids)
        return api_key, project_id
