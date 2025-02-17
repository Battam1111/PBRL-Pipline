#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
=========
本模块用于配置全局参数，例如计算设备（GPU 或 CPU）。
"""

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
