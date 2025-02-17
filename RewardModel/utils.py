#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
========
本模块包含一些辅助函数，用于处理图像数组等常见数据预处理工作。
"""

import numpy as np
from PIL import Image

def robust_fix_image_array(img):
    """
    确保图像数组为 3 维 (H, W, C) 格式。
    如果检测到额外的单一维度（例如 (1, H, W, C) 或 (1,1,H,W,C)），则移除这些单一维度；
    如果无法自动剔除，则选择第一个样本。

    参数：
        img (np.array): 待处理的图像数组。
    返回：
        np.array: 修正后的图像数组，形状为 (H, W, C) 或 (H, W)。
    """
    # 逐步去除前导的单一维度（例如 (1, H, W, C)）
    while img.ndim > 3 and img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    # 如果仍然多余，但可以压缩非最后一维的单一维度，则进行处理
    if img.ndim > 3:
        new_dims = []
        for i, d in enumerate(img.shape):
            if i == img.ndim - 1:  # 保留最后一维（颜色通道）
                new_dims.append(d)
            else:
                if d != 1:
                    new_dims.append(d)
        if len(new_dims) == 3:
            img = img.reshape(new_dims)
    # 如果仍然多余，则直接取第一个样本
    if img.ndim > 3:
        img = img[0]
    return img

def fix_image_array(img):
    """
    检查并处理图像数组的形状，移除多余的单一维度。
    例如：若输入为 (1, H, W, 3) 或 (1, 1, H, W, 3)，则将其压缩为 (H, W, 3)。

    参数：
        img (np.array): 待处理的图像数组。

    返回：
        np.array: 修正后的图像数组，维度为 (H, W, 3) 或 (H, W)。
    """
    # 如果数组维度超过3且存在大小为1的多余轴，则调用 np.squeeze 进行压缩
    if img.ndim > 3:
        img = np.squeeze(img)
    return img
