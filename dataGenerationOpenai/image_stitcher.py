#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
image_stitcher.py

本模块提供图像拼接功能，将同一样本下的多个视角图像拼接成一张图像，
支持以下布局：
  - grid  网格布局
  - horizontal  水平拼接
  - vertical  垂直拼接

若仅有一张图像，则直接复制保存。

【改进说明】：
1. 为确保图像拼接顺序固定，而非随机，本模块在处理前会对输入的图像路径列表进行自然排序，
   例如对于文件名中包含 "view1" 和 "view2" 的情况，排序后必定保持 view1 在前，view2 在后。
2. 详细中文注释说明每个函数和关键步骤，确保代码易于理解和维护。
"""

import os
import math
import re
from PIL import Image

def atoi(text):
    """
    将字符串中的数字部分转换为整数，非数字部分保持原样。
    用于自然排序（例如 "view10" 中提取 10）。
    """
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    用正则表达式将文本拆分为数字和非数字的序列，返回一个列表，
    该列表可用于自然排序。
    
    例如："view10.jpg" 拆分为 ["view", 10, ".jpg"]
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def stitch_images(image_paths, output_path, layout="grid"):
    """
    拼接多张图像为一张大图。
    
    :param image_paths: 待拼接图像路径列表（无序列表，将按文件名自然排序）
    :param output_path: 拼接后图像的保存路径
    :param layout: 拼接布局，可选 "grid"（网格）、"horizontal"（水平） 或 "vertical"（垂直）
    :return: 输出图像保存路径
    :raises ValueError: 若 image_paths 为空时抛出异常
    """
    if not image_paths:
        raise ValueError("没有提供待拼接的图像路径。")

    # 自然排序
    sorted_paths = sorted(image_paths, key=natural_keys)

    # 如果只有一张图，直接保存
    if len(sorted_paths) == 1:
        with Image.open(sorted_paths[0]) as im:
            im.save(output_path, quality=95)
        return output_path

    if layout == "horizontal":
        return stitch_horizontal(sorted_paths, output_path)
    elif layout == "vertical":
        return stitch_vertical(sorted_paths, output_path)
    elif layout == "grid":
        return stitch_grid(sorted_paths, output_path)
    else:
        raise ValueError(f"不支持的布局类型: {layout}")

def stitch_horizontal(image_paths, output_path):
    """按水平方向拼接图像。"""
    images = [Image.open(p) for p in image_paths]
    total_width = sum(im.width for im in images)
    max_height = max(im.height for im in images)
    stitched = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    x_offset = 0
    for im in images:
        stitched.paste(im, (x_offset, 0))
        x_offset += im.width
        im.close()
    stitched.save(output_path, quality=95)
    return output_path

def stitch_vertical(image_paths, output_path):
    """按垂直方向拼接图像。"""
    images = [Image.open(p) for p in image_paths]
    total_height = sum(im.height for im in images)
    max_width = max(im.width for im in images)
    stitched = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for im in images:
        stitched.paste(im, (0, y_offset))
        y_offset += im.height
        im.close()
    stitched.save(output_path, quality=95)
    return output_path

def stitch_grid(image_paths, output_path):
    """按网格方式拼接图像。"""
    images = [Image.open(p) for p in image_paths]
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    col_widths = [0] * cols
    row_heights = [0] * rows

    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        col_widths[c] = max(col_widths[c], im.width)
        row_heights[r] = max(row_heights[r], im.height)

    total_w = sum(col_widths)
    total_h = sum(row_heights)
    stitched = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))

    y_offset = 0
    idx = 0
    for r in range(rows):
        x_offset = 0
        for c in range(cols):
            if idx < num_images:
                im = images[idx]
                stitched.paste(im, (x_offset, y_offset))
                im.close()
                idx += 1
            x_offset += col_widths[c]
        y_offset += row_heights[r]

    stitched.save(output_path, quality=95)
    return output_path
