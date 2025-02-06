#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
image_stitcher.py

本模块提供图像拼接功能，将同一情景下的多个视角图像拼接成一张图像。
支持的拼接布局：
  - 网格布局（grid）
  - 水平布局（horizontal）
  - 垂直布局（vertical）

确保拼接后图像无重叠，且保持原始图像质量。
"""

import os
import math
from PIL import Image

def stitch_images(image_paths, output_path, layout="grid"):
    """
    将多个图像拼接成一张图像并保存至 output_path。

    参数：
      image_paths: 待拼接图像路径列表。
      output_path: 拼接后图像的保存路径。
      layout: 拼接布局，支持 "grid"、"horizontal"、"vertical"。
    返回：
      output_path（拼接后图像保存路径）。
    """
    if not image_paths:
        raise ValueError("没有提供待拼接的图像路径列表。")
    if len(image_paths) == 1:
        with Image.open(image_paths[0]) as im:
            im.save(output_path, quality=95)
        return output_path
    if layout == "horizontal":
        return stitch_horizontal(image_paths, output_path)
    elif layout == "vertical":
        return stitch_vertical(image_paths, output_path)
    elif layout == "grid":
        return stitch_grid(image_paths, output_path)
    else:
        raise ValueError(f"不支持的布局类型: {layout}")

def stitch_horizontal(image_paths, output_path):
    images = [Image.open(p) for p in image_paths]
    total_width = sum(im.width for im in images)
    max_height = max(im.height for im in images)
    stitched_image = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    current_x = 0
    for im in images:
        stitched_image.paste(im, (current_x, 0))
        current_x += im.width
        im.close()
    stitched_image.save(output_path, quality=95)
    return output_path

def stitch_vertical(image_paths, output_path):
    images = [Image.open(p) for p in image_paths]
    total_height = sum(im.height for im in images)
    max_width = max(im.width for im in images)
    stitched_image = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
    current_y = 0
    for im in images:
        stitched_image.paste(im, (0, current_y))
        current_y += im.height
        im.close()
    stitched_image.save(output_path, quality=95)
    return output_path

def stitch_grid(image_paths, output_path):
    images = [Image.open(p) for p in image_paths]
    num_images = len(images)
    columns = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / columns)
    grid = []
    for r in range(rows):
        row_images = []
        for c in range(columns):
            index = r * columns + c
            if index < num_images:
                row_images.append(images[index])
            else:
                row_images.append(None)
        grid.append(row_images)
    col_widths = [0] * columns
    row_heights = [0] * rows
    for r in range(rows):
        for c in range(columns):
            im = grid[r][c]
            if im is not None:
                col_widths[c] = max(col_widths[c], im.width)
                row_heights[r] = max(row_heights[r], im.height)
    total_width = sum(col_widths)
    total_height = sum(row_heights)
    stitched_image = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for r in range(rows):
        x_offset = 0
        for c in range(columns):
            im = grid[r][c]
            if im is not None:
                stitched_image.paste(im, (x_offset, y_offset))
                im.close()
            x_offset += col_widths[c]
        y_offset += row_heights[r]
    stitched_image.save(output_path, quality=95)
    return output_path
