#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
image_stitcher.py

本模块提供图像拼接功能，将同一 situation 下的多个视角图像拼接成一张高质量图像，
以解决因上传图像数量过多导致大模型产生幻觉的问题，并保证拼接后的图像无重叠、无明显画质下降。

主要功能：
  - 支持多种拼接布局：网格布局（grid）、水平布局（horizontal）、垂直布局（vertical）
  - 自动计算拼接图像的尺寸，在不缩放原图的前提下拼接所有图像
  - 提供鲁棒的异常处理，确保拼接过程稳定
"""

import os
import math
from PIL import Image

def stitch_images(image_paths, output_path, layout="grid"):
    """
    将多个图像拼接成一张图像并保存至指定路径。

    :param image_paths: 包含待拼接图像路径的列表
    :param output_path: 拼接后图像的保存路径
    :param layout: 拼接布局，支持 "grid"（网格布局）、"horizontal"（水平布局）、"vertical"（垂直布局）
    :return: 返回输出路径（即 output_path）
    """
    if not image_paths:
        raise ValueError("没有提供待拼接的图像路径列表。")
    
    # 如果仅有一张图像，则直接复制该图像
    if len(image_paths) == 1:
        with Image.open(image_paths[0]) as im:
            im.save(output_path, quality=95)
        return output_path

    # 根据不同的布局方式进行拼接
    if layout == "horizontal":
        return stitch_horizontal(image_paths, output_path)
    elif layout == "vertical":
        return stitch_vertical(image_paths, output_path)
    elif layout == "grid":
        return stitch_grid(image_paths, output_path)
    else:
        raise ValueError(f"不支持的布局类型: {layout}")

def stitch_horizontal(image_paths, output_path):
    """
    水平拼接图像，不缩放原图，保持原始质量。

    :param image_paths: 图像路径列表
    :param output_path: 拼接后图像的保存路径
    :return: 返回输出路径
    """
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
    """
    垂直拼接图像，不缩放原图，保持原始质量。

    :param image_paths: 图像路径列表
    :param output_path: 拼接后图像的保存路径
    :return: 返回输出路径
    """
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
    """
    网格拼接图像，将图像均匀排列成网格，不缩放原图，保持原始质量。

    拼接步骤：
      1. 根据图像数量计算网格的行数和列数（尽可能接近正方形）
      2. 按照行优先的顺序排列图像
      3. 计算每列的最大宽度和每行的最大高度
      4. 创建足够大的空白图像，并将每张图像粘贴到相应的位置，保持原始尺寸

    :param image_paths: 图像路径列表
    :param output_path: 拼接后图像的保存路径
    :return: 返回输出路径
    """
    images = [Image.open(p) for p in image_paths]
    num_images = len(images)
    # 计算网格尺寸，采用列数为 ceil(sqrt(n))，行数为 ceil(n / columns)
    columns = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / columns)

    # 将图像分配到网格中（行优先）
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

    # 计算每列的最大宽度和每行的最大高度
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

    # 创建空白图像，背景为白色
    stitched_image = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

    # 粘贴每个图像到对应的位置
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
