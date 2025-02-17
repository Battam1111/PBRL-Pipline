#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py
=========
本模块提供奖励模型所需的网络结构生成函数，包括基于 MLP 的非图像奖励模型、
基于卷积神经网络的图像奖励模型和基于 ResNet 的图像奖励模型生成函数。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_net import CNN, fanin_init

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    """
    生成一个多层感知机（MLP），用于处理状态-动作向量输入。

    参数：
        in_size (int): 输入维度。
        out_size (int): 输出维度。
        H (int): 隐藏层单元数。
        n_layers (int): 隐藏层数目。
        activation (str): 激活函数，可选 'tanh'、'sig' 或默认 ReLU。

    返回：
        list: 各层模块列表，可直接用于 nn.Sequential 构建网络。
    """
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())
    return net

def gen_image_net(image_height, image_width, 
                  conv_kernel_sizes=[5, 3, 3, 3], 
                  conv_n_channels=[16, 32, 64, 128], 
                  conv_strides=[3, 2, 2, 2]):
    """
    生成基于卷积网络的图像奖励模型。

    参数：
        image_height (int): 图像高度。
        image_width (int): 图像宽度。
        conv_kernel_sizes (list): 各卷积层核大小列表。
        conv_n_channels (list): 各卷积层通道数列表。
        conv_strides (list): 各卷积层步长列表。

    返回：
        CNN: 构建好的卷积神经网络模型。
    """
    conv_args = dict(
        kernel_sizes=conv_kernel_sizes,
        n_channels=conv_n_channels,
        strides=conv_strides,
        output_size=1,
    )
    conv_kwargs = dict(
        hidden_sizes=[],  # 卷积层后接的全连接层
        batch_norm_conv=False,
        batch_norm_fc=False,
    )
    return CNN(
        **conv_args,
        paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        input_height=image_height,
        input_width=image_width,
        input_channels=3,
        init_w=1e-3,
        hidden_init=fanin_init,
        **conv_kwargs
    )

def gen_image_net2():
    """
    生成基于 ResNet-18 的图像奖励模型。

    返回：
        ResNet: 构建的 ResNet-18 模型。
    """
    from torchvision.models.resnet import ResNet, BasicBlock
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    return model
