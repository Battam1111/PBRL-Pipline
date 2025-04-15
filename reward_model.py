#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_model.py
================

本模块实现奖励模型（Reward Model）的构建、数据采集、标签打分以及训练流程，
支持两种数据模式：基于图像（data_type="image"）和基于点云（data_type="pointcloud"）。
其中，当使用 "pointcloud" 模式且 vlm == 'pointllm_two_image' 时，
奖励模型采用 PointCloudNet 架构，输入形状为 (N, point_cloud_num_points, 6)（每个点含 xyz 与 rgb），
输出为单标量；而当 data_type 为 "image" 时，则采用传统图像模型（如 ResNet-18）实现奖励预测.

【本版本说明】
1. 针对点云数据，添加数据时使用深拷贝，确保各轨迹数据独立，避免共享内存导致数据重复。
2. 采样查询对时，当仅有一条轨迹时，从该轨迹内不同时间步随机采样，保证数据不重复。
3. 在 put_queries 函数中，若数据类型为 "pointcloud" 且输入形状为 (N, point_cloud_num_points, 6)，
   则自动在轴1插入维度，使其变为 (N, 1, point_cloud_num_points, 6) 以匹配预设缓冲区形状。
4. 在训练奖励模型时，针对点云数据采用较小的批量（如16），并使用混合精度训练（AMP），
   以降低显存占用并提高效率。若仍遇内存碎片问题，可考虑设置环境变量 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import datetime
import pickle as pkl
import random
import cv2
from PIL import Image
import asyncio

# 引入用于 VLM 查询的 prompt 模块
from prompt import (
    gemini_free_query_env_prompts, gemini_summary_env_prompts,
    gemini_free_query_prompt1, gemini_free_query_prompt2,
    gemini_single_query_env_prompts,
    gpt_free_query_env_prompts, gpt_summary_env_prompts,
    pointllm_free_query_env_prompts, pointllm_summary_env_prompts,
    pointllm_free_query_prompt1, pointllm_free_query_prompt2,
)

from vlms.gemini_infer import gemini_query_2, gemini_query_1
from conv_net import CNN, fanin_init
from vlms.pointllm_infer import pointllm_query_1, pointllm_query_2
# 导入点云奖励模型生成函数（PointCloudNet实现）
from pointCloudNet import gen_point_cloud_net
# from pointCloudNetLight import gen_point_cloud_net
from extract import extract_label_from_response

# 设置计算设备，优先使用 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# =============================================================================
# 定义一个简单的 MLP 生成函数，用于非图像（向量）奖励模型
# =============================================================================
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

# =============================================================================
# 定义基于卷积网络的图像奖励模型生成函数（非 ResNet 版本）
# =============================================================================
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

# =============================================================================
# 定义基于 TorchVision ResNet 的图像奖励模型生成函数（通常生成 ResNet-18）
# =============================================================================
def gen_image_net2():
    """
    生成基于 ResNet-18 的图像奖励模型。

    返回：
        ResNet: 构建的 ResNet-18 模型。
    """
    from torchvision.models.resnet import ResNet, BasicBlock
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    return model

# =============================================================================
# 优化版 K-Center Greedy 算法
# 利用向量化操作和 torch.cdist 加速距离计算，用于从候选查询中选择具有代表性的样本
# =============================================================================
def KCenterGreedy(obs, full_obs, num_new_sample):
    """
    使用 K-Center Greedy 算法选择具有代表性的样本索引。

    参数：
        obs (np.array): 候选样本数组，形状 (N, feature_dim)。
        full_obs (np.array): 已采样样本数组，形状 (M, feature_dim)。
        num_new_sample (int): 需要选择的新样本数量。

    返回：
        list: 选中的样本索引列表。
    """
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    for count in range(num_new_sample):
        # 利用 torch.cdist 向量化计算欧氏距离
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist).item()
        selected_index.append(current_index[max_index])
        # 更新候选集：删除已选样本
        del current_index[max_index]
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([full_obs, obs[[selected_index[-1]]]], axis=0)
    return selected_index

def compute_smallest_dist(obs, full_obs):
    """
    使用 torch.cdist 向量化计算每个候选样本与已选样本集合中最小欧氏距离。

    参数：
        obs (np.array): 候选样本数组，形状 (N, feature_dim)。
        full_obs (np.array): 已选样本数组，形状 (M, feature_dim)。

    返回：
        torch.Tensor: 每个候选样本与最近样本的距离，形状 (N, 1)。
    """
    obs_tensor = torch.from_numpy(obs).float().to(device)
    full_obs_tensor = torch.from_numpy(full_obs).float().to(device)
    with torch.no_grad():
        dists = torch.cdist(obs_tensor, full_obs_tensor, p=2)
        small_dists = dists.min(dim=1).values
    return small_dists.unsqueeze(1)

# =============================================================================
# 辅助函数：修正图像数组形状，确保传入 PIL 时为 (H,W,3) 或 (H,W)
# =============================================================================
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

# =============================================================================
# RewardModel 类
# =============================================================================
class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, 
                 max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0,
                 
                 # VLM 相关参数
                 vlm_label=True,
                 env_name="CartPole-v1",
                 vlm="gemini_free_form",
                 clip_prompt=None,
                 log_dir=None,
                 flip_vlm_label=False,
                 save_query_interval=25,
                 cached_label_path=None,

                 # 图像奖励相关参数
                 reward_model_layers=3,
                 reward_model_H=256,
                 image_reward=True,  # True 表示使用 VLM（图像或点云均可）
                 image_height=128,
                 image_width=128,
                 resize_factor=1,
                 resnet=False,
                 conv_kernel_sizes=[5, 3, 3, 3],
                 conv_n_channels=[16, 32, 64, 128],
                 conv_strides=[3, 2, 2, 2],
                 
                 # 数据类型： "image" 或 "pointcloud"
                 data_type="pointcloud",
                 # 当 data_type=="pointcloud" 时，点云中点的数量
                 point_cloud_num_points=8192,
                 **kwargs
                ):
        """
        初始化 RewardModel 类实例，构建奖励模型并初始化相关数据缓存及参数。

        参数：
            ds: 状态维度。
            da: 动作维度。
            ensemble_size: 模型集大小。
            lr: 学习率。
            mb_size: 采样批量大小。
            size_segment: 存储轨迹时每段数据的尺寸。
            max_size: 存储轨迹的最大数量。
            capacity: 缓冲区容量。
            large_batch: 大批量训练倍数。
            label_margin: 标签边缘。
            teacher_beta, teacher_gamma, teacher_eps_*: 教师模型相关参数。
            vlm_label: 是否使用 VLM 标注。
            env_name: 环境名称。
            vlm: VLM 模式。
            clip_prompt, log_dir, flip_vlm_label, save_query_interval, cached_label_path:
                其它相关配置参数。
            reward_model_layers, reward_model_H: 奖励模型网络结构参数。
            image_reward: 是否使用图像奖励。
            image_height, image_width, resize_factor: 图像相关参数。
            resnet: 是否使用 ResNet 架构。
            conv_kernel_sizes, conv_n_channels, conv_strides: 卷积网络参数。
            data_type: 数据类型，可选 "image" 或 "pointcloud"。
            point_cloud_num_points: 点云中点的数量。
        """
        # 保存 VLM 相关参数
        self.vlm_label = vlm_label
        self.env_name = env_name
        self.vlm = vlm
        self.clip_prompt = clip_prompt
        self.vlm_label_acc = 0
        self.log_dir = log_dir
        self.flip_vlm_label = flip_vlm_label
        self.train_times = 0
        self.save_query_interval = save_query_interval

        # 基础参数
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.capacity = int(capacity)
        self.reward_model_layers = reward_model_layers
        self.reward_model_H = reward_model_H
        self.image_reward = image_reward
        self.resnet = resnet
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_n_channels = conv_n_channels
        self.conv_strides = conv_strides

        # 保存数据类型和点云相关设置
        self.data_type = data_type
        self.point_cloud_num_points = point_cloud_num_points

        # 针对点云数据与图像数据分别初始化存储列表
        self.point_cloud_inputs = []
        self.img_inputs = []

        # 根据数据类型构建缓冲区（预分配内存以降低动态扩容开销）
        if self.data_type == "image":
            # 图像数据格式： (capacity, 1, image_height, image_width, 3)
            self.buffer_seg1 = np.empty((self.capacity, 1, image_height, image_width, 3), dtype=np.uint8)
            self.buffer_seg2 = np.empty((self.capacity, 1, image_height, image_width, 3), dtype=np.uint8)
            self.image_height = image_height
            self.image_width = image_width
            self.resize_factor = resize_factor

        elif self.data_type == "pointcloud" or self.vlm == "pointllm_two_image":
            # 点云数据格式： (capacity, 1, point_cloud_num_points, 6)
            self.buffer_seg1 = np.empty((self.capacity, 1, point_cloud_num_points, 6), dtype=np.float32)
            self.buffer_seg2 = np.empty((self.capacity, 1, point_cloud_num_points, 6), dtype=np.float32)
            
        else:
            # 默认使用状态-动作向量
            self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
            self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
            
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        # 存储轨迹数据（用于奖励模型打标签），预留列表避免频繁内存拷贝
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size

        # 针对点云数据，建议使用较小的训练批量以降低内存占用
        if self.data_type == "pointcloud":
            self.train_batch_size = 1
        else:
            self.train_batch_size = 64 if not self.resnet else 32

        self.CEloss = nn.CrossEntropyLoss()

        # 教师参数设置
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        self.label_margin = label_margin
        self.label_target = 1 - 2 * self.label_margin

        # 处理缓存标签路径
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        if cached_label_path is not None:
            self.cached_label_path = os.path.join(dir_path, cached_label_path)
        else:
            self.cached_label_path = None

        self.read_cache_idx = 0
        if self.cached_label_path is not None:
            all_cached_labels = sorted(os.listdir(self.cached_label_path))
            self.all_cached_labels = [os.path.join(self.cached_label_path, x) for x in all_cached_labels]
        else:
            self.all_cached_labels = []

        # 构建 ensemble 奖励模型
        self.construct_ensemble()

    def eval(self):
        """设置所有模型为评估模式"""
        for i in range(self.de):
            self.ensemble[i].eval()

    def train(self):
        """设置所有模型为训练模式"""
        for i in range(self.de):
            self.ensemble[i].train()

    def softXEnt_loss(self, input, target):
        """
        计算 soft cross entropy 损失，适用于软标签。

        参数：
            input (torch.Tensor): 模型输出 logits。
            target (torch.Tensor): 软标签。

        返回：
            torch.Tensor: 损失值。
        """
        logprobs = F.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def change_batch(self, new_frac):
        """动态调整批量大小"""
        self.mb_size = int(self.origin_mb_size * new_frac)

    def set_batch(self, new_batch):
        """设置批量大小"""
        self.mb_size = int(new_batch)

    def set_teacher_thres_skip(self, new_margin):
        """设置教师跳过阈值"""
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip

    def set_teacher_thres_equal(self, new_margin):
        """设置教师相等阈值"""
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    # =============================================================================
    # 构建 ensemble 奖励模型
    # 根据不同配置构建不同架构的模型：
    #   1. 非图像奖励：使用纯 MLP；
    #   2. 使用 pointllm_two_image 且 data_type 为 pointcloud：使用 PointCloudNet；
    #   3. 其它情况：根据 resnet 参数使用 gen_image_net 或 gen_image_net2。
    # =============================================================================
    def construct_ensemble(self):
        for i in range(self.de):
            if not self.image_reward:
                model = nn.Sequential(*gen_net(
                    in_size=self.ds + self.da,
                    out_size=1,
                    H=self.reward_model_H,
                    n_layers=self.reward_model_layers,
                    activation=self.activation
                )).float().to(device)
            if not self.vlm_label:
                if self.data_type == "pointcloud":
                    model = gen_point_cloud_net(num_points=self.point_cloud_num_points, input_dim=6, device=device, normalize=False)
                else:
                    if not self.resnet:
                        model = gen_image_net(
                            self.image_height, self.image_width,
                            self.conv_kernel_sizes, self.conv_n_channels, self.conv_strides
                        ).float().to(device)
                    else:
                        model = gen_image_net2().float().to(device)
            else:
                if self.vlm == 'pointllm_two_image':
                    if self.data_type == "pointcloud":
                        model = gen_point_cloud_net(num_points=self.point_cloud_num_points, input_dim=6, device=device, normalize=False)
                    else:
                        if not self.resnet:
                            model = gen_image_net(
                                self.image_height, self.image_width,
                                self.conv_kernel_sizes, self.conv_n_channels, self.conv_strides
                            ).float().to(device)
                        else:
                            model = gen_image_net2().float().to(device)
                else:
                    if not self.resnet:
                        model = gen_image_net(
                            self.image_height, self.image_width,
                            self.conv_kernel_sizes, self.conv_n_channels, self.conv_strides
                        ).float().to(device)
                    else:
                        model = gen_image_net2().float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    # =============================================================================
    # 数据采集：将新数据添加到轨迹中
    # 根据 data_type 或 vlm=="pointllm_two_image" 将图像数据存入 img_inputs，
    # 将点云数据存入 point_cloud_inputs。
    # =============================================================================
    def add_data(self, obs, act, rew, done, img=None, point_cloud=None):
        """
        将新数据加入轨迹缓存。

        参数：
            obs (np.array): 当前状态。
            act (np.array): 当前动作。
            rew (float 或 np.array): 当前奖励。
            done (bool): 当前是否终止。
            img (np.array): 对应图像数据（可选）。
            point_cloud (np.array): 对应点云数据（可选）。
        """
        sa_t = np.concatenate([obs, act], axis=-1)
        flat_input = sa_t.reshape(1, self.da + self.ds)
        r_t = np.array(rew).reshape(1, 1)
        if (self.data_type == "image" or self.vlm == "pointllm_two_image") and img is not None:
            flat_img = np.copy(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))
        if (self.data_type == "pointcloud" or self.vlm == "pointllm_two_image") and point_cloud is not None:
            point_cloud = point_cloud.astype(np.float32)
            flat_point_cloud = np.copy(point_cloud.reshape(1, self.point_cloud_num_points, 6))
        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(r_t)
            if (self.data_type == "image" or self.vlm == "pointllm_two_image") and img is not None:
                self.img_inputs.append(flat_img)
            if (self.data_type == "pointcloud" or self.vlm == "pointllm_two_image") and point_cloud is not None:
                self.point_cloud_inputs.append(flat_point_cloud)
        elif done:
            if 'Cloth' not in self.env_name:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], r_t])
                if (self.data_type == "image" or self.vlm == "pointllm_two_image") and img is not None:
                    self.img_inputs[-1] = np.concatenate([self.img_inputs[-1], flat_img], axis=0)
                if (self.data_type == "pointcloud" or self.vlm == "pointllm_two_image") and point_cloud is not None:
                    self.point_cloud_inputs[-1] = np.concatenate([self.point_cloud_inputs[-1], flat_point_cloud], axis=0)
                if len(self.inputs) > self.max_size:
                    self.inputs = self.inputs[1:]
                    self.targets = self.targets[1:]
                    if self.data_type == "image" or self.vlm == "pointllm_two_image":
                        self.img_inputs = self.img_inputs[1:]
                    if self.data_type == "pointcloud" or self.vlm == "pointllm_two_image":
                        self.point_cloud_inputs = self.point_cloud_inputs[1:]
                self.inputs.append([])
                self.targets.append([])
                if self.data_type == "image" or self.vlm == "pointllm_two_image":
                    self.img_inputs.append([])
                if self.data_type == "pointcloud" or self.vlm == "pointllm_two_image":
                    self.point_cloud_inputs.append([])
            else:
                self.inputs.append([flat_input])
                self.targets.append([r_t])
                if self.data_type == "image" or self.vlm == "pointllm_two_image":
                    self.img_inputs.append([flat_img])
                if self.data_type == "pointcloud" or self.vlm == "pointllm_two_image":
                    self.point_cloud_inputs.append([flat_point_cloud])
                if len(self.inputs) > self.max_size:
                    self.inputs = self.inputs[1:]
                    self.targets = self.targets[1:]
                    if self.data_type == "image" or self.vlm == "pointllm_two_image":
                        self.img_inputs = self.img_inputs[1:]
                    if self.data_type == "pointcloud" or self.vlm == "pointllm_two_image":
                        self.point_cloud_inputs = self.point_cloud_inputs[1:]
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = r_t
                if (self.data_type == "image" or self.vlm == "pointllm_two_image") and img is not None:
                    self.img_inputs[-1] = flat_img
                if (self.data_type == "pointcloud" or self.vlm == "pointllm_two_image") and point_cloud is not None:
                    self.point_cloud_inputs[-1] = flat_point_cloud
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], r_t])
                if (self.data_type == "image" or self.vlm == "pointllm_two_image") and img is not None:
                    self.img_inputs[-1] = np.concatenate([self.img_inputs[-1], flat_img], axis=0)
                if (self.data_type == "pointcloud" or self.vlm == "pointllm_two_image") and point_cloud is not None:
                    self.point_cloud_inputs[-1] = np.concatenate([self.point_cloud_inputs[-1], flat_point_cloud], axis=0)

    def add_data_batch(self, obses, rewards):
        """
        批量添加数据至轨迹缓存。

        参数：
            obses (np.array): 状态数组，形状 (num_env, ...)。
            rewards (np.array): 奖励数组，对应形状 (num_env, ...)。
        """
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])

    # =============================================================================
    # 从轨迹中随机采样查询对
    # 采样策略：
    #   - 当轨迹数（pool_size） > 1 时：按轨迹级采样；
    #   - 当 pool_size == 1 时：从该轨迹内随机采样不同时间步，确保返回数据来自不同帧。
    # 注意：本方法始终返回8个变量：
    #       (sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2)
    #       图像数据始终返回，而点云数据仅在
    #       (self.cfg.vlm == 'pointllm_two_image' or self.cfg.reward_data_type == "pointcloud")
    #       时返回，否则置为None。
    # =============================================================================
    def get_queries(self, mb_size=20):
        """
        从轨迹中随机采样一批查询对，用于奖励标签生成。始终返回8个变量：
            - sa_t_1: 状态-动作序列1，形状 (mb_size, ds+da)；若无则返回空数组。
            - sa_t_2: 状态-动作序列2，形状 (mb_size, ds+da)；若无则返回空数组。
            - r_t_1: 奖励序列1，形状 (mb_size, 1)。
            - r_t_2: 奖励序列2，形状 (mb_size, 1)。
            - pc_t_1: 点云数据1，形状 (mb_size, point_cloud_num_points, 6)；若不采样则返回None。
            - pc_t_2: 点云数据2，形状 (mb_size, point_cloud_num_points, 6)；若不采样则返回None。
            - img_t_1: 图像数据1，形状 (mb_size, image_height, image_width, 3)。
            - img_t_2: 图像数据2，形状 (mb_size, image_height, image_width, 3)。
        采样逻辑说明：
            1. 如果没有任何轨迹数据，则返回空状态-动作、奖励、图像数组，点云数据置为None。
            2. 过滤掉空轨迹后，构造状态-动作、奖励和图像数据（点云数据仅在条件满足时采样）。
            3. 当轨迹数量(pool_size)==1时，从单条轨迹中随机采样不同时间步；
               当pool_size>1时，先随机选择轨迹再从中采样时间步。
        """
        # ------------------------------
        # 1. 检查是否存在轨迹数据
        # ------------------------------
        if len(self.inputs) == 0:
            print("Warning: 没有可用的轨迹数据，返回空查询对")
            return (np.empty((0, self.ds + self.da)),  # sa_t_1
                    np.empty((0, self.ds + self.da)),  # sa_t_2
                    np.empty((0, 1)),                # r_t_1
                    np.empty((0, 1)),                # r_t_2
                    None,                            # 点云数据置为None
                    None,
                    np.empty((0, self.image_height, self.image_width, 3)),  # img_t_1
                    np.empty((0, self.image_height, self.image_width, 3)))  # img_t_2

        # ------------------------------
        # 2. 过滤掉空的轨迹数据
        # ------------------------------
        valid_indices = [i for i in range(len(self.inputs)) if len(self.inputs[i]) > 0]
        if len(valid_indices) == 0:
            print("Warning: 所有轨迹数据均为空，返回空查询对")
            return (np.empty((0, self.ds + self.da)),
                    np.empty((0, self.ds + self.da)),
                    np.empty((0, 1)),
                    np.empty((0, 1)),
                    None,
                    None,
                    np.empty((0, self.image_height, self.image_width, 3)),
                    np.empty((0, self.image_height, self.image_width, 3)))
        
        # ------------------------------
        # 3. 构造有效轨迹数据列表
        # ------------------------------
        # 从状态-动作和奖励数据中筛选有效轨迹
        filtered_inputs = [self.inputs[i] for i in valid_indices]
        filtered_targets = [self.targets[i] for i in valid_indices]
        # 图像数据始终返回
        filtered_images = [self.img_inputs[i] for i in valid_indices]
        # 判断是否需要采样点云数据：
        # 当 self.cfg.vlm == 'pointllm_two_image' 或 self.cfg.reward_data_type == "pointcloud" 时采样
        use_pointcloud = (self.vlm == 'pointllm_two_image' or self.data_type == "pointcloud")
        if use_pointcloud and len(self.point_cloud_inputs) >= len(self.inputs):
            filtered_pointclouds = [self.point_cloud_inputs[i] for i in valid_indices]
        else:
            filtered_pointclouds = []  # 如果条件不满足，则视为没有点云数据

        # ------------------------------
        # 4. 若多条轨迹且最后一条较短，则剔除最后一条以保证数据完整
        # ------------------------------
        if len(filtered_inputs) > 1 and len(filtered_inputs[-1]) < len(filtered_inputs[0]):
            filtered_inputs = filtered_inputs[:-1]
            filtered_targets = filtered_targets[:-1]
            filtered_images = filtered_images[:-1]
            if use_pointcloud and len(filtered_pointclouds) > 0:
                filtered_pointclouds = filtered_pointclouds[:-1]
        pool_size = len(filtered_inputs)
        
        # ------------------------------
        # 5. 将数据统一转换为 numpy 数组，便于后续采样索引
        # ------------------------------
        train_inputs = np.array(filtered_inputs)    # 状态-动作数据, 形状: (pool_size, traj_len, ds+da)
        train_targets = np.array(filtered_targets)    # 奖励数据, 形状: (pool_size, traj_len, 1)
        train_images = np.array(filtered_images)        # 图像数据, 形状: (pool_size, traj_len, H, W, 3)
        if use_pointcloud and len(filtered_pointclouds) > 0:
            train_point_clouds = np.array(filtered_pointclouds)  # 点云数据, 形状: (pool_size, traj_len, point_cloud_num_points, 6)
        else:
            train_point_clouds = None

        # ------------------------------
        # 6. 根据轨迹数量进行采样
        # ------------------------------
        if pool_size == 1:
            # 当只有一条轨迹时，从该轨迹内随机采样不同时间步
            traj_len = train_inputs.shape[1]
            if traj_len <= 0:
                print("Warning: 轨迹长度为0，无法采样数据")
                return (np.empty((0, self.ds + self.da)),
                        np.empty((0, self.ds + self.da)),
                        np.empty((0, 1)),
                        np.empty((0, 1)),
                        None,
                        None,
                        np.empty((0, self.image_height, self.image_width, 3)),
                        np.empty((0, self.image_height, self.image_width, 3)))
            time_indices1 = np.random.randint(0, traj_len, size=mb_size)
            time_indices2 = np.random.randint(0, traj_len, size=mb_size)
            sa_t_1 = train_inputs[0, time_indices1, :]
            sa_t_2 = train_inputs[0, time_indices2, :]
            r_t_1 = train_targets[0, time_indices1, :]
            r_t_2 = train_targets[0, time_indices2, :]
            img_t_1 = train_images[0, time_indices1, ...]
            img_t_2 = train_images[0, time_indices2, ...]
            if use_pointcloud and train_point_clouds is not None:
                pc_t_1 = train_point_clouds[0, time_indices1, ...]
                pc_t_2 = train_point_clouds[0, time_indices2, ...]
            else:
                pc_t_1 = None
                pc_t_2 = None
        else:
            # 当存在多条轨迹时，先随机选择轨迹再在各自轨迹中随机采样时间步
            batch_index_1 = np.random.choice(pool_size, size=mb_size, replace=True)
            batch_index_2 = np.random.choice(pool_size, size=mb_size, replace=True)
            traj_len = train_inputs.shape[1]
            time_indices1 = np.random.randint(0, traj_len, size=mb_size)
            time_indices2 = np.random.randint(0, traj_len, size=mb_size)
            sa_t_1 = train_inputs[batch_index_1, time_indices1, :]
            sa_t_2 = train_inputs[batch_index_2, time_indices2, :]
            r_t_1 = train_targets[batch_index_1, time_indices1, :]
            r_t_2 = train_targets[batch_index_2, time_indices2, :]
            img_t_1 = train_images[batch_index_1, time_indices1, ...]
            img_t_2 = train_images[batch_index_2, time_indices2, ...]
            if use_pointcloud and train_point_clouds is not None:
                pc_t_1 = train_point_clouds[batch_index_1, time_indices1, ...]
                pc_t_2 = train_point_clouds[batch_index_2, time_indices2, ...]
            else:
                pc_t_1 = None
                pc_t_2 = None

        # ------------------------------
        # 7. 返回统一的八元组：
        #     状态-动作数据、奖励数据、点云数据、图像数据
        # ------------------------------

        # 为图像baseline制定
        if (pc_t_1 == None) or (pc_t_2 == None):
            return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2

        return sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2









    # =============================================================================
    # put_queries: 将查询对及对应标签写入缓冲区
    # 采用预分配内存和 np.copyto 优化内存操作，确保数据高效写入缓冲区
    # =============================================================================
    def put_queries(self, query1, query2, labels):
        """
        将查询对及对应标签写入缓冲区。

        参数：
            query1 (np.array): 第一组查询数据。
                - 对于图像数据，其原始形状为 (N, H, W, 3)；
                - 对于点云数据，其原始形状为 (N, point_cloud_num_points, 6)。
            query2 (np.array): 第二组查询数据，与 query1 形状相同。
            labels (np.array): 对应标签，形状为 (N, 1)。

        实现说明：
            - 若数据类型为 "image"，则将数据 reshape 为 (N, 1, H, W, 3)。
            - 若数据类型为 "pointcloud" 且输入为 (N, point_cloud_num_points, 6)，则自动在轴 1 插入维度，
              转换为 (N, 1, point_cloud_num_points, 6)。
            - 随后使用 np.copyto 将数据写入预分配的缓冲区，支持环形写入。
        """
        total_sample = query1.shape[0]
        next_index = self.buffer_index + total_sample
        if self.data_type == "image":
            query1 = query1.reshape(query1.shape[0], 1, query1.shape[1], query1.shape[2], query1.shape[3])
            query2 = query2.reshape(query2.shape[0], 1, query2.shape[1], query2.shape[2], query2.shape[3])
        elif self.data_type == "pointcloud":
            if query1.ndim == 3:
                query1 = query1[:, None, :, :]
                query2 = query2[:, None, :, :]
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], query1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], query2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            remain = total_sample - maximum_index
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], query1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], query2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], query1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], query2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    # =============================================================================
    # 根据真实奖励生成理性标签，并结合 VLM（二次修正）生成最终标签
    # 针对不同数据类型分别处理。
    # =============================================================================
    def get_label(self, 
                sa_t_1, sa_t_2, 
                r_t_1, r_t_2,
                img_t_1=None, img_t_2=None,
                point_cloud_t_1=None, point_cloud_t_2=None):
        """
        生成奖励标签：先根据真实奖励计算理性标签，再通过 VLM（二次修正）生成最终标签，
        针对不同数据类型分别处理。
        
        参数：
            sa_t_1, sa_t_2: 状态-动作序列。
            r_t_1, r_t_2: 奖励序列。
            img_t_1, img_t_2: 图像数据（可选）。
            point_cloud_t_1, point_cloud_t_2: 点云数据（可选）。
        
        返回：
            根据配置返回不同数据元组（包含原始数据及生成的标签）。
        """
        # ------------------------------
        # A. 根据真实奖励生成理性标签
        # ------------------------------
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            valid_mask = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if np.sum(valid_mask) == 0:
                return None, None, None, None, []
            sa_t_1 = sa_t_1[valid_mask] if sa_t_1 is not None else None
            sa_t_2 = sa_t_2[valid_mask] if sa_t_2 is not None else None
            r_t_1 = r_t_1[valid_mask]
            r_t_2 = r_t_2[valid_mask]
            sum_r_t_1 = sum_r_t_1[valid_mask]
            sum_r_t_2 = sum_r_t_2[valid_mask]
            if self.data_type == "image" and img_t_1 is not None:
                img_t_1 = img_t_1[valid_mask]
                img_t_2 = img_t_2[valid_mask]
            if self.data_type == "pointcloud" and point_cloud_t_1 is not None:
                point_cloud_t_1 = point_cloud_t_1[valid_mask]
                point_cloud_t_2 = point_cloud_t_2[valid_mask]
        margin_mask = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for idx in range(seg_size - 1):
            temp_r_t_1[:, :idx + 1] *= self.teacher_gamma
            temp_r_t_2[:, :idx + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
        if self.teacher_beta > 0:
            r_hat = np.stack([sum_r_t_1, sum_r_t_2], axis=-1)
            r_hat_tensor = torch.from_numpy(r_hat).float().to(device) * self.teacher_beta
            prob_2 = F.softmax(r_hat_tensor, dim=-1)[:, 1]
            random_draw = torch.bernoulli(prob_2).int().cpu().numpy().reshape(-1, 1)
            labels = random_draw
        else:
            rational_labels = (sum_r_t_1 < sum_r_t_2).astype(int)
            labels = rational_labels.reshape(-1, 1)
        noise_mask = (np.random.rand(labels.shape[0]) <= self.teacher_eps_mistake)
        labels[noise_mask] = 1 - labels[noise_mask]
        labels[margin_mask] = -1
        # ★ 在此处添加一行代码，将原始计算的理性标签保存为 gt_labels，
        #     后续 VLM 修正均使用 gt_labels（ground-truth labels）以保持语义一致。
        gt_labels = labels.copy()
        
        # ------------------------------
        # B. 若不使用 VLM，则直接返回理性标签及相关数据
        # ------------------------------
        if not self.vlm_label:
            if not self.image_reward:
                return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
            if self.data_type == "image":
                return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels
            elif self.data_type == "pointcloud":
                return sa_t_1, sa_t_2, r_t_1, r_t_2, point_cloud_t_1, point_cloud_t_2, labels

        # ------------------------------
        # C. 使用 VLM 进行二次修正
        # ------------------------------
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        if self.vlm == 'pointllm_two_image':
            vlm_labels = []
            useful_indices = []
            for idx, (pc1, pc2) in enumerate(zip(point_cloud_t_1, point_cloud_t_2)):
                dist = np.linalg.norm(pc1 - pc2)
                if dist < 1e-3:
                    useful_indices.append(0)
                    vlm_labels.append(-1)
                else:
                    useful_indices.append(1)
                    prompt1 = pointllm_free_query_prompt1
                    prompt2 = pointllm_free_query_prompt2
                    main_prompt = pointllm_free_query_env_prompts[self.env_name]
                    summary_prompt = pointllm_summary_env_prompts[self.env_name]
                    res = pointllm_query_2([
                        prompt1, pc1.reshape(-1, 6),
                        prompt2, pc2.reshape(-1, 6),
                        main_prompt
                    ], summary_prompt)
                    try:
                        res_int = int(extract_label_from_response(res))
                        if res_int not in [0, 1, -1]:
                            res_int = -1
                    except:
                        res_int = -1
                    vlm_labels.append(res_int)
            vlm_labels = np.array(vlm_labels).reshape(-1, 1)
            useful_mask = (np.array(useful_indices) == 1) & (vlm_labels.reshape(-1) != -1)
            if (sa_t_1 is not None and sa_t_1.shape[0] != useful_mask.shape[0]) or (point_cloud_t_1.shape[0] != useful_mask.shape[0]):
                raise IndexError("数据长度不匹配！")
            if self.data_type == "pointcloud":
                point_cloud_t_1 = point_cloud_t_1[useful_mask]
                point_cloud_t_2 = point_cloud_t_2[useful_mask]
                r_t_1 = r_t_1[useful_mask]
                r_t_2 = r_t_2[useful_mask]
                gt_labels = gt_labels[useful_mask]
                vlm_labels = vlm_labels[useful_mask]
                return sa_t_1, sa_t_2, r_t_1, r_t_2, point_cloud_t_1, point_cloud_t_2, gt_labels, vlm_labels
            else:
                sa_t_1 = sa_t_1[useful_mask]
                sa_t_2 = sa_t_2[useful_mask]
                r_t_1 = r_t_1[useful_mask]
                r_t_2 = r_t_2[useful_mask]
                gt_labels = gt_labels[useful_mask]
                vlm_labels = vlm_labels[useful_mask]
                img_t_1 = img_t_1[useful_mask]
                img_t_2 = img_t_2[useful_mask]
                return sa_t_1, sa_t_2, r_t_1, r_t_2, point_cloud_t_1, point_cloud_t_2, img_t_1, img_t_2, gt_labels, vlm_labels
        else:
            # 其它 VLM 模式：采用原有逻辑，这里对图像数据进行额外预处理以确保维度正确
            gpt_two_image_paths = []
            combined_images_list = []
            file_path = os.path.abspath(__file__)
            dir_path = os.path.dirname(file_path)
            save_path = os.path.join(dir_path, "data", "gpt_query_image", self.env_name, time_string)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            useful_indices = []
            for idx, (im1, im2) in enumerate(zip(img_t_1, img_t_2)):
                im1 = robust_fix_image_array(im1)
                im2 = robust_fix_image_array(im2)
                combined_image = np.concatenate([im1, im2], axis=1)
                combined_images_list.append(combined_image)
                try:
                    combined_image_pil = Image.fromarray(combined_image)
                except Exception as e:
                    print("Error converting combined image to PIL, combined_image.shape =", combined_image.shape)
                    raise e
                first_image_save_path = os.path.join(save_path, f"first_{idx:06}.png")
                second_image_save_path = os.path.join(save_path, f"second_{idx:06}.png")
                Image.fromarray(robust_fix_image_array(im1)).save(first_image_save_path)
                Image.fromarray(robust_fix_image_array(im2)).save(second_image_save_path)
                gpt_two_image_paths.append([first_image_save_path, second_image_save_path])
                diff = np.linalg.norm(im1 - im2)
                if diff < 1e-3:
                    useful_indices.append(0)
                else:
                    useful_indices.append(1)
            if self.vlm == 'gpt4v_two_image':
                from vlms.gpt4_infer import gpt4v_infer_2
                vlm_labels = []
                for idx, (img_path_1, img_path_2) in enumerate(gpt_two_image_paths):
                    print(f"querying vlm {idx}/{len(gpt_two_image_paths)}")
                    query_prompt = gpt_free_query_env_prompts[self.env_name]
                    summary_prompt = gpt_summary_env_prompts[self.env_name]
                    res = gpt4v_infer_2(query_prompt, summary_prompt, img_path_1, img_path_2)
                    try:
                        label_res = int(res)
                    except:
                        label_res = -1
                    vlm_labels.append(label_res)
                    time.sleep(0.1)
            elif self.vlm == 'gemini_single_prompt':
                vlm_labels = []
                for idx, (im1, im2) in enumerate(zip(img_t_1, img_t_2)):
                    res = gemini_query_1([
                        gemini_free_query_prompt1,
                        Image.fromarray(robust_fix_image_array(im1)),
                        gemini_free_query_prompt2,
                        Image.fromarray(robust_fix_image_array(im2)),
                        gemini_single_query_env_prompts[self.env_name],
                    ])
                    try:
                        if "-1" in res:
                            res = -1
                        elif "0" in res:
                            res = 0
                        elif "1" in res:
                            res = 1
                        else:
                            res = -1
                    except:
                        res = -1
                    vlm_labels.append(res)
            elif self.vlm == "gemini_free_form":
                vlm_labels = []
                for idx, (im1, im2) in enumerate(zip(img_t_1, img_t_2)):
                    res = gemini_query_2([
                        gemini_free_query_prompt1,
                        Image.fromarray(robust_fix_image_array(im1)),
                        gemini_free_query_prompt2,
                        Image.fromarray(robust_fix_image_array(im2)),
                        gemini_free_query_env_prompts[self.env_name]
                    ],
                        gemini_summary_env_prompts[self.env_name]
                    )
                    try:
                        res = int(res)
                        if res not in [0, 1, -1]:
                            res = -1
                    except:
                        res = -1
                    vlm_labels.append(res)
            else:
                vlm_labels = []
            vlm_labels = np.array(vlm_labels).reshape(-1, 1)
            good_idx = (vlm_labels != -1).flatten()
            useful_indices = (np.array(useful_indices) == 1).flatten()
            good_idx = np.logical_and(good_idx, useful_indices)
            sa_t_1 = sa_t_1[good_idx]
            sa_t_2 = sa_t_2[good_idx]
            r_t_1 = r_t_1[good_idx]
            r_t_2 = r_t_2[good_idx]
            gt_labels = gt_labels[good_idx]
            vlm_labels = vlm_labels[good_idx]
            combined_images_list = np.array(combined_images_list)[good_idx]
            img_t_1 = img_t_1[good_idx]
            img_t_2 = img_t_2[good_idx]
            if self.flip_vlm_label:
                vlm_labels = 1 - vlm_labels
            if self.train_times % self.save_query_interval == 0 or 'gpt4v' in self.vlm:
                save_path = os.path.join(self.log_dir, "vlm_label_set")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(os.path.join(save_path, f"{time_string}.pkl"), "wb") as f:
                    pkl.dump([
                        combined_images_list, gt_labels, vlm_labels,
                        sa_t_1, sa_t_2, r_t_1, r_t_2
                    ], f, protocol=pkl.HIGHEST_PROTOCOL)
            acc = 0
            if len(vlm_labels) > 0:
                acc = np.sum(vlm_labels == gt_labels) / len(vlm_labels)
                print(f"vlm label acc: {acc}")
            else:
                print("no vlm label")
            self.vlm_label_acc = acc
            if self.data_type == "image":
                return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, gt_labels, vlm_labels
            else:
                return sa_t_1, sa_t_2, r_t_1, r_t_2, gt_labels, vlm_labels


    def get_label_from_cached_states(self):
        """
        从缓存标签文件中读取标签数据，避免重复调用 VLM 接口。

        返回：
            缓存的标签数据及对应的查询数据。
        """
        if self.read_cache_idx >= len(self.all_cached_labels):
            return None, None, None, None, None, []
        with open(self.all_cached_labels[self.read_cache_idx], 'rb') as f:
            data = pkl.load(f)
        self.read_cache_idx += 1
        combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = data
        return combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2

    # =============================================================================
    # 以下为不同采样策略，均利用优化后的 get_queries 和 get_label 函数实现高效采样
    # =============================================================================
    def kcenter_sampling(self):
        """
        K-Center Sampling 策略：使用状态-动作数据或点云数据，基于距离选择最具代表性的样本。

        返回：
            int: 成功写入缓冲区的标签数量。
        """
        num_init = self.mb_size * self.large_batch
        if self.data_type == "pointcloud":
            _, _, r_t_1, r_t_2, pc_t_1, pc_t_2 = self.get_queries(mb_size=num_init)
            temp_pc = np.concatenate([pc_t_1.reshape(pc_t_1.shape[0], -1),
                                      pc_t_2.reshape(pc_t_2.shape[0], -1)], axis=1)
            max_len = self.capacity if self.buffer_full else self.buffer_index
            tot_pc = self.buffer_seg1[:max_len].reshape(max_len, -1)
            selected_index = KCenterGreedy(temp_pc, tot_pc, self.mb_size)
            r_t_1 = r_t_1[selected_index]
            r_t_2 = r_t_2[selected_index]
            pc_t_1 = pc_t_1[selected_index]
            pc_t_2 = pc_t_2[selected_index]
            _, _, r_t_1, r_t_2, pc_t_1, pc_t_2, labels = self.get_label(None, None, r_t_1, r_t_2,
                                                                           point_cloud_t_1=pc_t_1, point_cloud_t_2=pc_t_2)
            if len(labels) > 0:
                self.put_queries(pc_t_1, pc_t_2, labels)
            return len(labels)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2, extra1, extra2 = self.get_queries(mb_size=num_init)
            temp_sa_t_1 = sa_t_1[:, :self.ds]
            temp_sa_t_2 = sa_t_2[:, :self.ds]
            temp_sa = np.concatenate([temp_sa_t_1, temp_sa_t_2], axis=1)
            max_len = self.capacity if self.buffer_full else self.buffer_index
            tot_sa_1 = self.buffer_seg1[:max_len].reshape(max_len, -1)
            tot_sa_2 = self.buffer_seg2[:max_len].reshape(max_len, -1)
            tot_sa = np.concatenate([tot_sa_1, tot_sa_2], axis=1)
            selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
            r_t_1 = r_t_1[selected_index]
            sa_t_1 = sa_t_1[selected_index]
            r_t_2 = r_t_2[selected_index]
            sa_t_2 = sa_t_2[selected_index]
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1=extra1, img_t_2=extra2)
            if len(labels) > 0:
                self.put_queries(sa_t_1, sa_t_2, labels)
            return len(labels)
    
    def kcenter_disagree_sampling(self):
        """
        K-Center Disagree Sampling 策略：
        基于模型预测不一致性选择样本进行标签生成。
        针对点云数据，由于状态-动作向量数据通常为 None，因此直接调用 uniform_sampling 策略。
        
        返回：
            int: 成功写入缓冲区的标签数量。
        """
        if self.data_type == "pointcloud":
            # 对于点云数据，采用 uniform_sampling（或可根据需要设计专用逻辑）
            return self.uniform_sampling()
        # 针对非点云情况，采用原有逻辑
        sa_t_1, sa_t_2, r_t_1, r_t_2, extra1, extra2 = self.get_queries(mb_size=self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1 = r_t_1[top_k_index]
        sa_t_1 = sa_t_1[top_k_index]
        r_t_2 = r_t_2[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1=extra1, img_t_2=extra2
        )
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    def kcenter_entropy_sampling(self):
        """
        K-Center Entropy Sampling 策略：
        基于预测熵选择样本进行标签生成。
        针对点云数据，同样直接采用 uniform_sampling 策略。
        
        返回：
            int: 成功写入缓冲区的标签数量。
        """
        if self.data_type == "pointcloud":
            # 对于点云数据，直接调用 uniform_sampling 以避免错误处理
            return self.uniform_sampling()
        num_init = self.mb_size * self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2, extra1, extra2 = self.get_queries(mb_size=num_init)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1 = r_t_1[top_k_index]
        sa_t_1 = sa_t_1[top_k_index]
        r_t_2 = r_t_2[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1=extra1, img_t_2=extra2
        )
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    
    def uniform_sampling(self):
        """
        Uniform Sampling 策略：
        根据是否使用图像奖励、点云奖励或状态-动作向量奖励，
        分别采样查询对并生成标签，最后将查询对及标签写入缓冲区。
        
        返回：
            int: 成功写入缓冲区的标签数量。
        """
        # 如果不使用 VLM 标注（即直接使用理性标签）
        if not self.vlm_label:
            # 针对数据类型为 "pointcloud" 的情况
            if self.data_type == "pointcloud":
                # 调用 get_queries 采样点云数据
                # 返回格式为: (sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2)
                sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2 = self.get_queries(mb_size=self.mb_size)
                # 使用 get_label 生成标签，并传入点云数据
                # 返回格式为: (sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, labels)
                sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, labels = self.get_label(
                    sa_t_1, sa_t_2, r_t_1, r_t_2,
                    point_cloud_t_1=pc_t_1, point_cloud_t_2=pc_t_2
                )
                # 将点云查询对和标签写入缓冲区
                if len(labels) > 0:
                    self.put_queries(pc_t_1, pc_t_2, labels)
            # 针对不使用图像奖励、且非点云情况（一般为状态-动作向量）的情况
            elif not self.image_reward:
                # 此分支调用 get_queries 返回状态-动作数据格式： (sa_t_1, sa_t_2, r_t_1, r_t_2)
                sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=self.mb_size)
                sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
                    sa_t_1, sa_t_2, r_t_1, r_t_2
                )
                if len(labels) > 0:
                    self.put_queries(sa_t_1, sa_t_2, labels)
            # 针对使用图像奖励的情况
            elif self.data_type == "image":
                # get_queries 返回格式为: (sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2)
                sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2 = self.get_queries(mb_size=self.mb_size)
                sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels = self.get_label(
                    sa_t_1, sa_t_2, r_t_1, r_t_2,
                    img_t_1=img_t_1, img_t_2=img_t_2
                )
                if len(labels) > 0:
                    self.put_queries(img_t_1, img_t_2, labels)
        else:
            # 当使用 VLM 标注时，分两种情况：有无缓存标签文件
            if self.cached_label_path is None:
                if self.vlm == 'pointllm_two_image':
                    if self.data_type == "pointcloud":
                        sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, _, _ = self.get_queries(mb_size=self.mb_size)
                        sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, labels, vlm_labels = self.get_label(
                            sa_t_1, sa_t_2, r_t_1, r_t_2,
                            point_cloud_t_1=pc_t_1, point_cloud_t_2=pc_t_2
                        )
                        if len(vlm_labels) > 0:
                            self.put_queries(pc_t_1, pc_t_2, vlm_labels)
                    elif self.data_type == "image":
                        sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2 = self.get_queries(mb_size=self.mb_size)
                        sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels = self.get_label(
                            sa_t_1, sa_t_2, r_t_1, r_t_2,
                            img_t_1=img_t_1, img_t_2=img_t_2
                        )
                        if len(vlm_labels) > 0:
                            self.put_queries(img_t_1, img_t_2, vlm_labels)
                else:
                    if self.data_type == "pointcloud":
                        sa_t_1, sa_t_2, r_t_1, r_t_2, _, _, img_t_1, img_t_2 = self.get_queries(mb_size=self.mb_size)
                        sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels = self.get_label(
                            sa_t_1, sa_t_2, r_t_1, r_t_2,
                            img_t_1=img_t_1, img_t_2=img_t_2
                        )
                        if len(vlm_labels) > 0:
                            self.put_queries(pc_t_1, pc_t_2, vlm_labels)
                    elif self.data_type == "image":
                        sa_t_1, sa_t_2, r_t_1, r_t_2, _, _, img_t_1, img_t_2 = self.get_queries(mb_size=self.mb_size)
                        sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels = self.get_label(
                            sa_t_1, sa_t_2, r_t_1, r_t_2,
                            img_t_1=img_t_1, img_t_2=img_t_2
                        )
                        if len(vlm_labels) > 0:
                            self.put_queries(img_t_1, img_t_2, vlm_labels)
            else:
                combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_label_from_cached_states()
                num, height, width, _ = combined_images_list.shape
                img_t_1 = combined_images_list[:, :, :width // 2, :]
                img_t_2 = combined_images_list[:, :, width // 2:, :]
                labels = vlm_labels
                if len(labels) > 0:
                    self.put_queries(img_t_1, img_t_2, labels)
        return len(labels)



    def disagreement_sampling(self):
        """
        Disagreement Sampling 策略：基于模型预测不一致性选择样本进行标签生成。

        返回：
            int: 成功写入缓冲区的标签数量。
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2, extra1, extra2 = self.get_queries(mb_size=self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1 = r_t_1[top_k_index]
        sa_t_1 = sa_t_1[top_k_index]
        r_t_2 = r_t_2[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2,
                                                               img_t_1=extra1, img_t_2=extra2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    
    def kcenter_entropy_sampling(self):
        """
        K-Center Entropy Sampling 策略：基于预测熵选择样本进行标签生成。

        返回：
            int: 成功写入缓冲区的标签数量。
        """
        num_init = self.mb_size * self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2, extra1, extra2 = self.get_queries(mb_size=num_init)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1 = r_t_1[top_k_index]
        sa_t_1 = sa_t_1[top_k_index]
        r_t_2 = r_t_2[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2,
                                                               img_t_1=extra1, img_t_2=extra2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    # =============================================================================
    # 单个输入数据奖励预测接口
    # =============================================================================
    def r_hat_member(self, x, member=-1):
        """
        对单个输入数据使用指定模型成员进行奖励预测。
        
        对于点云数据：
        - 外部要求输入数据格式为 (B, num_points, 6)，即每个样本包含若干点，每个点6维（例如 xyz+rgb）。
        - PointCloudNet 模型内部在 forward 方法中会执行一次转置操作：x = x.transpose(2, 1)，
            将数据从 (B, num_points, 6) 转为 (B, 6, num_points)，以满足 Conv1d 层的输入要求。
        
        因此，在本接口中，如果数据类型为 "pointcloud"，我们需要确保传入模型的数据格式为 (B, num_points, 6)。
        
        如果外部输入数据的形状为：
        - (B, num_points, 6)：说明数据格式正确，直接传入即可。
        - (B, 6, num_points)：则需要转置为 (B, num_points, 6)。
        - 如果数据存在冗余维度（例如 (B, 1, num_points, 6)），则先 squeeze 掉该冗余维度。
        
        参数：
        x: 输入数据（numpy 数组或 torch.Tensor）。
        member (int): 模型成员索引。
        
        返回：
        torch.Tensor: 模型输出奖励。
        """
        if isinstance(x, torch.Tensor):
            x_tensor = x.float().to(device)
        else:
            x_tensor = torch.from_numpy(x).float().to(device)
        
        # 针对点云数据进行额外的维度调整
        if self.data_type == "pointcloud":
            # 如果存在冗余的单一维度，例如 (B, 1, num_points, 6)，则移除该维度
            if x_tensor.ndim == 4 and x_tensor.size(1) == 1:
                x_tensor = x_tensor.squeeze(1)
            if x_tensor.ndim == 3:
                # 判断数据的形状：
                # 如果最后一维为6，则数据形状为 (B, num_points, 6)，符合 PointCloudNet 的要求，不需要转置
                # 如果最后一维不为6，而第二维为6，则数据形状为 (B, 6, num_points)，需要转置为 (B, num_points, 6)
                if x_tensor.shape[-1] != 6 and x_tensor.shape[1] == 6:
                    x_tensor = x_tensor.transpose(1, 2)
                # 若既不是 (B, 6, num_points) 也不是 (B, num_points, 6)，则保持原状（假定数据已符合要求）
        
        output = self.ensemble[member](x_tensor)
        return output



    def r_hat(self, x):
        """
        对单个输入数据进行奖励预测，使用所有模型成员的平均输出。

        参数：
            x: 输入数据。

        返回：
            float: 平均预测奖励。
        """
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)

    def r_hat_batch(self, x):
        """
        对一批输入数据进行奖励预测，使用所有模型成员的平均输出。

        参数：
            x: 输入数据。

        返回：
            np.array: 批量预测奖励。
        """
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats, axis=0)

    def save(self, model_dir, step):
        """
        保存当前 ensemble 中所有模型的参数至指定目录。

        参数：
            model_dir (str): 模型保存目录。
            step (int): 当前训练步数，用于文件命名。
        """
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(),
                f'{model_dir}/reward_model_{step}_{member}.pt'
            )

    def load(self, model_dir, step):
        """
        从指定目录加载 ensemble 中所有模型的参数。

        参数：
            model_dir (str): 模型加载目录。
            step (int): 保存时使用的步数，用于文件命名。
        """
        file_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(file_dir, model_dir)
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load(f'{model_dir}/reward_model_{step}_{member}.pt')
            )

    def get_train_acc(self):
        """
        计算当前缓冲区中数据的训练准确率，评估 ensemble 模型预测一致性。

        返回：
            float: 训练准确率。
        """
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        batch_size = 256
        num_epochs = int(np.ceil(max_len / batch_size))
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch + 1) * batch_size
            if last_index > max_len:
                last_index = max_len
            sa_t_1 = self.buffer_seg1[epoch * batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch * batch_size:last_index]
            labels = self.buffer_label[epoch * batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                r_hat1 = self.r_hat_member(sa_t_1, member=member).sum(axis=1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member).sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)

    # =============================================================================
    # 奖励模型训练：使用交叉熵损失对 <seg1, seg2> 进行二分类训练
    # 针对点云数据采用混合精度训练（AMP）及较小批量以降低显存占用
    # =============================================================================
    def train_reward(self):
        """
        训练奖励模型（硬标签），使用交叉熵损失对两个查询对输出进行二分类训练。
        采用混合精度训练以降低内存占用，对于点云数据自动使用较小的训练批量。

        返回：
            每个模型成员的训练准确率（平均）。
        """
        self.train_times += 1
        # 创建混合精度训练所需的 GradScaler
        scaler = torch.cuda.amp.GradScaler()
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        # 针对点云数据建议使用更小的批量
        train_batch = self.train_batch_size
        if self.data_type == "pointcloud":
            train_batch = min(self.train_batch_size, 16)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        # 为每个模型成员生成随机排列的索引
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len / train_batch))
        total = 0
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss_all = 0.0
            last_index = (epoch + 1) * train_batch
            if last_index > max_len:
                last_index = max_len
            for member in range(self.de):
                # 获取当前批次对应的索引
                idxs = total_batch_index[member][epoch * train_batch:last_index]
                # 提取存储在缓冲区中的数据
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                if member == 0:
                    total += labels.size(0)
                # 针对图像数据进行预处理：调整维度并归一化
                if self.data_type == "image":
                    # 将 (B,1,H,W,3) 转为 (B,1,3,H,W)，再 squeeze 成 (B,3,H,W)
                    sa_t_1 = np.transpose(sa_t_1, (0, 1, 4, 2, 3)).astype(np.float32) / 255.0
                    sa_t_2 = np.transpose(sa_t_2, (0, 1, 4, 2, 3)).astype(np.float32) / 255.0
                    sa_t_1 = sa_t_1.squeeze(1)
                    sa_t_2 = sa_t_2.squeeze(1)
                # 对于点云数据，无需额外预处理
                sa_t_1 = torch.from_numpy(sa_t_1).float().to(device)
                sa_t_2 = torch.from_numpy(sa_t_2).float().to(device)
                with torch.cuda.amp.autocast():
                    # 对两个输入分支分别进行奖励预测
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat2 = self.r_hat_member(sa_t_2, member=member)
                    # 针对图像数据（或非点云数据）将输出沿 axis=1 求和，
                    # 同时保持维度不变（即保持 shape 为 [B, 1]）
                    if self.data_type != "pointcloud":
                        r_hat1 = r_hat1.sum(axis=1, keepdim=True)
                        r_hat2 = r_hat2.sum(axis=1, keepdim=True)
                    # 将两个分支的输出沿第二个维度拼接，形成 shape 为 [batch_size, 2] 的 logits
                    r_hat = torch.cat([r_hat1, r_hat2], dim=1)
                    # 计算交叉熵损失，注意 labels 的 shape 应为 [B]
                    curr_loss = self.CEloss(r_hat, labels)
                loss_all += curr_loss
                # 统计预测准确个数
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            # 反向传播并更新梯度
            scaler.scale(loss_all).backward()
            scaler.step(self.opt)
            scaler.update()
            torch.cuda.empty_cache()
        ensemble_acc = ensemble_acc / total
        return ensemble_acc


    def train_soft_reward(self):
        """
        训练奖励模型（软标签），使用 soft cross entropy 损失对输出进行训练。
        同样采用混合精度训练以降低内存占用。

        返回：
            每个模型成员的训练准确率（平均）。
        """
        self.train_times += 1
        scaler = torch.cuda.amp.GradScaler()
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        # 针对点云数据建议使用更小的批量
        train_batch = self.train_batch_size
        if self.data_type == "pointcloud":
            train_batch = min(self.train_batch_size, 16)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        # 为每个模型成员生成随机排列的索引
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len / train_batch))
        total = 0
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss_all = 0.0
            last_index = (epoch + 1) * train_batch
            if last_index > max_len:
                last_index = max_len
            for member in range(self.de):
                # 提取当前批次的数据索引
                idxs = total_batch_index[member][epoch * train_batch:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                if member == 0:
                    total += labels.size(0)
                # 对图像数据进行预处理：调整维度并归一化
                if self.data_type == "image":
                    sa_t_1 = np.transpose(sa_t_1, (0, 1, 4, 2, 3)).astype(np.float32) / 255.0
                    sa_t_2 = np.transpose(sa_t_2, (0, 1, 4, 2, 3)).astype(np.float32) / 255.0
                    sa_t_1 = sa_t_1.squeeze(1)
                    sa_t_2 = sa_t_2.squeeze(1)
                sa_t_1 = torch.from_numpy(sa_t_1).float().to(device)
                sa_t_2 = torch.from_numpy(sa_t_2).float().to(device)
                with torch.cuda.amp.autocast():
                    # 分别计算两个分支的奖励预测
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat2 = self.r_hat_member(sa_t_2, member=member)
                    # 对非点云数据求和时保持维度不变
                    if self.data_type != "pointcloud":
                        r_hat1 = r_hat1.sum(axis=1, keepdim=True)
                        r_hat2 = r_hat2.sum(axis=1, keepdim=True)
                    # 拼接为 shape [batch_size, 2]
                    r_hat = torch.cat([r_hat1, r_hat2], dim=1)
                    # 对标签中值为 -1 的样本设置为 0（类别编号）
                    uniform_index = (labels == -1)
                    labels[uniform_index] = 0
                    # 构建 one-hot 目标，scatter 会将 labels 对应位置设置为 self.label_target
                    target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                    target_onehot += self.label_margin
                    # 对于被认为是均匀样本的（标签原本为 -1）直接设置为 0.5
                    if uniform_index.sum() > 0:
                        target_onehot[uniform_index] = 0.5
                    curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss_all += curr_loss
                # 统计预测正确的个数
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            # 反向传播更新梯度
            scaler.scale(loss_all).backward()
            scaler.step(self.opt)
            scaler.update()
            torch.cuda.empty_cache()
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
