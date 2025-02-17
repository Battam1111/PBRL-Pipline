#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sampling.py
===========
本模块实现优化版 K-Center Greedy 算法及相关辅助函数，
利用向量化操作和 torch.cdist 加速距离计算，用于从候选查询中选择具有代表性的样本。
"""

import numpy as np
import torch
from RewardModel.config import device

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
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist).item()
        selected_index.append(current_index[max_index])
        # 更新候选集：删除已选样本
        del current_index[max_index]
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([full_obs, obs[[selected_index[-1]]]], axis=0)
    return selected_index
