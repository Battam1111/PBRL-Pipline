#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py
===========
本模块提供奖励模型训练相关的函数和类，包括硬标签和软标签训练流程、
预测接口、模型保存与加载等功能。为降低显存占用，均采用混合精度训练（AMP）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from RewardModel.config import device

class RewardModelTrainer:
    def __init__(self, ensemble, paramlst, lr, data_type, train_batch_size, resnet=False,
                 teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, label_margin=0.0, teacher_eps_equal=0):
        """
        初始化奖励模型训练器。

        参数：
            ensemble (list): 模型集合。
            paramlst (iterable): 模型参数列表。
            lr (float): 学习率。
            data_type (str): 数据类型，"image" 或 "pointcloud"。
            train_batch_size (int): 训练批次大小。
            resnet (bool): 是否使用 ResNet 架构（针对图像数据）。
            teacher_beta (float): 教师参数 beta。
            teacher_gamma (float): 教师参数 gamma。
            teacher_eps_mistake (float): 教师错误概率。
            label_margin (float): 标签边缘。
            teacher_eps_equal (float): 教师相等误差参数。
        """
        self.ensemble = ensemble
        self.paramlst = paramlst
        self.lr = lr
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)
        self.data_type = data_type
        self.train_batch_size = train_batch_size
        self.resnet = resnet
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.label_margin = label_margin
        self.teacher_eps_equal = teacher_eps_equal
        self.label_target = 1 - 2 * self.label_margin
        self.CEloss = nn.CrossEntropyLoss()
        self.train_times = 0

    def r_hat_member(self, model, x):
        """
        对单个输入数据使用指定模型进行奖励预测。

        参数：
            model (nn.Module): 模型。
            x: 输入数据（numpy数组或 torch.Tensor）。
        返回：
            torch.Tensor: 模型输出奖励。
        """
        if isinstance(x, torch.Tensor):
            x_tensor = x.float().to(device)
        else:
            x_tensor = torch.from_numpy(x).float().to(device)
        if self.data_type == "pointcloud":
            if x_tensor.ndim == 4 and x_tensor.size(1) == 1:
                x_tensor = x_tensor.squeeze(1)
            if x_tensor.ndim == 3:
                if x_tensor.shape[-1] != 6 and x_tensor.shape[1] == 6:
                    x_tensor = x_tensor.transpose(1,2)
        output = model(x_tensor)
        return output

    def r_hat(self, x):
        """
        使用所有模型成员平均输出进行奖励预测。

        参数：
            x: 输入数据。
        返回：
            float: 平均预测奖励。
        """
        r_hats = [self.r_hat_member(model, x).detach().cpu().numpy() for model in self.ensemble]
        r_hats = np.array(r_hats)
        return np.mean(r_hats)

    def r_hat_batch(self, x):
        """
        对一批数据进行奖励预测。

        参数：
            x: 输入数据。
        返回：
            np.array: 批量预测奖励。
        """
        r_hats = [self.r_hat_member(model, x).detach().cpu().numpy() for model in self.ensemble]
        r_hats = np.array(r_hats)
        return np.mean(r_hats, axis=0)

    def save_models(self, model_dir, step):
        """
        保存所有模型参数至指定目录。

        参数：
            model_dir (str): 模型保存目录。
            step (int): 当前训练步数，用于文件命名。
        """
        for i, model in enumerate(self.ensemble):
            torch.save(model.state_dict(), f'{model_dir}/reward_model_{step}_{i}.pt')

    def load_models(self, model_dir, step):
        """
        从指定目录加载所有模型参数。

        参数：
            model_dir (str): 模型加载目录。
            step (int): 保存时使用的步数。
        """
        import os
        base_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(base_dir, model_dir)
        for i, model in enumerate(self.ensemble):
            model.load_state_dict(torch.load(f'{model_dir}/reward_model_{step}_{i}.pt'))

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

    def get_train_acc(self, buffer_seg1, buffer_seg2, buffer_label, capacity, buffer_full):
        """
        计算缓冲区数据的训练准确率，评估模型预测一致性。

        参数：
            buffer_seg1 (np.array): 缓冲区分段1数据。
            buffer_seg2 (np.array): 缓冲区分段2数据。
            buffer_label (np.array): 缓冲区标签。
            capacity (int): 缓冲区容量。
            buffer_full (bool): 是否满缓冲区。
        返回：
            float: 平均训练准确率。
        """
        ensemble_acc = np.zeros(len(self.ensemble))
        max_len = capacity if buffer_full else buffer_label.shape[0]
        batch_size = 256
        num_epochs = int(np.ceil(max_len / batch_size))
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if last_index > max_len:
                last_index = max_len
            sa_t_1 = buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = buffer_seg2[epoch*batch_size:last_index]
            labels = torch.from_numpy(buffer_label[epoch*batch_size:last_index].flatten()).long().to(device)
            total += labels.size(0)
            for i, model in enumerate(self.ensemble):
                r_hat1 = self.r_hat_member(model, sa_t_1).sum(axis=1)
                r_hat2 = self.r_hat_member(model, sa_t_2).sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                _, predicted = torch.max(r_hat.data, 1)
                ensemble_acc[i] += (predicted == labels).sum().item()
        ensemble_acc = ensemble_acc/total
        return np.mean(ensemble_acc)

    def train_reward(self, buffer_seg1, buffer_seg2, buffer_label, capacity, buffer_full):
        """
        使用交叉熵损失训练奖励模型（硬标签），采用混合精度训练降低显存占用。

        参数：
            buffer_seg1, buffer_seg2, buffer_label: 缓冲区数据。
            capacity (int): 缓冲区容量。
            buffer_full (bool): 缓冲区是否已满。
        返回：
            np.array: 各模型成员平均训练准确率。
        """
        self.train_times += 1
        scaler = torch.cuda.amp.GradScaler()
        ensemble_acc = np.zeros(len(self.ensemble))
        train_batch = self.train_batch_size
        if self.data_type == "pointcloud":
            train_batch = min(self.train_batch_size, 16)
        max_len = capacity if buffer_full else buffer_label.shape[0]
        total_batch_index = [np.random.permutation(max_len) for _ in self.ensemble]
        num_epochs = int(np.ceil(max_len / train_batch))
        total = 0
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss_all = 0.0
            last_index = (epoch+1)*train_batch
            if last_index > max_len:
                last_index = max_len
            for i, model in enumerate(self.ensemble):
                idxs = total_batch_index[i][epoch*train_batch:last_index]
                sa_t_1_batch = buffer_seg1[idxs]
                sa_t_2_batch = buffer_seg2[idxs]
                labels = torch.from_numpy(buffer_label[idxs].flatten()).long().to(device)
                if i == 0:
                    total += labels.size(0)
                if self.data_type == "image":
                    sa_t_1_batch = np.transpose(sa_t_1_batch, (0,1,4,2,3)).astype(np.float32)/255.0
                    sa_t_2_batch = np.transpose(sa_t_2_batch, (0,1,4,2,3)).astype(np.float32)/255.0
                    sa_t_1_batch = sa_t_1_batch.squeeze(1)
                    sa_t_2_batch = sa_t_2_batch.squeeze(1)
                sa_t_1_batch = torch.from_numpy(sa_t_1_batch).float().to(device)
                sa_t_2_batch = torch.from_numpy(sa_t_2_batch).float().to(device)
                with torch.cuda.amp.autocast():
                    r_hat1 = self.r_hat_member(model, sa_t_1_batch)
                    r_hat2 = self.r_hat_member(model, sa_t_2_batch)
                    if self.data_type != "pointcloud":
                        r_hat1 = r_hat1.sum(axis=1, keepdim=True)
                        r_hat2 = r_hat2.sum(axis=1, keepdim=True)
                    r_hat = torch.cat([r_hat1, r_hat2], dim=1)
                    curr_loss = self.CEloss(r_hat, labels)
                loss_all += curr_loss
                _, predicted = torch.max(r_hat.data, 1)
                ensemble_acc[i] += (predicted == labels).sum().item()
            scaler.scale(loss_all).backward()
            scaler.step(self.opt)
            scaler.update()
            torch.cuda.empty_cache()
        ensemble_acc = ensemble_acc/total
        return ensemble_acc

    def train_soft_reward(self, buffer_seg1, buffer_seg2, buffer_label, capacity, buffer_full):
        """
        使用 soft cross entropy 损失训练奖励模型（软标签），采用混合精度训练降低显存占用。

        参数同 train_reward。
        返回：
            np.array: 各模型成员平均训练准确率。
        """
        self.train_times += 1
        scaler = torch.cuda.amp.GradScaler()
        ensemble_acc = np.zeros(len(self.ensemble))
        train_batch = self.train_batch_size
        if self.data_type == "pointcloud":
            train_batch = min(self.train_batch_size, 16)
        max_len = capacity if buffer_full else buffer_label.shape[0]
        total_batch_index = [np.random.permutation(max_len) for _ in self.ensemble]
        num_epochs = int(np.ceil(max_len/train_batch))
        total = 0
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss_all = 0.0
            last_index = (epoch+1)*train_batch
            if last_index > max_len:
                last_index = max_len
            for i, model in enumerate(self.ensemble):
                idxs = total_batch_index[i][epoch*train_batch:last_index]
                sa_t_1_batch = buffer_seg1[idxs]
                sa_t_2_batch = buffer_seg2[idxs]
                labels = torch.from_numpy(buffer_label[idxs].flatten()).long().to(device)
                if i == 0:
                    total += labels.size(0)
                if self.data_type == "image":
                    sa_t_1_batch = np.transpose(sa_t_1_batch, (0,1,4,2,3)).astype(np.float32)/255.0
                    sa_t_2_batch = np.transpose(sa_t_2_batch, (0,1,4,2,3)).astype(np.float32)/255.0
                    sa_t_1_batch = sa_t_1_batch.squeeze(1)
                    sa_t_2_batch = sa_t_2_batch.squeeze(1)
                sa_t_1_batch = torch.from_numpy(sa_t_1_batch).float().to(device)
                sa_t_2_batch = torch.from_numpy(sa_t_2_batch).float().to(device)
                with torch.cuda.amp.autocast():
                    r_hat1 = self.r_hat_member(model, sa_t_1_batch)
                    r_hat2 = self.r_hat_member(model, sa_t_2_batch)
                    if self.data_type != "pointcloud":
                        r_hat1 = r_hat1.sum(axis=1, keepdim=True)
                        r_hat2 = r_hat2.sum(axis=1, keepdim=True)
                    r_hat = torch.cat([r_hat1, r_hat2], dim=1)
                    uniform_index = (labels == -1)
                    labels[uniform_index] = 0
                    target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                    target_onehot += self.label_margin
                    if uniform_index.sum() > 0:
                        target_onehot[uniform_index] = 0.5
                    curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss_all += curr_loss
                _, predicted = torch.max(r_hat.data, 1)
                ensemble_acc[i] += (predicted == labels).sum().item()
            scaler.scale(loss_all).backward()
            scaler.step(self.opt)
            scaler.update()
            torch.cuda.empty_cache()
        ensemble_acc = ensemble_acc/total
        return ensemble_acc
