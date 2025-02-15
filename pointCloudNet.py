#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointCloudNet 与 T-Net 模块：处理点云数据的神经网络
======================================================
本文件实现了 T-Net 和 PointCloudNet 模块，主要用于对点云数据进行仿射对齐以及特征提取，
最终预测单标量奖励。为解决在计算变换矩阵时偶发出现极大数值（数值爆炸）的情况，
代码中做了如下改进：
  1. 修改所有 GroupNorm 的 eps 参数为 1e-3，防止极小方差引起除零问题；
  2. 对所有卷积层和全连接层采用 Kaiming 正态初始化，保证激活函数下的数值稳定性；
  3. 在 T-Net 的 fc3 输出后，先经过 tanh 激活并乘以一个较小的尺度因子（例如 0.1），
     再加上单位矩阵，从而使得输出的变换矩阵始终处于单位矩阵附近，避免因过大偏差而导致数值爆炸；
  4. 对 PointCloudNet 中最后输出层 fc3 进行“微初始化”，将其权重缩放到极小尺度，确保初始输出稳定；
  5. 提供了变换矩阵正则化函数，用于计算 T-Net 输出与正交矩阵之间的偏差（训练时可作为正则项）。
  
作者：您的姓名
日期：2023-…（更新日期）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# T-Net模块：对输入点云进行仿射变换以对齐数据
# -----------------------------
class TNet(nn.Module):
    def __init__(self, k=6, use_groupnorm=True):
        """
        参数：
          k：输入数据的特征维度，本例中为6（例如xyz+rgb）
          use_groupnorm：是否使用 GroupNorm（本代码固定使用 GroupNorm，并设 eps=1e-3）
        """
        super(TNet, self).__init__()
        self.k = k

        # 定义归一化层，固定使用 GroupNorm，num_groups=1，并设置 eps=1e-3 防止除零
        def norm_layer(channels):
            return nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-3)

        # 卷积层部分（kernel_size=1，相当于对每个点独立进行线性映射）
        self.conv1 = nn.Conv1d(k, 64, kernel_size=1)
        self.norm1 = norm_layer(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.norm2 = norm_layer(128)
        
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.norm3 = norm_layer(1024)
        
        # 全连接层部分
        self.fc1 = nn.Linear(1024, 512)
        self.norm4 = norm_layer(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.norm5 = norm_layer(256)
        
        # 最后一层映射到 k*k，用于生成仿射变换矩阵的参数
        self.fc3 = nn.Linear(256, k * k)
        # 对 fc3 层进行特殊初始化：将权重、偏置均设为0，使得初始输出为0
        # 后续我们会对其输出进行 tanh 限幅，再加上单位矩阵，得到近似单位矩阵的变换
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.constant_(self.fc3.bias, 0)
        
        # 对除 fc3 外的所有卷积层和全连接层采用 Kaiming 正态初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc3:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        输入：
          x: 点云数据，形状为 (N, k, num_points)
        输出：
          变换矩阵，形状为 (N, k, k)
        """
        # 依次经过卷积层、GroupNorm和ReLU激活，提取局部特征
        x = F.relu(self.norm1(self.conv1(x)))    # 输出形状：(N, 64, num_points)
        x = F.relu(self.norm2(self.conv2(x)))      # 输出形状：(N, 128, num_points)
        x = F.relu(self.norm3(self.conv3(x)))      # 输出形状：(N, 1024, num_points)
        
        # 全局最大池化，沿着点维度聚合特征，得到 (N, 1024)
        x = torch.max(x, 2)[0]
        
        # 依次经过全连接层、GroupNorm和ReLU激活
        x = F.relu(self.norm4(self.fc1(x)))        # 输出形状：(N, 512)
        x = F.relu(self.norm5(self.fc2(x)))        # 输出形状：(N, 256)
        
        # 最后一层 fc3 输出 (N, k*k)
        x = self.fc3(x)
        # 为了避免输出数值过大，先对 fc3 输出进行 tanh 激活，再乘以一个较小尺度（例如 0.1）
        # 这样得到的扰动始终处于 [-0.1, 0.1] 范围内
        x = 0.1 * torch.tanh(x)
        
        # 将得到的扰动与单位矩阵相加，使得最终变换矩阵接近于单位矩阵
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(x.size(0), 1)
        x = x + identity
        
        # 重塑为 (N, k, k)
        x = x.view(-1, self.k, self.k)
        
        # 防护性检查：若输出中含有 NaN，则打印警告并返回单位矩阵
        if torch.isnan(x).any():
            print("警告：T-Net输出中检测到NaN，返回单位矩阵作为变换矩阵")
            x = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        return x

    def transformation_regularizer(self, trans):
        """
        计算变换矩阵正则项：使得 trans * trans^T 接近单位矩阵
        输入：
          trans: 变换矩阵，形状为 (N, k, k)
        返回：
          正则化损失（标量）
        """
        batchsize = trans.size(0)
        k = trans.size(1)
        # 计算 trans * trans^T
        trans_transpose = trans.transpose(2, 1)
        product = torch.bmm(trans, trans_transpose)
        identity = torch.eye(k, device=trans.device).unsqueeze(0).repeat(batchsize, 1, 1)
        loss = torch.mean(torch.norm(product - identity, dim=(1,2)))
        return loss

# -----------------------------
# PointCloudNet模块：处理点云数据，提取全局特征并预测奖励标量
# -----------------------------
class PointCloudNet(nn.Module):
    def __init__(self, num_points=8192, input_dim=6, normalize=True, use_groupnorm=True):
        """
        参数：
          num_points: 每个点云中的点数（默认8192）
          input_dim: 每个点的特征维度（默认6，即 xyz 和 rgb）
          normalize: 是否对输入点云进行中心化与尺度归一化（若外部已归一化可设为 False）
          use_groupnorm: 是否使用 GroupNorm，本代码固定使用 GroupNorm，并设 eps=1e-3
        """
        super(PointCloudNet, self).__init__()
        self.num_points = num_points
        self.input_dim = input_dim
        self.normalize = normalize

        # 定义归一化层（使用 GroupNorm，eps=1e-3）
        def norm_layer(channels):
            return nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-3)
        
        # T-Net模块，用于对输入点云进行仿射对齐，输入维度为 input_dim
        self.input_transform = TNet(k=input_dim, use_groupnorm=use_groupnorm)
        
        # 共享1D卷积层，逐点提取局部特征
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.norm1 = norm_layer(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.norm2 = norm_layer(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.norm3 = norm_layer(256)
        
        # 全连接层，将全局特征映射为奖励标量
        self.fc1 = nn.Linear(256, 256)
        self.norm4 = norm_layer(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.norm5 = norm_layer(128)
        
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(128, 1)
        # 对最后输出层 fc3 进行微初始化，使得权重尺度很小（例如标准差1e-3），以避免初始输出过大
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc3.bias, 0)
        
        # 对其它卷积层和全连接层（除fc3外）采用 Kaiming 正态初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc3:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        输入：
          x: 点云数据，形状为 (N, num_points, input_dim)
        输出：
          out: 奖励预测标量，形状为 (N, 1)
        说明：
          若 normalize 为 True，则先对点云数据进行中心化和尺度归一化；
          然后将输入转置为 (N, input_dim, num_points) 以适应1D卷积，
          利用 T-Net 对点云进行仿射对齐，再经过一系列卷积和全局池化提取全局特征，
          最后通过全连接层映射为单标量输出。
        """
        if self.normalize:
            # 计算点云中心
            centroid = torch.mean(x, dim=1, keepdim=True)  # 形状：(N, 1, input_dim)
            x = x - centroid
            # 计算各点到中心的欧氏距离
            distances = torch.sqrt(torch.sum(x ** 2, dim=2, keepdim=True))
            max_distance, _ = torch.max(distances, dim=1, keepdim=True)
            x = x / (max_distance + 1e-6)
        # 转置为 (N, input_dim, num_points) 以适应1D卷积
        x = x.transpose(2, 1)
        
        # 调用 T-Net 计算仿射变换矩阵（输出形状：(N, input_dim, input_dim)）
        trans = self.input_transform(x)
        
        # 使用仿射变换对点云进行对齐
        x = torch.bmm(trans, x)
        
        # 逐点卷积提取局部特征
        x = F.relu(self.norm1(self.conv1(x)))  # (N, 64, num_points)
        x = F.relu(self.norm2(self.conv2(x)))  # (N, 128, num_points)
        x = F.relu(self.norm3(self.conv3(x)))  # (N, 256, num_points)
        
        # 全局最大池化，提取全局特征 (N, 256)
        x = torch.max(x, 2)[0]
        
        # 全连接层映射
        x = F.relu(self.norm4(self.fc1(x)))    # (N, 256)
        x = F.relu(self.norm5(self.fc2(x)))    # (N, 128)
        x = self.dropout(x)
        out = self.fc3(x)                      # (N, 1)
        return out

# -----------------------------
# 工厂函数：生成处理点云数据的模型
# -----------------------------
def gen_point_cloud_net(num_points=8192, input_dim=6, device='cuda', normalize=True, use_groupnorm=True):
    """
    参数说明：
      num_points: 每个点云包含的点数
      input_dim: 每个点的特征维度（例如6表示xyz+rgb）
      device: 指定设备，例如 'cuda' 或 'cpu'
      normalize: 是否在网络内部对点云进行中心化和尺度归一化
      use_groupnorm: 是否使用 GroupNorm（本代码固定使用 GroupNorm，eps=1e-3）
    返回：
      一个处理点云数据的 PointCloudNet 模型
    """
    model = PointCloudNet(num_points=num_points, input_dim=input_dim, normalize=normalize, use_groupnorm=use_groupnorm)
    model = model.float().to(device)
    return model

# -----------------------------
# 测试代码示例
# -----------------------------
if __name__ == "__main__":
    # 生成随机点云数据（4个样本，每个包含8192个点，每个点6维：xyz 与 rgb）
    N = 4
    num_points = 8192
    input_dim = 6
    # 生成随机数（分布在 [0,1] 之间）
    dummy_input = torch.rand(N, num_points, input_dim)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 测试时设置 normalize=False（模拟外部已归一化情况）并启用 GroupNorm
    model = gen_point_cloud_net(num_points=num_points, input_dim=input_dim, device=device, normalize=False, use_groupnorm=True)
    dummy_input = dummy_input.to(device)
    
    # 前向传播，检查输出是否正确且数值稳定
    output = model(dummy_input)
    print("模型输出形状:", output.shape)  # 预期输出: (4, 1)
    print("模型输出:", output)
