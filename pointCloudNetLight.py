#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量版 PointCloudNet 模型架构
======================================================
本文件实现了轻量版的 T-Net（TNetLite）与 PointCloudNet（LightPointCloudNet）模块，
在保持输入格式 (N, num_points, input_dim) 与输出 (N, 1) 不变的前提下，
通过降低通道数和简化结构大幅降低内存占用。
同时仍采用自注意力模块以增强局部与全局特征融合，
并对 T-Net 输出和全连接层输出做了边界控制，确保训练时数值稳定。

作者：您的姓名
日期：2023-…（更新日期）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 自注意力模块：捕获点云中点与点之间的关系
# -----------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        """
        参数：
          in_channels: 输入特征通道数（例如128）
        """
        super(SelfAttentionBlock, self).__init__()
        self.in_channels = in_channels
        # 将输入投影到较低维度作为 query 与 key（降维比例为8）
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        # 投影为 value，保持原通道数
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        # 可学习参数 gamma 控制残差比例，初始为 0
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        输入：
          x: 特征张量，形状 (N, C, num_points)
        输出：
          out: 经自注意力模块处理后的特征，形状保持 (N, C, num_points)
        """
        N, C, L = x.size()  # L 为点数
        proj_query = self.query_conv(x)            # (N, C//8, L)
        proj_key   = self.key_conv(x)              # (N, C//8, L)
        # 计算注意力矩阵：将 query 转置后与 key 矩阵乘法，结果 (N, L, L)
        energy = torch.bmm(proj_query.transpose(2, 1), proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x)            # (N, C, L)
        out = torch.bmm(proj_value, attention.transpose(2, 1))  # (N, C, L)
        # 残差连接：gamma 控制注意力模块输出的比例
        out = self.gamma * out + x
        return out

# -----------------------------
# 轻量版 T-Net 模块（TNetLite）：对输入点云进行仿射对齐
# -----------------------------
class TNetLite(nn.Module):
    def __init__(self, k=6, use_groupnorm=True):
        """
        参数：
          k: 输入点的特征维度（例如6：xyz+rgb）
          use_groupnorm: 是否使用 GroupNorm，本代码固定使用 GroupNorm（eps=1e-3）
        """
        super(TNetLite, self).__init__()
        self.k = k

        # 定义 GroupNorm 层，eps=1e-3
        def norm_layer(channels):
            return nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-3)

        # 逐点卷积层：减少通道数以降低参数
        self.conv1 = nn.Conv1d(k, 32, kernel_size=1)
        self.norm1 = norm_layer(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.norm2 = norm_layer(64)
        
        self.conv3 = nn.Conv1d(64, 256, kernel_size=1)
        self.norm3 = norm_layer(256)
        
        # 全连接层部分
        self.fc1 = nn.Linear(256, 128)
        self.norm4 = norm_layer(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.norm5 = norm_layer(64)
        
        # 最后一层映射到 k*k，生成仿射变换参数
        self.fc3 = nn.Linear(64, k * k)
        # 将 fc3 的权重与偏置初始化为 0
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.constant_(self.fc3.bias, 0)
        
        # 对其它层采用 Kaiming 正态初始化
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
        x = F.relu(self.norm1(self.conv1(x)))    # (N, 32, num_points)
        x = F.relu(self.norm2(self.conv2(x)))      # (N, 64, num_points)
        x = F.relu(self.norm3(self.conv3(x)))      # (N, 256, num_points)
        
        x = torch.max(x, 2)[0]                     # 全局最大池化 (N, 256)
        x = F.relu(self.norm4(self.fc1(x)))        # (N, 128)
        x = F.relu(self.norm5(self.fc2(x)))        # (N, 64)
        x = self.fc3(x)                          # (N, k*k)
        x = 0.1 * torch.tanh(x)                    # 限幅到 [-0.1, 0.1]
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(x.size(0), 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        if torch.isnan(x).any():
            print("警告：TNetLite检测到NaN，返回单位矩阵")
            x = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        return x

    def transformation_regularizer(self, trans):
        """
        计算正则项，使得 trans * trans^T 接近单位矩阵
        """
        batchsize = trans.size(0)
        k = trans.size(1)
        trans_transpose = trans.transpose(2, 1)
        product = torch.bmm(trans, trans_transpose)
        identity = torch.eye(k, device=trans.device).unsqueeze(0).repeat(batchsize, 1, 1)
        loss = torch.mean(torch.norm(product - identity, dim=(1, 2)))
        return loss

# -----------------------------
# 轻量版 PointCloudNet 模块（LightPointCloudNet）：提取全局特征并预测奖励标量
# -----------------------------
class LightPointCloudNet(nn.Module):
    def __init__(self, num_points=8192, input_dim=6, normalize=True, use_groupnorm=True):
        """
        参数：
          num_points: 每个点云中的点数（默认8192）
          input_dim: 每个点的特征维度（默认6，即 xyz+rgb）
          normalize: 是否在网络内部对点云进行中心化与尺度归一化（若外部已归一化可设为 False）
          use_groupnorm: 是否使用 GroupNorm（本代码固定使用 GroupNorm，eps=1e-3）
        """
        super(LightPointCloudNet, self).__init__()
        self.num_points = num_points
        self.input_dim = input_dim
        self.normalize = normalize

        def norm_layer(channels):
            return nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-3)
        
        # 使用轻量版 T-Net 对输入点云进行仿射对齐
        self.input_transform = TNetLite(k=input_dim, use_groupnorm=use_groupnorm)
        
        # 逐点特征提取：将通道数降低以减少内存消耗
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=1)
        self.norm1 = norm_layer(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.norm2 = norm_layer(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1)
        self.norm3 = norm_layer(128)
        
        # 引入自注意力模块以捕获点间关系（输入通道为128）
        self.attention = SelfAttentionBlock(in_channels=128)
        
        # 全局特征聚合后，通过全连接层映射到输出标量
        self.fc1 = nn.Linear(128, 64)
        self.norm4 = norm_layer(64)
        
        self.fc2 = nn.Linear(64, 32)
        self.norm5 = norm_layer(32)
        
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(32, 1)
        # 对 fc3 层进行微初始化，确保输出尺度小
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc3.bias, 0)
        
        # 对其它层采用 Kaiming 正态初始化
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
          1. 若 normalize 为 True，则对输入点云进行中心化与尺度归一化；
          2. 转置为 (N, input_dim, num_points) 以适应 1D 卷积；
          3. 使用轻量版 T-Net 对点云进行仿射对齐；
          4. 通过逐点卷积提取低维特征，并利用自注意力模块增强点间信息；
          5. 全局最大池化后，通过全连接层映射为输出标量；
          6. 最终输出经过 clamp 限幅，保证数值稳定。
        """
        if self.normalize:
            # 中心化
            centroid = torch.mean(x, dim=1, keepdim=True)  # (N, 1, input_dim)
            x = x - centroid
            # 尺度归一化
            distances = torch.sqrt(torch.sum(x ** 2, dim=2, keepdim=True))
            max_distance, _ = torch.max(distances, dim=1, keepdim=True)
            x = x / (max_distance + 1e-6)
        # 转置为 (N, input_dim, num_points)
        x = x.transpose(2, 1)
        
        # T-Net 对输入进行对齐
        trans = self.input_transform(x)  # (N, input_dim, input_dim)
        x = torch.bmm(trans, x)
        
        # 逐点卷积提取特征
        x = F.relu(self.norm1(self.conv1(x)))  # (N, 32, num_points)
        x = F.relu(self.norm2(self.conv2(x)))    # (N, 64, num_points)
        x = F.relu(self.norm3(self.conv3(x)))    # (N, 128, num_points)
        
        # 自注意力模块增强点间关系
        x = self.attention(x)                    # (N, 128, num_points)
        
        # 全局最大池化聚合特征：获得 (N, 128)
        x = torch.max(x, 2)[0]
        
        # 全连接映射
        x = F.relu(self.norm4(self.fc1(x)))      # (N, 64)
        x = F.relu(self.norm5(self.fc2(x)))      # (N, 32)
        x = self.dropout(x)
        out = self.fc3(x)                        # (N, 1)
        # 对输出做 clamp 限幅，确保输出稳定
        out = torch.clamp(out, -100.0, 100.0)
        return out

# -----------------------------
# 工厂函数：生成轻量版 PointCloudNet 模型
# -----------------------------
def gen_point_cloud_net(num_points=8192, input_dim=6, device='cuda', normalize=True, use_groupnorm=True):
    """
    参数说明：
      num_points: 每个点云包含的点数
      input_dim: 每个点的特征维度（例如6表示xyz+rgb）
      device: 指定设备，例如 'cuda' 或 'cpu'
      normalize: 是否在网络内部对点云进行中心化与尺度归一化
      use_groupnorm: 是否使用 GroupNorm（固定使用 eps=1e-3）
    返回：
      一个轻量版 PointCloudNet 模型，其输入形状为 (N, num_points, input_dim)，输出为 (N, 1)
    """
    model = LightPointCloudNet(num_points=num_points, input_dim=input_dim, normalize=normalize, use_groupnorm=use_groupnorm)
    model = model.float().to(device)
    return model

# -----------------------------
# 测试代码示例
# -----------------------------
if __name__ == "__main__":
    # 生成随机点云数据（例如4个样本，每个包含8192个点，每个点6维：xyz 与 rgb）
    N = 4
    num_points = 8192
    input_dim = 6
    # 生成分布在 [0, 1] 之间的随机数模拟点云数据
    dummy_input = torch.rand(N, num_points, input_dim)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 测试时设置 normalize=False（模拟外部已归一化情况）并启用 GroupNorm
    model = gen_point_cloud_net(num_points=num_points, input_dim=input_dim, device=device, normalize=False, use_groupnorm=True)
    dummy_input = dummy_input.to(device)
    
    # 前向传播，检查输出形状和数值稳定性
    output = model(dummy_input)
    print("模型输出形状:", output.shape)  # 预期输出: (4, 1)
    print("模型输出:", output)
