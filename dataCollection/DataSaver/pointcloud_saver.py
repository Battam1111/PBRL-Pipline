# -*- coding: utf-8 -*-
"""
PointCloudSaver 模块：用于保存点云数据，基于简单 MLP 提取点云 embedding 的实现

核心功能：
  1. 接受点云数据，格式可为 (N,6) 的 numpy 数组或由多个块组成的 list
  2. 采用简单 MLP 模型提取点云特征，并进行 L2 归一化，输出 (1,512) 的 embedding
  3. 将第一块点云数据保存到 data_dir，其余块保存到 second_data_dir（若设置）
  4. 将 meta 信息保存到 meta_dir，同时支持在替换或重新分箱时同步处理 second_data_dir 中的文件
  
继承自 MultiClusterSaver，实现 _compute_embedding() 与 _do_save() 方法，并覆盖部分文件管理方法
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

from dataCollection.DataSaver.multi_cluster_saver import MultiClusterSaver
from dataCollection.DataSaver.utils import md5_hash_for_array

class SimplePointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=6, output_dim=512):
        """
        初始化简单点云特征提取器，采用 MLP 模型提取点云特征
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
          - 输入 x: 形状 (B, N, 6)
          - 将 x 展平为 (B*N,6)，经过 MLP 得到 (B*N,512)
          - 重塑为 (B, N, 512)，对 N 维取均值得到 (B,512)
          - 最后进行 L2 归一化
        """
        with torch.no_grad():
            B, N, D = x.size()
            x = x.view(B * N, D)
            feats = self.mlp(x)
        feats = feats.view(B, N, -1)
        feats_mean = feats.mean(dim=1)
        norm = feats_mean.norm(p=2, dim=1, keepdim=True) + 1e-10
        feats_mean = feats_mean / norm
        return feats_mean

class PointCloudSaver(MultiClusterSaver):
    def __init__(self, task: str, output_dir: str, second_output_dir: str = None, **kwargs):
        """
        初始化 PointCloudSaver
        
        参数：
          - task: 任务名称
          - output_dir: 主输出目录
          - second_output_dir: 第二数据目录（用于保存除第一块外的点云数据），可选
          - 其它参数传递给父类 MultiClusterSaver
        """
        super().__init__(task=task, output_dir=output_dir, **kwargs)
        self.second_output_dir_root = second_output_dir
        self.second_data_dir = None
        if second_output_dir:
            second_task_dir = os.path.join(second_output_dir, task)
            os.makedirs(second_task_dir, exist_ok=True)
            self.second_data_dir = os.path.join(second_task_dir, "")
            os.makedirs(self.second_data_dir, exist_ok=True)
        
        if self.compute_embedding:
            self.feature_extractor = SimplePointNetFeatureExtractor(6, self.dim).eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor.to(self.device)
    
    def _compute_embedding(self, pc_input):
        """
        计算点云 embedding：
          - 如果 pc_input 为 list，则取第 0 块数据进行计算
          - 如果为 numpy 数组，则直接计算
          - 要求输入形状为 (N,6)
        """
        if pc_input is None:
            return None
        if isinstance(pc_input, list):
            arr = pc_input[0]
        else:
            arr = pc_input
        
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"PointCloud 必须为 (N,6)，当前 shape 为 {arr.shape}")
        
        x = torch.from_numpy(arr).float().unsqueeze(0).to(self.device)
        feats = self.feature_extractor(x)
        return feats.cpu().numpy().astype('float32')
    
    def _do_save(self, data, emb, reward, bin_id: int, sample_id: int, time_stamp: float):
        """
        保存点云数据：
          - 如果 data 为 numpy 数组，则视为单块；如果为 list，则分块保存
          - 第一块数据保存到 data_dir，其余块保存到 second_data_dir（如果设置）
          - 保存对应的 meta 信息到 meta_dir
          - 返回 (主文件名, meta 文件名, [额外文件名列表])
        """
        if isinstance(data, np.ndarray):
            pc_list = [data]
        elif isinstance(data, list):
            pc_list = data
        else:
            raise ValueError("PointCloudSaver 仅接受 np.ndarray 或 list 格式的数据")
        
        short_hash = md5_hash_for_array(emb)
        safe_bin = max(bin_id, 0)
        
        main_fname = f"pc_{sample_id:06d}_bin_{safe_bin}_r_{reward:.2f}_t_{time_stamp:.2f}_emb_{short_hash}.npy"
        main_path = os.path.join(self.data_dir, main_fname)
        np.save(main_path, pc_list[0])
        
        extra_files = []
        for i in range(1, len(pc_list)):
            if self.second_data_dir:
                sec_fname = f"pc_{sample_id:06d}_bin_{safe_bin}_r_{reward:.2f}_t_{time_stamp:.2f}_emb_{short_hash}.npy"
                sec_path = os.path.join(self.second_data_dir, sec_fname)
                np.save(sec_path, pc_list[i])
                extra_files.append(sec_fname)
        
        meta_filename = f"meta_{sample_id:06d}.json"
        meta_path = os.path.join(self.meta_dir, meta_filename)
        
        emb_list = []
        if emb is not None:
            if emb.ndim == 2:
                emb_list = emb[0].tolist()
            else:
                emb_list = emb.tolist()
        
        meta_dict = {
            "sample_id": sample_id,
            "reward": reward,
            "time": time_stamp,
            "bin_id": bin_id,
            "embedding_dim": self.dim,
            "embedding": emb_list,
            "short_hash": short_hash,
            "extra_filenames": extra_files
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_dict, f, indent=2)
        
        return main_fname, meta_filename, extra_files
    
    def _get_second_data_path(self, fname: str, meta: dict):
        """
        如果 second_data_dir 存在，则返回对应文件路径
        """
        if not self.second_data_dir or not fname:
            return None
        return os.path.join(self.second_data_dir, fname)
    
    def _move_extra_files_if_needed(self, meta: dict, old_bin: int, new_bin: int):
        """
        当 bin 发生变化时，更新 extra_filenames 中存储的文件名，将旧 bin 标识替换为新 bin 标识
        """
        extra_files = meta.get('extra_filenames', [])
        if not extra_files or (old_bin == new_bin):
            return
        if not self.second_data_dir:
            return
        
        for i, old_fname in enumerate(extra_files):
            new_fname = self._rename_file_bin_if_needed(old_fname, old_bin, new_bin, meta, is_second=True)
            extra_files[i] = new_fname
        meta['extra_filenames'] = extra_files
    
    def _remove_extra_files_if_needed(self, meta: dict, bin_id: int):
        """
        删除该样本在 second_data_dir 中的额外文件
        """
        extra_files = meta.get('extra_filenames', [])
        if not extra_files:
            return
        
        for fname in extra_files:
            sec_path = self._get_second_data_path(fname, meta)
            if sec_path and os.path.exists(sec_path):
                os.remove(sec_path)
