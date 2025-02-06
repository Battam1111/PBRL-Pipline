# -*- coding: utf-8 -*-
"""
ImageSaver 模块：基于预训练 ResNet 提取图像 embedding 的保存器实现

核心功能：
  1. 使用预训练 ResNet18（去除最后全连接层）作为特征提取器，输出 512 维特征
  2. 对图像进行 Resize、归一化等预处理，提取特征后做 L2 归一化
  3. 保存图像为 .jpg 文件，同时生成对应的 meta 信息 (.json)，文件名中包含 bin_id、reward、time、embedding 的 MD5 哈希等信息
  
继承自 MultiClusterSaver，实现 _compute_embedding() 和 _do_save() 方法
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from dataCollection.DataSaver.multi_cluster_saver import MultiClusterSaver
from dataCollection.DataSaver.utils import md5_hash_for_array

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        """
        初始化 ResNetFeatureExtractor，使用预训练 ResNet18 去掉最后的全连接层，
        输出 512 维特征
        """
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：不计算梯度，输出 (B,512) 的特征向量
        """
        with torch.no_grad():
            feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        return feats

class ImageSaver(MultiClusterSaver):
    def __init__(self, task: str, output_dir: str, **kwargs):
        """
        初始化 ImageSaver
        
        参数：
          - task: 任务名称
          - output_dir: 输出目录
          - 其它参数由父类 MultiClusterSaver 处理
        """
        super().__init__(task=task, output_dir=output_dir, **kwargs)
        if self.compute_embedding:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            self.feature_extractor = ResNetFeatureExtractor(self.dim).eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor.to(self.device)
    
    def _compute_embedding(self, image_np):
        """
        计算图像 embedding：
          - 输入 image_np: numpy 数组格式的图像，形状 (H, W, 3)，dtype=uint8
          - 输出: (1,512) 的 numpy float32 数组
        """
        if image_np is None:
            return None
        pil_img = Image.fromarray(image_np)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(x)
        feats = feats / feats.norm(dim=1, keepdim=True)
        return feats.cpu().numpy().astype('float32')
    
    def _do_save(self, data, emb, reward, bin_id: int, sample_id: int, time_stamp: float):
        """
        保存图像及其 meta 信息：
          - 将图像保存为 .jpg 文件，文件名中包含 bin_id、reward、time 及 embedding 的 MD5 哈希
          - 将 meta 信息保存为 .json 文件
          - 返回 (data_filename, meta_filename, extra_files)
        """
        short_hash = md5_hash_for_array(emb)
        safe_bin = max(bin_id, 0)
        data_filename = f"img_{sample_id:06d}_bin_{safe_bin}_r_{reward:.2f}_t_{time_stamp:.2f}_emb_{short_hash}.jpg"
        data_path = os.path.join(self.data_dir, data_filename)
        Image.fromarray(data).save(data_path)
        
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
            "short_hash": short_hash
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_dict, f, indent=2)
        
        return data_filename, meta_filename, []
