import os
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

import faiss  # pip install faiss-gpu 或 faiss-cpu


# -------------------- 通用基类 --------------------

class BaseSaver:
    """
    通用保存器基类，封装了图片和点云保存器的共同行为。
    子类需实现 _compute_embedding 和 _do_save 方法，以适应具体数据类型。
    """
    def __init__(
        self, 
        task,
        output_dir, 
        max_items=500, # 450个就足够组10w条偏好对数据
        sample_compare_size=256,
        use_ann_search=True,
        replace_strategy="random",
        faiss_index_type='flat',  # 新增参数，用于选择FAISS索引类型
        dim=512,
        **faiss_kwargs
    ):
        # 基本参数设置
        self.task = task
        self.output_dir = os.path.join(output_dir, task)
        self.max_items = max_items
        self.sample_compare_size = sample_compare_size
        self.use_ann_search = use_ann_search
        self.replace_strategy = replace_strategy

        # 创建保存目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 全局唯一ID，用于文件命名
        self.current_global_id = 0

        # 设备设置、FAISS索引和存储初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.faiss_index = None
        self.index2meta = {}
        self.dim = dim  # 嵌入向量维度
        if self.use_ann_search:
            self._init_faiss_index(index_type=faiss_index_type, dim=self.dim, **faiss_kwargs)

        # 使用GPU张量存储嵌入（非ANN模式下使用）
        self.saved_embeddings = torch.empty((0, self.dim), device=self.device)
        self.saved_records = []

        # 用于动态阈值调整的记录
        self.accepted_distances = []

        # 初始阶段的最小接受数量，内部参数无需用户干预
        self.min_initial = min(int(self.max_items * 0.05), self.max_items)

        # 初始化阈值为 None，表示尚未设置
        self.dist_thresh = None

    def _init_faiss_index(self, index_type='flat', dim=512, **faiss_kwargs):
        """
        初始化 FAISS 索引，支持多种索引类型。
        使用CPU上的IndexFlatL2以支持remove_ids操作。
        """
        if index_type == 'flat':
            index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

        # 使用IndexIDMap2以支持ID映射和删除操作
        self.faiss_index = faiss.IndexIDMap2(index)

        # 注意：这里不进行 GPU 转换，以确保remove_ids可用。

    def _get_current_count(self):
        """获取当前已保存的数据数量"""
        if self.use_ann_search:
            return self.faiss_index.ntotal
        else:
            return self.saved_embeddings.size(0)

    def _get_min_distance(self, emb_new):
        """
        计算新数据 embedding 与已保存数据的最小距离，
        并过滤掉 NaN 和 Inf 值。
        """
        if self.use_ann_search:
            distances, _ = self.faiss_index.search(emb_new, k=1)
            distance = distances[0][0]
        else:
            count = self.saved_embeddings.size(0)
            if count == 0:
                return float('inf')
            compare_size = min(self.sample_compare_size, count)
            indices = torch.randint(0, count, (compare_size,))
            sampled = self.saved_embeddings[indices]
            distances = torch.norm(sampled - emb_new, dim=1)
            distance = distances.min().item()
        return distance if np.isfinite(distance) else float('inf')

    def _replace_one(self, data, emb):
        """根据替换策略在数据库满时进行替换"""
        if self.replace_strategy == "random":
            self._replace_random(data, emb)
        elif self.replace_strategy == "nearest":
            self._replace_nearest(data, emb)
        else:
            raise ValueError(f"Unsupported replace strategy: {self.replace_strategy}")

    def _replace_random(self, data, emb):
        """随机替换已有数据"""
        count = self._get_current_count()
        if count == 0:
            return
        if self.use_ann_search:
            old_id = random.choice(list(self.index2meta.keys()))
            self._remove_by_id(old_id)
        else:
            idx = random.randint(0, self.saved_embeddings.size(0) - 1)
            if idx < len(self.saved_records):
                _, old_fname = self.saved_records[idx]
                old_path = os.path.join(self.output_dir, old_fname)
                if os.path.exists(old_path):
                    os.remove(old_path)
            self.saved_embeddings = torch.cat([self.saved_embeddings[:idx], self.saved_embeddings[idx+1:]], dim=0)
            if self.saved_records:
                self.saved_records.pop(idx)
        self._do_save(data, emb)

    def _replace_nearest(self, data, emb):
        """用最近邻策略替换最相似的数据"""
        count = self._get_current_count()
        if count == 0:
            return
        if self.use_ann_search:
            distances, ids = self.faiss_index.search(emb, k=1)
            old_id = int(ids[0][0])
            self._remove_by_id(old_id)
        else:
            distances = torch.norm(self.saved_embeddings - emb, dim=1)
            best_dist, best_idx = torch.min(distances, dim=0)
            best_idx = best_idx.item()
            if best_idx < len(self.saved_records):
                _, old_fname = self.saved_records[best_idx]
                old_path = os.path.join(self.output_dir, old_fname)
                if os.path.exists(old_path):
                    os.remove(old_path)
            self.saved_embeddings = torch.cat([self.saved_embeddings[:best_idx], self.saved_embeddings[best_idx+1:]], dim=0)
            if self.saved_records:
                self.saved_records.pop(best_idx)
        self._do_save(data, emb)

    def _remove_by_id(self, old_id):
        """
        使用 FAISS 提供的 remove_ids 方法通过ID删除指定数据，并清理相关资源。
        """
        # 移除文件和meta信息
        meta = self.index2meta.pop(old_id, None)
        if meta:
            old_path = os.path.join(self.output_dir, meta["filename"])
            if os.path.exists(old_path):
                os.remove(old_path)

        # 使用 FAISS 的 remove_ids 方法删除索引中的数据
        if self.use_ann_search:
            try:
                id_array = np.array([old_id], dtype=np.int64)
                self.faiss_index.remove_ids(id_array)
            except Exception as e:
                print(f"Warning: 删除ID {old_id}时出错：{e}")

        # 更新 non-ANN模式下的数据结构
        if not self.use_ann_search:
            new_records = []
            indices_to_keep = []
            for idx, (record_id, fname) in enumerate(self.saved_records):
                if record_id != old_id:
                    new_records.append((record_id, fname))
                    indices_to_keep.append(idx)
            self.saved_records = new_records
            if indices_to_keep:
                self.saved_embeddings = self.saved_embeddings[indices_to_keep]
            else:
                self.saved_embeddings = torch.empty((0, self.dim), device=self.device)

    def _update_threshold(self, new_distance):
        """
        更新相似度门槛基于已接受数据的距离分布，
        使用中位数作为新的阈值。
        """
        self.accepted_distances.append(new_distance)
        # 保持accepted_distances长度不超过max_items
        if len(self.accepted_distances) > self.max_items:
            self.accepted_distances.pop(0)
        # 更新阈值为当前接受距离的中位数
        self.dist_thresh = np.median(self.accepted_distances)

    # 抽象方法，由子类实现
    def _compute_embedding(self, data):
        raise NotImplementedError

    def _compute_embeddings_batch(self, data_list):
        raise NotImplementedError

    def _do_save(self, data, emb):
        raise NotImplementedError

    def save_data(self, data):
        """
        通用保存流程：
        1. 计算 embedding
        2. 判断相似度，决定保存、替换或过滤
        3. 自动调整阈值
        """
        emb = self._compute_embedding(data)

        if np.isnan(emb).any():
            print("Warning: 计算得到的embedding包含NaN，跳过保存。")
            return

        total_count = self._get_current_count()

        if total_count < self.min_initial:
            # 初始阶段，接受所有数据
            self._do_save(data, emb)
            # 计算与现有数据的最小距离
            min_dist = self._get_min_distance(emb)
            if min_dist < float('inf'):
                self.accepted_distances.append(min_dist)
                self._update_threshold(min_dist)
            else:
                # 第一个数据点，与自身的距离为0，设置为0
                self.accepted_distances.append(0.0)
                self._update_threshold(0.0)
            return

        # 达到最小初始数量后，判断相似性
        min_dist = self._get_min_distance(emb)

        if self.dist_thresh is None:
            self.dist_thresh = 0.0  # 若尚未设置阈值，则初始为0

        if min_dist < self.dist_thresh:
            # 如果新数据与现有数据过于相似，则拒绝
            return

        if total_count < self.max_items:
            self._do_save(data, emb)
        else:
            self._replace_one(data, emb)

        # 更新阈值
        self._update_threshold(min_dist)

    def batch_save_data(self, data_list):
        """
        批量保存流程：
        1. 批量计算 embeddings
        2. 对每个数据项进行相似度判断和保存决策
        """
        embeddings = self._compute_embeddings_batch(data_list)  # 形状: (B, 512)
        for data, emb in zip(data_list, embeddings):
            self.save_data(data)


# -------------------- 图像保存器 --------------------

class ResNetFeatureExtractor(nn.Module):
    """用于图像的 ResNet18 特征提取器"""
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            feats = self.features(x)
        return feats.view(feats.size(0), -1)


class ImageSaver(BaseSaver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.feature_extractor = ResNetFeatureExtractor().to(self.device).eval()

    def _compute_embedding(self, image_np):
        """计算输入图像的嵌入向量"""
        img_tensor = self.transform(Image.fromarray(image_np)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(img_tensor)
        emb = feats.cpu().numpy().astype('float32')
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    def _compute_embeddings_batch(self, image_list):
        """批量计算图像嵌入"""
        tensors = []
        for img_np in image_list:
            img = Image.fromarray(img_np)
            tensors.append(self.transform(img))
        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(batch_tensor)
        embeddings = feats.cpu().numpy().astype('float32')
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _do_save(self, image, emb):
        """保存图像文件及更新索引/列表"""
        file_name = f"image_{self.current_global_id:06d}.png"
        file_path = os.path.join(self.output_dir, file_name)
        Image.fromarray(image).save(file_path)

        if self.use_ann_search:
            self.faiss_index.add_with_ids(emb, np.array([self.current_global_id], dtype=np.int64))
            self.index2meta[self.current_global_id] = {"filename": file_name}
        else:
            self.saved_embeddings = torch.cat([
                self.saved_embeddings, 
                torch.from_numpy(emb).to(self.device)
            ], dim=0)
            self.saved_records.append((self.current_global_id, file_name))

        self.current_global_id += 1


# -------------------- 点云保存器 --------------------

class SimplePointNetFeatureExtractor(nn.Module):
    """简单且稳定的点云特征提取器，将 [N,6] 点云转换为512维向量"""
    def __init__(self, input_dim=6, output_dim=512):
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

    def forward(self, x):
        B, N, D = x.size()
        x = x.view(B * N, D)
        feats = self.mlp(x)
        feats = feats.view(B, N, -1)
        feats_mean = feats.mean(dim=1)
        norm = feats_mean.norm(p=2, dim=1, keepdim=True) + 1e-10
        return feats_mean / norm


class PointCloudSaver(BaseSaver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = SimplePointNetFeatureExtractor(input_dim=6, output_dim=512)\
                                    .to(self.device).eval()

    def _compute_embedding(self, pointcloud_np):
        """计算输入点云的嵌入向量"""
        if pointcloud_np.ndim != 2 or pointcloud_np.shape[1] != 6:
            raise ValueError(f"点云数据应为形状[N,6]，当前形状为{pointcloud_np.shape}")
        pc_tensor = torch.from_numpy(pointcloud_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(pc_tensor)
        emb = feats.cpu().numpy().astype('float32')
        if not np.all(np.isfinite(emb)):
            print("Warning: 特征提取后得到的embedding存在非有限值，进行归一化处理。")
            emb = np.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
            norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
            emb = emb / norm
        return emb

    def _compute_embeddings_batch(self, pointcloud_list):
        """批量计算点云嵌入"""
        tensors = []
        for pc in pointcloud_list:
            if pc.ndim != 2 or pc.shape[1] != 6:
                raise ValueError(f"点云数据应为形状[N,6]，当前形状为{pc.shape}")
            tensors.append(torch.from_numpy(pc).float())
        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(batch_tensor)
        embeddings = feats.cpu().numpy().astype('float32')
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _do_save(self, pointcloud, emb):
        """保存点云文件及更新索引/列表"""
        file_name = f"pointcloud_{self.current_global_id:06d}.npy"
        file_path = os.path.join(self.output_dir, file_name)
        np.save(file_path, pointcloud)

        if self.use_ann_search:
            self.faiss_index.add_with_ids(emb, np.array([self.current_global_id], dtype=np.int64))
            self.index2meta[self.current_global_id] = {"filename": file_name}
        else:
            self.saved_embeddings = torch.cat([
                self.saved_embeddings,
                torch.from_numpy(emb).to(self.device)
            ], dim=0)
            self.saved_records.append((self.current_global_id, file_name))

        self.current_global_id += 1
