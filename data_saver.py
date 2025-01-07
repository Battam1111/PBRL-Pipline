import os
import numpy as np
from PIL import Image
import random
import collections

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

import faiss  # pip install faiss-gpu (或 faiss-cpu)

class ResNetFeatureExtractor(nn.Module):
    """
    以 ResNet18 为例，可替换成任意预训练模型或自定义网络。
    输出 embedding 向量(512维) 用于相似度计算。
    """
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        # 去掉最后的全连接层，保留到 Global Pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # (batch, 512, 1, 1)

    def forward(self, x):
        with torch.no_grad():
            feats = self.features(x)  # (batch, 512, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (batch, 512)
        return feats


class ImageSaver:
    def __init__(
        self, 
        output_dir="/home/star/Yanjun/RL-VLM-F/test/images", 
        max_images=1000,
        dist_thresh=5.0,               # 初始dist阈值
        sample_compare_size=256,         # 当不使用ANN时, 随机抽样比较的数量
        use_ann_search=True,             # 是否使用Faiss近似搜索 (IndexHNSWFlat)
        replace_strategy="random",       # "random"或"nearest"
        hnsw_M=32,                       # HNSW图邻居数量
        efSearch=50,                     # HNSW搜索范围
        efConstruction=40,               # HNSW构建范围
        
        # ================ 自动调节阈值的相关参数 ================
        auto_tune_dist_thresh=True,      # 是否启用自动调节
        target_accept_ratio=0.3,         # 目标接受率(下限)
        upper_accept_ratio=0.7,          # 目标接受率(上限)
        adjust_factor=0.1,               # 每次调整幅度(例如0.1 => 调整10%)
        history_window=50,               # 在多少次尝试后才检查并调节
        warmup_saves=10,                 # 前多少次尝试不进行自动调节(热身)
    ):
        """
        :param output_dir:   图像保存目录
        :param max_images:   最大保存图像数量
        :param dist_thresh:  初始相似度过滤阈值
        :param sample_compare_size: 不用ANN时,随机对比embedding的数量
        :param use_ann_search: True => 用Faiss(HNSWFlat+IDMap)，False => list-based
        :param replace_strategy: 满库后的替换策略: "random" 或 "nearest"
        :param hnsw_M:     HNSW中的图构建参数
        :param efSearch:   HNSW搜索范围
        :param efConstruction: HNSW构建范围

        # 以下是自动调节dist_thresh的参数
        :param auto_tune_dist_thresh: 是否开启自适应阈值
        :param target_accept_ratio:    目标最低接受率, 若低于此值就增大阈值
        :param upper_accept_ratio:     接受率的上限, 若高于此值就减小阈值
        :param adjust_factor:          每次调节幅度, dist_thresh *= (1 +/- adjust_factor)
        :param history_window:         统计最近多少次"尝试"以计算接受率
        :param warmup_saves:          前多少次成功保存不进行调节(热身期)
        """
        self.output_dir = output_dir
        self.max_images = max_images
        self.dist_thresh = dist_thresh
        self.sample_compare_size = sample_compare_size
        self.use_ann_search = use_ann_search
        self.replace_strategy = replace_strategy
        
        self.auto_tune = auto_tune_dist_thresh
        self.target_accept_ratio = target_accept_ratio
        self.upper_accept_ratio = upper_accept_ratio
        self.adjust_factor = adjust_factor
        self.history_window = history_window
        self.warmup_saves = warmup_saves

        # 创建保存目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 全局自增ID(写文件时使用)
        self.current_global_id = 0
        
        # ================ 预处理与模型 ================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.feature_extractor = ResNetFeatureExtractor().to(self.device).eval()

        # ================ 索引结构 (HNSW + IDMap) ================
        self.faiss_index = None
        self.index2meta = {}
        if self.use_ann_search:
            self._init_hnsw_index(dim=512, hnsw_M=hnsw_M, efSearch=efSearch, efConstruction=efConstruction)

        # ================ 不用ANN时, list管理 ================
        self.saved_embeddings = []
        self.saved_records = []
        
        # ================ 自动调参相关统计 ================
        # 我们用一个队列记录最近的 (attempt_result) => 1表示"该次图像被保存", 0表示"被过滤"
        self.history_results = collections.deque(maxlen=self.history_window)
        # 同时我们统计已经成功保存多少次, 以支持 warmup
        self.save_count = 0

    def _init_hnsw_index(self, dim=512, hnsw_M=32, efSearch=50, efConstruction=40):
        # 构建 HNSWFlat 索引 + IDMap
        hnsw_index = faiss.IndexHNSWFlat(dim, hnsw_M, faiss.METRIC_L2)
        hnsw_index.hnsw.efSearch = efSearch
        hnsw_index.hnsw.efConstruction = efConstruction
        self.faiss_index = faiss.IndexIDMap(hnsw_index)

    def _compute_embedding(self, image_np):
        """
        将(Height,Width,3) => (1,3,224,224) => (1,512) embedding
        """
        img_tensor = self.transform(Image.fromarray(image_np)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(img_tensor)
        return feats.cpu().numpy()  # (1,512)

    def save_image(self, render_image):
        """
        主接口:
          1) 计算embedding
          2) 计算与已有数据的最小距离
          3) 若 < dist_thresh => 不保存(too similar)
          4) 若 >= dist_thresh => 保存/或满库时替换
          5) 记录本次尝试结果(1 or 0) => 用于自适应调节
        """
        emb = self._compute_embedding(render_image)
        total_count = self._get_current_count()

        # (A) 若库空 => 直接保存
        if total_count == 0:
            self._do_save(render_image, emb)
            # 记录: 此次为 "保存成功"
            self.history_results.append(1)
            self._try_auto_tune()
            return

        # (B) 获取min_dist
        min_dist = self._get_min_distance(emb)
        # (C) 判断是否过近
        if min_dist < self.dist_thresh:
            # 不保存
            self.history_results.append(0)
            self._try_auto_tune()
            return

        # (D) 超过阈值 => 保存或替换
        if total_count < self.max_images:
            self._do_save(render_image, emb)
        else:
            self._replace_one(render_image, emb)
        
        # (E) 记录成功
        self.history_results.append(1)
        self._try_auto_tune()

    def _get_current_count(self):
        if self.use_ann_search:
            return self.faiss_index.ntotal
        else:
            return len(self.saved_embeddings)

    def _get_min_distance(self, emb_new):
        if self.use_ann_search:
            distances, ids = self.faiss_index.search(emb_new, k=1)
            return distances[0][0]
        else:
            compare_size = min(self.sample_compare_size, len(self.saved_embeddings))
            indices = random.sample(range(len(self.saved_embeddings)), compare_size)
            min_dist = float('inf')
            for idx in indices:
                dist = np.linalg.norm(self.saved_embeddings[idx] - emb_new[0])
                if dist < min_dist:
                    min_dist = dist
            return min_dist

    def _do_save(self, render_image, emb):
        """
        真的往磁盘写文件 + 写索引/列表
        """
        file_name = f"image_{self.current_global_id:06d}.png"
        file_path = os.path.join(self.output_dir, file_name)
        Image.fromarray(render_image).save(file_path)

        if self.use_ann_search:
            self.faiss_index.add_with_ids(emb, np.array([self.current_global_id], dtype=np.int64))
            self.index2meta[self.current_global_id] = {"filename": file_name}
        else:
            self.saved_embeddings.append(emb[0])
            self.saved_records.append((self.current_global_id, file_name))

        self.current_global_id += 1
        self.save_count += 1  # 成功保存次数+1

    def _replace_one(self, render_image, emb):
        """
        当库满时, 执行替换策略
        """
        if self.replace_strategy == "random":
            self._replace_random(render_image, emb)
        elif self.replace_strategy == "nearest":
            self._replace_nearest(render_image, emb)
        else:
            # 未知策略 => 不保存
            pass

    def _replace_random(self, render_image, emb):
        count = self._get_current_count()
        if count == 0:
            return
        if self.use_ann_search:
            old_id = random.choice(list(self.index2meta.keys()))
            self._remove_by_id(old_id)
            self._do_save(render_image, emb)
        else:
            idx = random.randrange(len(self.saved_records))
            old_id, old_fname = self.saved_records[idx]
            old_path = os.path.join(self.output_dir, old_fname)
            if os.path.exists(old_path):
                os.remove(old_path)
            self.saved_embeddings.pop(idx)
            self.saved_records.pop(idx)
            self._do_save(render_image, emb)

    def _replace_nearest(self, render_image, emb):
        count = self._get_current_count()
        if count == 0:
            return
        if self.use_ann_search:
            distances, ids = self.faiss_index.search(emb, k=1)
            old_id = ids[0][0]
            self._remove_by_id(old_id)
            self._do_save(render_image, emb)
        else:
            best_dist = float('inf')
            best_idx = 0
            for i, old_emb in enumerate(self.saved_embeddings):
                dist = np.linalg.norm(old_emb - emb[0])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            old_id, old_fname = self.saved_records[best_idx]
            old_path = os.path.join(self.output_dir, old_fname)
            if os.path.exists(old_path):
                os.remove(old_path)
            self.saved_embeddings.pop(best_idx)
            self.saved_records.pop(best_idx)
            self._do_save(render_image, emb)

    def _remove_by_id(self, old_id):
        """
        删除 old_id 对应的数据(索引 & 文件)
        """
        if self.use_ann_search:
            remove_ids = np.array([old_id], dtype=np.int64)
            self.faiss_index.remove_ids(remove_ids)
            meta = self.index2meta.pop(old_id, None)
            if meta is not None:
                old_path = os.path.join(self.output_dir, meta["filename"])
                if os.path.exists(old_path):
                    os.remove(old_path)
        else:
            # 已在替换函数里做了删除
            pass

    # ================ 自动调节 dist_thresh 逻辑 ================
    def _try_auto_tune(self):
        """
        在每次保存/过滤后，都调用一次，检查:
          1) 是否启用了auto_tune
          2) 是否已经过了warmup阶段
          3) 统计最近history_window次尝试的接受率
          4) 若低于 target_accept_ratio => dist_thresh增大
             若高于 upper_accept_ratio  => dist_thresh减小
          5) 可对dist_thresh做安全范围限制, 比如>=1, <=1e5等
        """
        if not self.auto_tune:
            return

        # 是否已达 warmup_saves 次数(指成功保存次数)
        if self.save_count < self.warmup_saves:
            return

        # 若history不够(少于history_window次尝试), 等凑满后再调
        if len(self.history_results) < self.history_window:
            return

        # 计算最近 history_window 次的接受率
        arr = list(self.history_results)  # 0/1
        accept_ratio = sum(arr) / len(arr)

        if accept_ratio < self.target_accept_ratio:
            # 增加dist_thresh => 更宽松 => 更多保存
            old_val = self.dist_thresh
            self.dist_thresh *= (1.0 + self.adjust_factor)
            # 你可在此加日志:
            # print(f"[AutoTune] Accept ratio={accept_ratio:.2f}< target={self.target_accept_ratio}, dist_thresh {old_val:.2f}=>{self.dist_thresh:.2f}")
        elif accept_ratio > self.upper_accept_ratio:
            # 减小dist_thresh => 更严格 => 更少保存
            old_val = self.dist_thresh
            self.dist_thresh *= (1.0 - self.adjust_factor)
            # print(f"[AutoTune] Accept ratio={accept_ratio:.2f}> upper={self.upper_accept_ratio}, dist_thresh {old_val:.2f}=>{self.dist_thresh:.2f}")
        
        # 限制一下 dist_thresh 范围, 防止无限大或接近0
        self.dist_thresh = max(self.dist_thresh, 1.0)
        self.dist_thresh = min(self.dist_thresh, 10.0)
