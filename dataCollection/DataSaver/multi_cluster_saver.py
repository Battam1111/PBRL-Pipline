# 文件: /home/star/Yanjun/RL-VLM-F/dataCollection/DataSaver/multi_cluster_saver.py
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
MultiClusterSaver 模块：基于 (embedding + reward) 多维聚类分箱的实现  
  
核心功能：
  1. 在 do_update_bins() 中收集所有样本的 (embedding, reward*reward_cluster_scale) 拼接向量，
     并在 [min_bins, max_bins] 范围内通过 KMeans 与 silhouette_score 选择最佳聚类数；
  2. 对聚类结果进行后处理：根据聚类中心向量中奖励部分（最后一维）升序排序，
     重映射聚类标签，使得奖励越高的样本对应的 bin 编号越大；
  3. 通过聚类结果重新分箱，调用 reassign_all_bins() 完成样本 bin 分配，并保存最新的重排序聚类中心；
  4. 在插入新样本时，根据当前（已排序）聚类中心计算新样本与各中心的距离，
     返回距离最小的中心索引作为新样本的 bin 编号，从而保证 bin 编号与奖励呈正相关。

参数：
  - reward_cluster_scale: 控制 reward 在聚类时的相对权重，默认 0.3。
  
说明：  
   本模块依赖 BaseSaver 提供的多线程安全数据保存、文件管理、FAISS 索引构建等功能，
   并在此基础上扩展了奖励与 bin 关联的逻辑。
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from dataCollection.DataSaver.base_saver import BaseSaver

class MultiClusterSaver(BaseSaver):
    def __init__(self, *args, reward_cluster_scale: float = 0.3, **kwargs):
        """
        初始化 MultiClusterSaver 实例，并设置 reward_cluster_scale 参数。
        
        参数：
          - reward_cluster_scale: 在聚类时对 reward 进行缩放的因子，默认值为 0.3。
          - 其它参数由 BaseSaver 负责（如 task、output_dir、desired_total、dim 等）。
        """
        super().__init__(*args, **kwargs)
        self.reward_cluster_scale = reward_cluster_scale
        self.cluster_centers_ = None  # 聚类中心，形状为 (k, dim+1)

    def do_update_bins(self):
        """
        执行多维聚类重新分箱操作，确保生成的 bin 编号与奖励成正比。
        
        具体流程：
          1. 若当前样本总数不足 desired_total 或不足 min_bins，则跳过分箱；
          2. 遍历所有样本，将每个样本的 embedding（展平后）与经过 reward_cluster_scale 缩放的 reward 拼接，
             得到拼接向量，存入矩阵 X；
          3. 在 [min_bins, max_bins] 范围内依次尝试 KMeans 聚类，计算 silhouette_score，选择最佳聚类数 k；
          4. 利用最佳 k 值进行聚类，获得最终聚类标签 labs_final 和聚类中心 centers（形状为 (k, dim+1)）；
          5. 对聚类中心按照最后一维（奖励部分）升序排序，生成排序索引 sorted_order；
          6. 根据 sorted_order 创建映射 mapping：原始聚类标签 → 新标签，新标签即为最终 bin 编号（保证奖励低的聚类获得较小编号）；
          7. 重映射所有样本的聚类标签，得到新的分箱分配 new_assignments；
          8. 如果最佳聚类数 k 与当前 bin 数不一致，则调用 set_bin_count() 和 create_bin_structs() 更新分箱结构；
          9. 调用 reassign_all_bins(new_assignments) 完成样本的 bin 更新；
          10. 更新 self.cluster_centers_ 为重排序后的聚类中心数组 new_centers，以便后续新样本插入时使用。
        """
        total_count = self.get_total_count()
        if total_count < self.desired_total or total_count < self.min_bins:
            return

        sids = []
        X = []
        # 遍历所有样本，构造拼接向量：embedding 展平后与 reward*reward_cluster_scale 拼接
        for sid, meta in self.sample_meta.items():
            emb = meta['embedding']
            r = meta['reward']
            if emb is None:
                continue
            e_flat = emb.flatten()
            scaled_r = r * self.reward_cluster_scale
            vec = np.concatenate([e_flat, np.array([scaled_r], dtype=e_flat.dtype)])
            sids.append(sid)
            X.append(vec)
        X = np.array(X, dtype=np.float32)
        if len(X) < 2:
            return

        best_k = self.min_bins
        best_score = -999
        max_k = min(self.max_bins, len(X))
        best_labels = None

        # 尝试不同 k 值，选择 silhouette_score 最高的 k
        for k in range(self.min_bins, max_k + 1):
            if k == 1:
                labs = np.zeros(len(X), dtype=int)
                score = -1
            else:
                km = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42)
                labs = km.fit_predict(X)
                if len(set(labs)) < 2:
                    score = -1
                else:
                    score = silhouette_score(X, labs)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labs

        if best_labels is None:
            return

        # 利用最佳 k 值进行聚类，获得最终聚类标签和聚类中心
        km_final = KMeans(n_clusters=best_k, init='k-means++', n_init=5, random_state=42)
        labs_final = km_final.fit_predict(X)
        centers = km_final.cluster_centers_  # 形状 (best_k, dim+1)

        # --- 关键步骤：对聚类中心按奖励部分（最后一维）升序排序 ---
        sorted_order = np.argsort(centers[:, -1])
        # 重新排序聚类中心，保证第 0 个中心奖励最小，第 k-1 个中心奖励最大
        new_centers = centers[sorted_order]
        # 构建映射：原始聚类标签 -> 新标签（新标签即为 bin 编号，保证奖励低的分在前面）
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_order)}

        # 重映射每个样本的聚类标签，生成新的分箱分配字典 new_assignments
        new_assignments = {}
        idx_ptr = 0
        for sid, meta in self.sample_meta.items():
            if meta['embedding'] is not None:
                orig_label = int(labs_final[idx_ptr])
                new_label = label_mapping.get(orig_label, 0)
                new_assignments[sid] = new_label
                idx_ptr += 1
            else:
                new_assignments[sid] = 0

        # 如果最佳聚类数与当前 bin 数不一致，则更新 bin 数量及结构
        if best_k != self.bin_count:
            self.set_bin_count(best_k)
            self.create_bin_structs()

        # 根据新分箱分配重新分配所有样本
        self.reassign_all_bins(new_assignments)
        # 更新聚类中心为重排序后的结果，确保后续新样本分配时使用的中心顺序与奖励正相关
        self.cluster_centers_ = new_centers

    def _assign_bin_for_new_sample(self, emb: np.ndarray, reward: float, time_stamp: float):
        """
        插入新样本时，根据当前聚类中心判断其所属 bin。  
        具体逻辑：
          - 若当前未有聚类中心或 emb 为 None，则直接返回 0；
          - 否则，将新样本的 embedding 展平后与经过 reward_cluster_scale 缩放的 reward 拼接，
            得到向量 vcat；
          - 计算 vcat 与每个聚类中心之间的欧氏距离，返回距离最小的中心的索引；
          - 由于在 do_update_bins() 中已将聚类中心按奖励升序排列，因此返回的索引保证了奖励较低的样本获得较小的 bin 编号，
            奖励较高的样本获得较高的 bin 编号。
        
        参数：
          - emb: 新样本的 embedding（numpy 数组）
          - reward: 新样本的奖励
          - time_stamp: 新样本的插入时间戳（此处未参与计算，可供后续扩展）
        返回：
          - bin_id: 新样本应分配的 bin 编号（整数）
        """
        if (self.cluster_centers_ is None) or (emb is None):
            return 0
        scaled_r = reward * self.reward_cluster_scale
        e_flat = emb.flatten()
        # 拼接 embedding 与缩放后的 reward 形成查询向量
        vcat = np.concatenate([e_flat, np.array([scaled_r], dtype=e_flat.dtype)])
        # 计算与所有聚类中心的欧氏距离
        dists = np.linalg.norm(self.cluster_centers_ - vcat, axis=1)
        # 返回距离最小的聚类中心对应的 bin 编号（此时的编号已经是按奖励升序排序）
        return int(np.argmin(dists))
