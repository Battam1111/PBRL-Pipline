# -*- coding: utf-8 -*-
"""
BaseSaver 模块：提供通用的多线程安全数据保存功能

核心功能点：
  1. 并发安全：通过多线程锁 (use_lock) 实现安全操作
  2. 异步 rebin：支持可选的异步 rebin 操作 (rebin_async)
  3. EXACT fill 机制：当当前样本数量小于 desired_total 时，直接插入样本（不进行相似度过滤）
  4. 相似度阈值：利用所有插入时计算的最近邻距离的中位数动态更新阈值
  5. 替换策略：当目标 bin 满时，根据 replace_strategy（"nearest" 或 "random"）执行替换
  6. 文件结构：数据文件存放在 data_dir，元信息存放在 meta_dir；支持同步处理 second_data_dir（由子类覆盖）
  7. 每个 bin 都配有一个 FAISS 索引 (IndexIDMap2 + IndexFlatL2)，用于快速最近邻检索

子类需要实现：
  - _compute_embedding(data) -> embedding
  - _do_save(data, emb, reward, bin_id, sample_id, time_stamp) -> (data_fname, meta_fname, extra_files)
  - do_update_bins() -> 具体的重新分箱或聚类逻辑
"""

import os
import math
import random
import re
import queue
import threading
import time
import json

import numpy as np
import faiss  # 需安装 faiss-gpu 或 faiss-cpu

from dataCollection.DataSaver.utils import multi_factor_distance  # 导入多因素距离计算函数

class BaseSaver:
    def __init__(
        self,
        task: str,
        output_dir: str,
        desired_total: int = 1000,
        dim: int = 512,
        min_bins: int = 3,
        max_bins: int = 5,
        auto_expand_max_bins: bool = True,
        hard_bin_limit: int = 10,
        compute_embedding: bool = True,
        faiss_index_type: str = "flat",
        replace_strategy: str = "nearest",  # "nearest" 或 "random"
        alpha: float = 1.0,
        beta:  float = 0.0,
        gamma: float = 0.0,
        auto_similarity: bool = True,
        sim_scale_factor: float = 0.5,
        init_similarity_thresh: float = 0.05,
        rebin_frequency: int = 50,
        use_lock: bool = True,
        rebin_async: bool = True,
        faiss_kNN: int = 50
    ):
        """
        初始化 BaseSaver 实例。
        
        参数：
          - task: 任务名称，用于在 output_dir 下生成子目录
          - output_dir: 输出根目录（所有文件存放在 output_dir/task 下）
          - desired_total: 期望保存的样本总数（EXACT fill 阶段直接插入直到达到此数目）
          - dim: embedding 的维度
          - min_bins, max_bins: 分箱数量的最小值和最大值（用于聚类或分箱）
          - auto_expand_max_bins: 是否允许自动扩展 bin 数量
          - hard_bin_limit: bin 数量的硬性上限
          - compute_embedding: 是否在插入时计算 embedding
          - faiss_index_type: FAISS 索引类型，目前支持 "flat"
          - replace_strategy: 替换策略，"nearest" 或 "random"
          - alpha, beta, gamma: 多因素距离中各部分的权重
          - auto_similarity: 是否自动更新相似度阈值
          - sim_scale_factor: 相似度阈值因子（阈值 = median_dist * sim_scale_factor）
          - init_similarity_thresh: 初始相似度阈值
          - rebin_frequency: 每插入多少个样本后触发 rebin 操作
          - use_lock: 是否启用多线程锁，保证并发安全
          - rebin_async: 是否启用异步 rebin 操作
          - faiss_kNN: FAISS 检索最近邻时的 k 值
        """
        self.task = task
        self.output_dir = os.path.join(output_dir, task)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 数据文件目录
        self.data_dir = os.path.join(self.output_dir, "")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 元数据目录
        self.meta_dir = os.path.join(self.output_dir, "meta")
        os.makedirs(self.meta_dir, exist_ok=True)
        
        # embedding 相关
        self.dim = dim
        self.compute_embedding = compute_embedding
        
        # FAISS 相关设置
        self.faiss_index_type = faiss_index_type
        self.faiss_kNN = faiss_kNN
        
        # 替换策略
        self.replace_strategy = replace_strategy
        
        # 多因素距离权重
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        
        # 相似度阈值设置
        self.auto_similarity = auto_similarity
        self.sim_scale_factor = sim_scale_factor
        self.init_similarity_thresh = init_similarity_thresh
        self.similarity_thresh = init_similarity_thresh
        self.all_nn_dists = []  # 存储每次插入时计算的最近邻距离
        
        # 期望样本数
        self.desired_total = desired_total
        
        # bin 数量设置
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.auto_expand_max_bins = auto_expand_max_bins
        self.hard_bin_limit = hard_bin_limit
        
        # 当前 bin 数量及相关结构
        self.bin_count = min_bins
        self.bin_edges = None  
        self.bin_capacity = []        # 每个 bin 的容量列表
        self.bin_faiss_index = []     # 每个 bin 对应的 FAISS 索引
        self.bin_records = []         # 每个 bin 内保存的样本 ID 列表
        
        # rebin 统计
        self.rebin_frequency = rebin_frequency
        self.new_sample_count = 0
        
        # 存储所有样本元信息（sample_meta），键为 sample_id
        self.sample_meta = {}
        
        # 全局自增样本 ID
        self.current_global_id = 0
        # 全局插入时序（或时间戳计数）
        self.global_insertion_timestamp = 0
        
        # 并发锁
        self.use_lock = use_lock
        self._lock = threading.Lock() if use_lock else None
        
        # 异步 rebin 设置
        self.rebin_async = rebin_async
        self._rebin_queue = None
        self._rebin_thread = None
        self._stop_rebin_thread = False
        if rebin_async:
            self._rebin_queue = queue.Queue()
            self._rebin_thread = threading.Thread(target=self._rebin_worker_loop, daemon=True)
            self._rebin_thread.start()
        
        # 初始化 bin 设置
        self.set_bin_count(self.bin_count)
        self.create_bin_structs()
    
    # ---------------------- 并发锁方法 ----------------------
    def _lock_acquire(self):
        if self.use_lock:
            self._lock.acquire()
    
    def _lock_release(self):
        if self.use_lock:
            self._lock.release()
    
    # ---------------------- 异步 rebin 方法 ----------------------
    def close(self):
        """
        关闭时停止异步 rebin 后台线程
        """
        if self.rebin_async and self._rebin_thread is not None:
            self._stop_rebin_thread = True
            self._rebin_queue.put(None)
            self._rebin_thread.join()
            self._rebin_thread = None
    
    def _rebin_worker_loop(self):
        """
        后台 rebin 线程循环，从队列中取出任务并调用 do_update_bins()
        """
        while not self._stop_rebin_thread:
            task = self._rebin_queue.get()
            if task is None:
                break
            self._lock_acquire()
            try:
                self.do_update_bins()
            finally:
                self._lock_release()
    
    def update_bins(self):
        """
        触发 rebin 操作：
          - 若 rebin_async 为 True，则将任务放入队列
          - 否则直接调用 do_update_bins()
        """
        if self.rebin_async:
            self._rebin_queue.put(True)
        else:
            self._lock_acquire()
            try:
                self.do_update_bins()
            finally:
                self._lock_release()
    
    # ---------------------- 子类需实现的分箱/聚类核心 ----------------------
    def do_update_bins(self):
        """
        由子类实现具体的重新分箱或聚类逻辑。
        """
        pass
    
    # ---------------------- bin 初始化方法 ----------------------
    def set_bin_count(self, k: int):
        """
        设置 bin 数量为 k，并根据 desired_total 均分各 bin 容量
        """
        self.bin_count = k
        if k <= 0:
            raise ValueError("bin_count 必须 >= 1")
        
        base_cap = self.desired_total // k
        caps = [base_cap] * k
        remainder = self.desired_total - sum(caps)
        idx = 0
        while remainder > 0:
            caps[idx] += 1
            remainder -= 1
            idx = (idx + 1) % k
        self.bin_capacity = caps
    
    def create_bin_structs(self):
        """
        为每个 bin 创建 FAISS 索引和记录列表
        """
        self.bin_faiss_index = []
        self.bin_records = []
        for _ in range(self.bin_count):
            idx = self.create_faiss_index(self.faiss_index_type, self.dim)
            self.bin_faiss_index.append(idx)
            self.bin_records.append([])
    
    # ---------------------- reassign 相关方法 ----------------------
    def reassign_all_bins(self, new_assignments: dict):
        """
        根据 new_assignments（sample_id -> new_bin_id）重新分配所有样本到新的 bin，
        同时更新文件名和重建 FAISS 索引。
        """
        # 1) 清空各 bin 的 FAISS 索引和记录
        for i in range(self.bin_count):
            self.bin_faiss_index[i].reset()
            self.bin_records[i].clear()
        
        # 2) 更新每个样本的 bin 分配，并改名（若需要）
        for sid, meta in self.sample_meta.items():
            old_bin = meta['bin']
            new_bin = new_assignments.get(sid, 0)
            meta['bin'] = new_bin
            self.bin_records[new_bin].append(sid)
            old_fname = meta['filename']
            new_fname = self._rename_file_bin_if_needed(old_fname, old_bin, new_bin, meta, is_second=False)
            meta['filename'] = new_fname
            self._move_extra_files_if_needed(meta, old_bin, new_bin)
        
        # 3) 重新构建各 bin 的 FAISS 索引
        for sid, meta in self.sample_meta.items():
            new_bin = meta['bin']
            emb = meta['embedding']
            if emb is not None:
                # 修改处：确保如果 emb 为一维则转换为二维
                emb_arr = emb.reshape(1, -1) if emb.ndim == 1 else emb
                self.bin_faiss_index[new_bin].add_with_ids(
                    emb_arr.astype('float32'),
                    np.array([sid], dtype=np.int64)
                )
        # 4) 对于超出容量的 bin，随机删除多余样本
        self.post_reassign_trim()
    
    def _rename_file_bin_if_needed(self, old_fname: str, old_bin: int, new_bin: int, meta: dict, is_second: bool):
        """
        如果 bin_id 发生变化，则更新文件名中的 bin 标识，并移动文件到新位置
        """
        if (old_bin == new_bin) or (not old_fname):
            return old_fname
        
        pattern = re.compile(r"_bin_\d+")
        new_fname = pattern.sub(f"_bin_{new_bin}", old_fname)
        
        if is_second:
            old_path = self._get_second_data_path(old_fname, meta)
            new_path = self._get_second_data_path(new_fname, meta)
        else:
            old_path = os.path.join(self.data_dir, old_fname)
            new_path = os.path.join(self.data_dir, new_fname)
        
        if os.path.exists(old_path) and (old_path != new_path):
            os.rename(old_path, new_path)
        
        return new_fname
    
    def _get_second_data_path(self, fname: str, meta: dict):
        """
        子类若使用 second_data_dir，可覆盖此方法返回对应文件路径
        """
        return None
    
    def post_reassign_trim(self):
        """
        对于每个 bin，如果样本数量超过容量，则随机删除多余样本
        """
        for i in range(self.bin_count):
            cap = self.bin_capacity[i]
            recs = self.bin_records[i]
            while len(recs) > cap:
                idx = random.randint(0, len(recs) - 1)
                sid = recs[idx]
                self.remove_sample(i, sid)
    
    # ---------------------- 统计相关 ----------------------
    def get_total_count(self) -> int:
        """
        返回当前已保存样本的总数
        """
        return sum(len(r) for r in self.bin_records)
    
    # ---------------------- 插入数据 ----------------------
    def save_data(self, data, reward: float):
        """
        插入单个样本：
          1. 计算 embedding（若需要）
          2. 判断是否处于 EXACT fill 阶段（当总数 < desired_total 时直接插入）
          3. 达到 desired_total 后进行相似度判断，若过于相似则跳过，否则执行插入或替换操作
        """
        self._lock_acquire()
        try:
            self.new_sample_count += 1
            if (self.new_sample_count % self.rebin_frequency) == 1:
                self.update_bins()
            
            emb = None
            if self.compute_embedding:
                emb = self._compute_embedding(data)
                if emb is not None and emb.ndim == 2 and emb.shape[0] == 1:
                    emb = emb[0]
            
            tstamp = float(self.global_insertion_timestamp)
            self.global_insertion_timestamp += 1
            
            total_now = self.get_total_count()
            if total_now < self.desired_total:
                bin_id = self._assign_bin_for_new_sample(emb, reward, tstamp)
                sid = self.do_insert(bin_id, data, reward, emb, tstamp)
                dval = self._compute_nn_dist_in_bin(bin_id, emb, reward, tstamp)
                if dval is not None:
                    self.all_nn_dists.append(dval)
                    self.update_similarity_thresh()
            else:
                bin_id = self._assign_bin_for_new_sample(emb, reward, tstamp)
                dval = self._compute_nn_dist_in_bin(bin_id, emb, reward, tstamp)
                if (dval is not None) and (dval < self.similarity_thresh):
                    return
                if len(self.bin_records[bin_id]) >= self.bin_capacity[bin_id]:
                    sid_replace = self._replace_in_bin(bin_id, data, reward, emb, tstamp)
                    if sid_replace is not None and dval is not None:
                        self.all_nn_dists.append(dval)
                        self.update_similarity_thresh()
                else:
                    sid = self.do_insert(bin_id, data, reward, emb, tstamp)
                    if dval is not None:
                        self.all_nn_dists.append(dval)
                        self.update_similarity_thresh()
        finally:
            self._lock_release()
    
    def batch_save_data(self, data_list, reward_list):
        """
        批量插入样本：依次对每个样本执行 save_data 逻辑
        """
        if len(data_list) != len(reward_list):
            raise ValueError("batch_save_data 中 data_list 与 reward_list 长度不匹配")
        
        embeddings = []
        if self.compute_embedding:
            for d in data_list:
                e = self._compute_embedding(d)
                if e is not None and e.ndim == 2 and e.shape[0] == 1:
                    e = e[0]
                embeddings.append(e)
        else:
            embeddings = [None] * len(data_list)
        
        for i in range(len(data_list)):
            d = data_list[i]
            r = reward_list[i]
            e = embeddings[i]
            
            self._lock_acquire()
            try:
                self.new_sample_count += 1
                if (self.new_sample_count % self.rebin_frequency) == 1:
                    self.update_bins()
                tstamp = float(self.global_insertion_timestamp)
                self.global_insertion_timestamp += 1
                total_now = self.get_total_count()
                if total_now < self.desired_total:
                    bin_id = self._assign_bin_for_new_sample(e, r, tstamp)
                    sid = self.do_insert(bin_id, d, r, e, tstamp)
                    dval = self._compute_nn_dist_in_bin(bin_id, e, r, tstamp)
                    if dval is not None:
                        self.all_nn_dists.append(dval)
                        self.update_similarity_thresh()
                else:
                    bin_id = self._assign_bin_for_new_sample(e, r, tstamp)
                    dval = self._compute_nn_dist_in_bin(bin_id, e, r, tstamp)
                    if (dval is not None) and (dval < self.similarity_thresh):
                        self._lock_release()
                        continue
                    if len(self.bin_records[bin_id]) >= self.bin_capacity[bin_id]:
                        sid_replace = self._replace_in_bin(bin_id, d, r, e, tstamp)
                        if sid_replace is not None and dval is not None:
                            self.all_nn_dists.append(dval)
                            self.update_similarity_thresh()
                    else:
                        sid = self.do_insert(bin_id, d, r, e, tstamp)
                        if dval is not None:
                            self.all_nn_dists.append(dval)
                            self.update_similarity_thresh()
            finally:
                self._lock_release()
    
    # ---------------------- 新样本分配 bin ----------------------
    def _assign_bin_for_new_sample(self, emb: np.ndarray, reward: float, time_stamp: float):
        """
        子类可覆盖：根据当前聚类中心或其他策略判断新样本应分配到哪个 bin。
        默认实现返回 0。
        """
        return 0
    
    # ---------------------- 计算 bin 内最近邻距离 ----------------------
    def _compute_nn_dist_in_bin(self, bin_id: int, emb_new: np.ndarray, reward_new: float, time_new: float):
        """
        在指定 bin 中计算新样本与已有样本的最小多因素距离，若 bin 为空返回 None。
        """
        recs = self.bin_records[bin_id]
        if not recs:
            return None
        if emb_new is None:
            return self._compute_nn_dist_in_bin_cpu(bin_id, emb_new, reward_new, time_new, recs)
        if (self.faiss_kNN > 0) and (self.bin_faiss_index[bin_id].ntotal > 0):
            query = emb_new[None, :] if emb_new.ndim == 1 else emb_new
            k = min(len(recs), self.faiss_kNN)
            dist_vals, ids_vals = self.bin_faiss_index[bin_id].search(query.astype('float32'), k)
            mind = float('inf')
            for cid in ids_vals[0]:
                if cid < 0:
                    continue
                sid = int(cid)
                old_meta = self.sample_meta[sid]
                old_emb = old_meta['embedding'] if old_meta['embedding'] is not None else np.zeros(self.dim, dtype=np.float32)
                d = multi_factor_distance(
                    query[0], old_emb,
                    reward_new, old_meta['reward'],
                    time_new, old_meta['time'],
                    alpha=self.alpha, beta=self.beta, gamma=self.gamma
                )
                if d < mind:
                    mind = d
            return None if mind == float('inf') else mind
        else:
            return self._compute_nn_dist_in_bin_cpu(bin_id, emb_new, reward_new, time_new, recs)
    
    def _compute_nn_dist_in_bin_cpu(self, bin_id, emb_new: np.ndarray, reward_new: float, time_new: float, recs):
        """
        通过 CPU 遍历方式计算新样本与 bin 内所有样本的最小多因素距离
        """
        mind = float('inf')
        emb_new_use = emb_new if emb_new is not None else np.zeros(self.dim, dtype=np.float32)
        for sid in recs:
            old_meta = self.sample_meta[sid]
            old_emb = old_meta['embedding'] if old_meta['embedding'] is not None else np.zeros(self.dim, dtype=np.float32)
            d = multi_factor_distance(
                emb_new_use, old_emb,
                reward_new, old_meta['reward'],
                time_new, old_meta['time'],
                alpha=self.alpha, beta=self.beta, gamma=self.gamma
            )
            if d < mind:
                mind = d
        return None if mind == float('inf') else mind
    
    # ---------------------- 替换策略 ----------------------
    def _replace_in_bin(self, bin_id: int, data, reward: float, emb: np.ndarray, tstamp: float):
        """
        当指定 bin 已满时，根据替换策略选择替换已有样本：
          - "nearest": 替换与新样本多因素距离最小的样本
          - "random": 随机替换一个样本
        """
        if self.replace_strategy == "nearest":
            if emb is None:
                return self._replace_random(bin_id, data, reward, emb, tstamp)
            else:
                return self._replace_nearest(bin_id, data, reward, emb, tstamp)
        else:
            return self._replace_random(bin_id, data, reward, emb, tstamp)
    
    def _replace_random(self, bin_id: int, data, reward: float, emb: np.ndarray, tstamp: float):
        """
        随机替换：随机选取一个已有样本进行删除，然后插入新样本
        """
        recs = self.bin_records[bin_id]
        if not recs:
            return self.do_insert(bin_id, data, reward, emb, tstamp)
        idx = random.randint(0, len(recs) - 1)
        sid_old = recs[idx]
        self.remove_sample(bin_id, sid_old)
        new_id = self.do_insert(bin_id, data, reward, emb, tstamp)
        return new_id
    
    def _replace_nearest(self, bin_id: int, data, reward: float, emb: np.ndarray, tstamp: float):
        """
        最近邻替换：遍历 bin 内样本，计算多因素距离，替换与新样本距离最小的那个
        """
        recs = self.bin_records[bin_id]
        if not recs:
            return self.do_insert(bin_id, data, reward, emb, tstamp)
        best_id = None
        best_dist = float('inf')
        for sid in recs:
            old_meta = self.sample_meta[sid]
            old_emb = old_meta['embedding'] if old_meta['embedding'] is not None else np.zeros(self.dim, dtype=np.float32)
            d = multi_factor_distance(
                emb, old_emb,
                reward, old_meta['reward'],
                tstamp, old_meta['time'],
                alpha=self.alpha, beta=self.beta, gamma=self.gamma
            )
            if d < best_dist:
                best_dist = d
                best_id = sid
        self.remove_sample(bin_id, best_id)
        new_id = self.do_insert(bin_id, data, reward, emb, tstamp)
        return new_id
    
    # ---------------------- 插入与删除操作 ----------------------
    def do_insert(self, bin_id: int, data, reward: float, emb: np.ndarray, time_stamp: float):
        """
        执行插入操作：
          - 写入数据文件、meta 文件
          - 更新 bin_records 和 FAISS 索引
          - 返回生成的全局样本 ID
        """
        sid = self.current_global_id
        self.current_global_id += 1
        
        data_fname, meta_fname, extra_files = self._do_save(data, emb, reward, bin_id, sid, time_stamp)
        meta_dict = {
            'reward': reward,
            'time': time_stamp,
            'bin': bin_id,
            'filename': data_fname,
            'extra_filenames': extra_files,
            'embedding': emb,
            'meta_filename': meta_fname
        }
        self.sample_meta[sid] = meta_dict
        
        if (bin_id >= 0) and (bin_id < self.bin_count):
            self.bin_records[bin_id].append(sid)
            if emb is not None:
                # 修改处：如果 emb 为一维，则转换为二维，确保形状为 (1, dim)
                if emb.ndim == 1:
                    emb_arr = emb.reshape(1, -1)
                else:
                    emb_arr = emb
                self.bin_faiss_index[bin_id].add_with_ids(
                    emb_arr.astype('float32'),
                    np.array([sid], dtype=np.int64)
                )
        
        return sid
    
    def remove_sample(self, bin_id: int, sid: int):
        """
        删除指定 bin 中的样本 sid：
          - 从 bin_records 移除
          - 删除数据文件（主文件及额外文件）
          - 删除 meta 文件
          - 从 FAISS 索引中删除
          - 从 sample_meta 移除
        """
        if sid not in self.sample_meta:
            return
        meta = self.sample_meta[sid]
        data_fname = meta.get('filename', None)
        meta_fname = meta.get('meta_filename', None)
        
        if (bin_id >= 0) and (bin_id < len(self.bin_records)):
            recs = self.bin_records[bin_id]
            if sid in recs:
                recs.remove(sid)
        
        if data_fname:
            data_path = os.path.join(self.data_dir, data_fname)
            if os.path.exists(data_path):
                os.remove(data_path)
        
        self._remove_extra_files_if_needed(meta, bin_id)
        
        if meta_fname:
            meta_path = os.path.join(self.meta_dir, meta_fname)
            if os.path.exists(meta_path):
                os.remove(meta_path)
        
        if (bin_id >= 0) and (bin_id < len(self.bin_faiss_index)):
            arr = np.array([sid], dtype=np.int64)
            try:
                self.bin_faiss_index[bin_id].remove_ids(arr)
            except Exception:
                pass
        
        del self.sample_meta[sid]
    
    # ---------------------- FAISS 工厂方法 ----------------------
    def create_faiss_index(self, index_type: str, dim: int):
        """
        根据 index_type 创建 FAISS 索引，目前支持 "flat" 类型
        """
        if index_type == "flat":
            base_idx = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"不支持的 faiss 索引类型: {index_type}")
        return faiss.IndexIDMap2(base_idx)
    
    # ---------------------- 相似度阈值更新 ----------------------
    def update_similarity_thresh(self):
        """
        基于 all_nn_dists 的中位数和 sim_scale_factor 更新相似度阈值
        """
        if not self.auto_similarity:
            return
        n = len(self.all_nn_dists)
        if n < 5:
            self.similarity_thresh = self.init_similarity_thresh
            return
        med = np.median(self.all_nn_dists)
        if med < 1e-9:
            self.similarity_thresh = self.init_similarity_thresh
            return
        self.similarity_thresh = self.sim_scale_factor * med
    
    # ---------------------- 抽象方法 ----------------------
    def _compute_embedding(self, data):
        """
        由子类实现：将 data 转换为 embedding（numpy 数组）
        """
        raise NotImplementedError
    
    def _do_save(self, data, emb, reward, bin_id: int, sample_id: int, time_stamp: float):
        """
        由子类实现：保存数据文件及 meta 文件，返回 (data_filename, meta_filename, extra_files)
        """
        raise NotImplementedError
    
    def _move_extra_files_if_needed(self, meta: dict, old_bin: int, new_bin: int):
        """
        由子类实现：当 bin 发生变化时，移动额外文件（例如 second_data_dir 中的文件）
        """
        pass
    
    def _remove_extra_files_if_needed(self, meta: dict, bin_id: int):
        """
        由子类实现：删除额外文件（例如 second_data_dir 中的文件）
        """
        pass
