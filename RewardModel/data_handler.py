#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_handler.py
===============
本模块用于统一管理轨迹数据的存储与采样。
每个轨迹以一个字典存储，包含以下键：
    - "sa": 状态-动作数据列表
    - "reward": 奖励数据列表
    - "img": 图像数据列表（如有）
    - "pc": 点云数据列表（如有）
    - "finished": 轨迹是否结束的标志
通过这种设计可以保证各数据项在同一轨迹内严格对应，从而避免数据索引错误。
"""

import numpy as np

class DataHandler:
    def __init__(self, ds, data_type, capacity, image_height=None, image_width=None, point_cloud_num_points=None, max_size=100):
        """
        初始化 DataHandler。

        参数：
            ds (int): 状态-动作数据维度。
            image_height (int): 图像高度（如适用）。
            image_width (int): 图像宽度（如适用）。
            point_cloud_num_points (int): 点云中点的数量（如适用）。
            max_size (int): 最大存储轨迹数。
        """
        self.ds = ds
        self.image_height = image_height
        self.image_width = image_width
        self.point_cloud_num_points = point_cloud_num_points
        self.max_size = max_size
        # 使用列表存储所有轨迹，每个轨迹为一个字典
        self.trajectories = []
        self.buffer_index = 0
        self.data_type = data_type
        self.capacity = capacity
        # 初始化用于缓冲训练查询对的缓冲区（结构与原设计保持一致）
        if self.data_type == "image":
            self.buffer_seg1 = np.empty((self.capacity, 1, self.image_height, self.image_width, 3), dtype=np.uint8)
            self.buffer_seg2 = np.empty((self.capacity, 1, self.image_height, self.image_width, 3), dtype=np.uint8)
        elif self.data_type == "pointcloud" or self.vlm == "pointllm_two_image":
            self.buffer_seg1 = np.empty((self.capacity, 1, self.point_cloud_num_points, 6), dtype=np.float32)
            self.buffer_seg2 = np.empty((self.capacity, 1, self.point_cloud_num_points, 6), dtype=np.float32)
        else:
            self.buffer_seg1 = np.empty((self.capacity, self.size_segment, self.ds+self.da), dtype=np.float32)
            self.buffer_seg2 = np.empty((self.capacity, self.size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_full = False
    
    def add_data(self, obs, act, rew, done, img=None, point_cloud=None):
        """
        将新数据加入轨迹缓存，统一存储在 self.trajectories 中。

        参数：
            obs (np.array): 当前状态。
            act (np.array): 当前动作。
            rew (float 或 np.array): 当前奖励。
            done (bool): 是否为终止步。
            img (np.array): 图像数据（如有）。
            point_cloud (np.array): 点云数据（如有）。
        """
        # 计算状态-动作向量，并调整为二维数组
        sa = np.concatenate([obs, act], axis=-1)
        sa = sa.reshape(1, -1)
        # 奖励数据
        r = np.array(rew).reshape(1, 1)
        # 图像数据（若有）
        img_data = np.copy(img.reshape(1, *img.shape)) if img is not None else None
        # 点云数据（若有）
        if point_cloud is not None:
            point_cloud = point_cloud.astype(np.float32)
            pc_data = np.copy(point_cloud.reshape(1, self.point_cloud_num_points, 6))
        else:
            pc_data = None
        
        # 如果没有轨迹或最后一条轨迹已结束，则新建一个轨迹字典
        if not self.trajectories or self.trajectories[-1].get("finished", False):
            new_traj = {"sa": [], "reward": [], "img": [], "pc": [], "finished": False}
            self.trajectories.append(new_traj)
        curr_traj = self.trajectories[-1]
        curr_traj["sa"].append(sa)
        curr_traj["reward"].append(r)
        if img_data is not None:
            curr_traj["img"].append(img_data)
        if pc_data is not None:
            curr_traj["pc"].append(pc_data)
        if done:
            curr_traj["finished"] = True
            if len(self.trajectories) > self.max_size:
                self.trajectories.pop(0)
            # 预创建新轨迹以便后续数据添加
            self.trajectories.append({"sa": [], "reward": [], "img": [], "pc": [], "finished": False})
    
    def get_queries(self, mb_size=20):
        """
        从存储的轨迹数据中随机采样一批查询对，用于奖励标签生成。
        返回的八元组分别为：
            - sa_t_1, sa_t_2：状态-动作数据对，形状 (mb_size, ds)
            - r_t_1, r_t_2：奖励数据对，形状 (mb_size, 1)
            - pc_t_1, pc_t_2：点云数据对（如有，否则为 None）
            - img_t_1, img_t_2：图像数据对（如有，否则为 None）

        采样策略：
            1. 过滤出至少包含状态-动作数据的有效轨迹。
            2. 取所有轨迹中最短长度，防止索引越界。
            3. 若只有一条轨迹则从中随机采样；若多条轨迹则随机选择轨迹和时间步。

        返回：
            Tuple: (sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2)
        """
        valid_trajs = [traj for traj in self.trajectories if traj["sa"]]
        if not valid_trajs:
            print("Warning: 没有可用的轨迹数据")
            return (np.empty((0, self.ds)),
                    np.empty((0, self.ds)),
                    np.empty((0, 1)),
                    np.empty((0, 1)),
                    None, None,
                    np.empty((0, self.image_height, self.image_width, 3)),
                    np.empty((0, self.image_height, self.image_width, 3)))
        traj_sa = [np.concatenate(traj["sa"], axis=0) for traj in valid_trajs]
        traj_reward = [np.concatenate(traj["reward"], axis=0) for traj in valid_trajs]
        traj_img = [np.concatenate(traj["img"], axis=0) if traj["img"] else None for traj in valid_trajs]
        traj_pc = [np.concatenate(traj["pc"], axis=0) if traj["pc"] else None for traj in valid_trajs]
        pool_size = len(traj_sa)
        min_length = min(traj.shape[0] for traj in traj_sa)
        if pool_size == 1:
            time_idx1 = np.random.randint(0, min_length, size=mb_size)
            time_idx2 = np.random.randint(0, min_length, size=mb_size)
            sa_t_1 = traj_sa[0][time_idx1, :]
            sa_t_2 = traj_sa[0][time_idx2, :]
            r_t_1 = traj_reward[0][time_idx1, :]
            r_t_2 = traj_reward[0][time_idx2, :]
            img_t_1 = traj_img[0][time_idx1, ...] if traj_img[0] is not None else None
            img_t_2 = traj_img[0][time_idx2, ...] if traj_img[0] is not None else None
            pc_t_1 = traj_pc[0][time_idx1, ...] if traj_pc[0] is not None else None
            pc_t_2 = traj_pc[0][time_idx2, ...] if traj_pc[0] is not None else None
        else:
            batch_idx1 = np.random.choice(pool_size, size=mb_size, replace=True)
            batch_idx2 = np.random.choice(pool_size, size=mb_size, replace=True)
            time_idx1 = np.random.randint(0, min_length, size=mb_size)
            time_idx2 = np.random.randint(0, min_length, size=mb_size)
            sa_t_1 = np.array([traj_sa[i][t] for i, t in zip(batch_idx1, time_idx1)])
            sa_t_2 = np.array([traj_sa[i][t] for i, t in zip(batch_idx2, time_idx2)])
            r_t_1 = np.array([traj_reward[i][t] for i, t in zip(batch_idx1, time_idx1)])
            r_t_2 = np.array([traj_reward[i][t] for i, t in zip(batch_idx2, time_idx2)])
            if any(img is not None for img in traj_img):
                img_t_1 = np.array([traj_img[i][t] if traj_img[i] is not None else np.zeros((self.image_height, self.image_width, 3))
                                     for i, t in zip(batch_idx1, time_idx1)])
                img_t_2 = np.array([traj_img[i][t] if traj_img[i] is not None else np.zeros((self.image_height, self.image_width, 3))
                                     for i, t in zip(batch_idx2, time_idx2)])
            else:
                img_t_1 = img_t_2 = None
            if any(pc is not None for pc in traj_pc):
                pc_t_1 = np.array([traj_pc[i][t] if traj_pc[i] is not None else np.zeros((self.point_cloud_num_points, 6))
                                    for i, t in zip(batch_idx1, time_idx1)])
                pc_t_2 = np.array([traj_pc[i][t] if traj_pc[i] is not None else np.zeros((self.point_cloud_num_points, 6))
                                    for i, t in zip(batch_idx2, time_idx2)])
            else:
                pc_t_1 = pc_t_2 = None
        return sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2

    def put_queries(self, query1, query2, labels):
        """
        将查询对及对应标签写入预分配缓冲区。

        参数说明与原代码一致。
        """
        total_sample = query1.shape[0]
        next_index = self.buffer_index + total_sample
        if self.data_type == "image":
            query1 = query1.reshape(query1.shape[0], 1, query1.shape[1], query1.shape[2], query1.shape[3])
            query2 = query2.reshape(query2.shape[0], 1, query2.shape[1], query2.shape[2], query2.shape[3])
        elif self.data_type == "pointcloud":
            if query1.ndim == 3:
                query1 = query1[:, None, :, :]
                query2 = query2[:, None, :, :]
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], query1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], query2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            remain = total_sample - maximum_index
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], query1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], query2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], query1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], query2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
