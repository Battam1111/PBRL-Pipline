#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_model.py
===============
本模块是奖励模型的综合接口，整合了模型构建、数据管理、采样、标签生成以及训练流程。
该模块调用其它模块（config、utils、models、sampling、data_handler、training）实现了
高效、鲁棒且模块化的设计，同时所有代码均附有详细中文注释说明功能。

【本版本说明】
1. 数据管理部分采用统一的轨迹字典存储方式，保证状态-动作、奖励、图像、点云数据在同一轨迹中严格对应，避免数据不一致问题。
2. 各功能模块（模型生成、数据采集、采样、训练、标签生成）分别拆分到独立文件中，降低单文件代码量。
3. 使用混合精度训练（AMP）和较小批量训练（针对点云数据）以降低显存占用并提高效率。
"""

import os
import time
import datetime
import pickle as pkl
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RewardModel.config import device
from RewardModel.utils import robust_fix_image_array, fix_image_array
from RewardModel.models import gen_net, gen_image_net, gen_image_net2
from RewardModel.sampling import KCenterGreedy, compute_smallest_dist
from RewardModel.data_handler import DataHandler
from RewardModel.training import RewardModelTrainer

# 引入用于 VLM 查询的 prompt 模块（需要自行配置）
from prompt import (
    gemini_free_query_env_prompts, gemini_summary_env_prompts,
    gemini_free_query_prompt1, gemini_free_query_prompt2,
    gemini_single_query_env_prompts,
    gpt_free_query_env_prompts, gpt_summary_env_prompts,
    pointllm_free_query_env_prompts, pointllm_summary_env_prompts,
    pointllm_free_query_prompt1, pointllm_free_query_prompt2,
)
from vlms.gemini_infer import gemini_query_2, gemini_query_1
from vlms.pointllm_infer import pointllm_query_1, pointllm_query_2
from extract import extract_label_from_response
from pointCloudNet import gen_point_cloud_net

class RewardModel:
    def __init__(self, ds, da,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1,
                 max_size=100, activation='tanh', capacity=5e5,
                 large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0,
                 # VLM参数
                 vlm_label=True,
                 env_name="CartPole-v1",
                 vlm="gemini_free_form",
                 clip_prompt=None,
                 log_dir=None,
                 flip_vlm_label=False,
                 save_query_interval=25,
                 cached_label_path=None,
                 # 图像奖励参数
                 reward_model_layers=3,
                 reward_model_H=256,
                 image_reward=True,
                 image_height=128,
                 image_width=128,
                 resize_factor=1,
                 resnet=False,
                 conv_kernel_sizes=[5,3,3,3],
                 conv_n_channels=[16,32,64,128],
                 conv_strides=[3,2,2,2],
                 # 数据类型
                 data_type="pointcloud",
                 point_cloud_num_points=8192,
                 **kwargs):
        """
        初始化 RewardModel 实例，构建模型集合、数据管理器和训练器。

        参数说明请参见各模块详细注释。
        """
        self.ds = ds
        self.da = da
        self.ensemble_size = ensemble_size
        self.lr = lr
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.size_segment = size_segment
        self.max_size = max_size
        self.activation = activation
        self.capacity = int(capacity)
        self.large_batch = large_batch
        self.label_margin = label_margin
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_eps_equal = teacher_eps_equal

        self.vlm_label = vlm_label
        self.env_name = env_name
        self.vlm = vlm
        self.clip_prompt = clip_prompt
        self.log_dir = log_dir
        self.flip_vlm_label = flip_vlm_label
        self.save_query_interval = save_query_interval
        self.cached_label_path = cached_label_path

        self.reward_model_layers = reward_model_layers
        self.reward_model_H = reward_model_H
        self.image_reward = image_reward
        self.image_height = image_height
        self.image_width = image_width
        self.resize_factor = resize_factor
        self.resnet = resnet
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_n_channels = conv_n_channels
        self.conv_strides = conv_strides

        self.data_type = data_type
        self.point_cloud_num_points = point_cloud_num_points

        # 构建模型集合
        self.ensemble = []
        self.paramlst = []
        for i in range(self.ensemble_size):
            if not self.image_reward:
                model = nn.Sequential(*gen_net(in_size=self.ds+self.da, out_size=1,
                                               H=self.reward_model_H,
                                               n_layers=self.reward_model_layers,
                                               activation=self.activation)).float().to(device)
            else:
                if self.vlm_label and self.vlm == 'pointllm_two_image' and self.data_type=="pointcloud":
                    model = gen_point_cloud_net(num_points=self.point_cloud_num_points, input_dim=6,
                                                 device=device, normalize=False)
                else:
                    if not self.resnet:
                        model = gen_image_net(self.image_height, self.image_width,
                                              self.conv_kernel_sizes, self.conv_n_channels,
                                              self.conv_strides).float().to(device)
                    else:
                        model = gen_image_net2().float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        # 初始化数据管理器（统一管理轨迹数据）
        self.data_handler = DataHandler(ds=self.ds,
                                        data_type=self.data_type,
                                        capacity=self.capacity,
                                        image_height=self.image_height,
                                        image_width=self.image_width,
                                        point_cloud_num_points=self.point_cloud_num_points,
                                        max_size=self.max_size)
        # 初始化训练器
        self.trainer = RewardModelTrainer(self.ensemble, self.paramlst, self.lr,
                                          data_type=self.data_type,
                                          train_batch_size=64 if not self.resnet else 32,
                                          teacher_beta=self.teacher_beta,
                                          teacher_gamma=self.teacher_gamma,
                                          teacher_eps_mistake=self.teacher_eps_mistake,
                                          label_margin=self.label_margin,
                                          teacher_eps_equal=self.teacher_eps_equal)

        self.train_times = 0
        self.vlm_label_acc = 0

    def eval(self):
        """
        设置所有内部模型为评估模式。
        """
        for model in self.ensemble:
            model.eval()

    def train(self):
        """
        设置所有内部模型为训练模式。
        """
        for model in self.ensemble:
            model.train()

    def change_batch(self, new_frac):
        """动态调整批量大小"""
        self.mb_size = int(self.origin_mb_size * new_frac)

    def set_batch(self, new_batch):
        """设置批量大小"""
        self.mb_size = int(new_batch)

    def set_teacher_thres_skip(self, new_margin):
        """设置教师跳过阈值"""
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip

    def set_teacher_thres_equal(self, new_margin):
        """设置教师相等阈值"""
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    def add_data(self, obs, act, rew, done, img=None, point_cloud=None):
        """
        添加一条新的数据，调用数据管理器的 add_data 方法。

        参数详见 data_handler.py 中的说明。
        """
        self.data_handler.add_data(obs, act, rew, done, img, point_cloud)

    def sample_queries(self, mb_size=20):
        """
        从数据管理器中采样查询对。

        返回：
            八元组 (sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2)
        """
        return self.data_handler.get_queries(mb_size)

    # =============================================================================
    # 根据真实奖励生成理性标签，并结合 VLM（二次修正）生成最终标签
    # 针对不同数据类型分别处理。
    # =============================================================================
    def get_label(self, 
                sa_t_1, sa_t_2, 
                r_t_1, r_t_2,
                img_t_1=None, img_t_2=None,
                point_cloud_t_1=None, point_cloud_t_2=None):
        """
        生成奖励标签：先根据真实奖励计算理性标签，再通过 VLM（二次修正）生成最终标签，
        针对不同数据类型分别处理。

        参数：
            sa_t_1, sa_t_2: 状态-动作序列。
            r_t_1, r_t_2: 奖励序列。
            img_t_1, img_t_2: 图像数据（可选）。
            point_cloud_t_1, point_cloud_t_2: 点云数据（可选）。

        返回：
            根据配置返回不同数据元组（包含原始数据及生成的标签）。
        """
        # ------------------------------
        # A. 根据真实奖励生成理性标签
        # ------------------------------
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            valid_mask = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if np.sum(valid_mask) == 0:
                return None, None, None, None, []
            sa_t_1 = sa_t_1[valid_mask] if sa_t_1 is not None else None
            sa_t_2 = sa_t_2[valid_mask] if sa_t_2 is not None else None
            r_t_1 = r_t_1[valid_mask]
            r_t_2 = r_t_2[valid_mask]
            sum_r_t_1 = sum_r_t_1[valid_mask]
            sum_r_t_2 = sum_r_t_2[valid_mask]
            if self.data_type == "image" and img_t_1 is not None:
                img_t_1 = img_t_1[valid_mask]
                img_t_2 = img_t_2[valid_mask]
            if self.data_type == "pointcloud" and point_cloud_t_1 is not None:
                point_cloud_t_1 = point_cloud_t_1[valid_mask]
                point_cloud_t_2 = point_cloud_t_2[valid_mask]
        margin_mask = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for idx in range(seg_size - 1):
            temp_r_t_1[:, :idx + 1] *= self.teacher_gamma
            temp_r_t_2[:, :idx + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
        if self.teacher_beta > 0:
            r_hat = np.stack([sum_r_t_1, sum_r_t_2], axis=-1)
            r_hat_tensor = torch.from_numpy(r_hat).float().to(device) * self.teacher_beta
            prob_2 = F.softmax(r_hat_tensor, dim=-1)[:, 1]
            random_draw = torch.bernoulli(prob_2).int().cpu().numpy().reshape(-1, 1)
            labels = random_draw
        else:
            rational_labels = (sum_r_t_1 < sum_r_t_2).astype(int)
            labels = rational_labels.reshape(-1, 1)
        noise_mask = (np.random.rand(labels.shape[0]) <= self.teacher_eps_mistake)
        labels[noise_mask] = 1 - labels[noise_mask]
        labels[margin_mask] = -1
        # ★ 保存原始理性标签为 gt_labels，后续 VLM 修正使用
        gt_labels = labels.copy()
        
        # ------------------------------
        # B. 若不使用 VLM，则直接返回理性标签及相关数据
        # ------------------------------
        if not self.vlm_label:
            if not self.image_reward:
                return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
            if self.data_type == "image":
                return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels
            elif self.data_type == "pointcloud":
                return sa_t_1, sa_t_2, r_t_1, r_t_2, point_cloud_t_1, point_cloud_t_2, labels

        # ------------------------------
        # C. 使用 VLM 进行二次修正
        # ------------------------------
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        if self.vlm == 'pointllm_two_image':
            vlm_labels = []
            useful_indices = []
            for idx, (pc1, pc2) in enumerate(zip(point_cloud_t_1, point_cloud_t_2)):
                dist = np.linalg.norm(pc1 - pc2)
                if dist < 1e-3:
                    useful_indices.append(0)
                    vlm_labels.append(-1)
                else:
                    useful_indices.append(1)
                    prompt1 = pointllm_free_query_prompt1
                    prompt2 = pointllm_free_query_prompt2
                    main_prompt = pointllm_free_query_env_prompts[self.env_name]
                    summary_prompt = pointllm_summary_env_prompts[self.env_name]
                    res = pointllm_query_2([
                        prompt1, pc1.reshape(-1, 6),
                        prompt2, pc2.reshape(-1, 6),
                        main_prompt
                    ], summary_prompt)
                    try:
                        res_int = int(extract_label_from_response(res))
                        if res_int not in [0, 1, -1]:
                            res_int = -1
                    except:
                        res_int = -1
                    vlm_labels.append(res_int)
            vlm_labels = np.array(vlm_labels).reshape(-1, 1)
            useful_mask = (np.array(useful_indices) == 1) & (vlm_labels.reshape(-1) != -1)
            if (sa_t_1 is not None and sa_t_1.shape[0] != useful_mask.shape[0]) or (point_cloud_t_1.shape[0] != useful_mask.shape[0]):
                raise IndexError("数据长度不匹配！")
            
            if self.data_type == "pointcloud":
                point_cloud_t_1 = point_cloud_t_1[useful_mask]
                point_cloud_t_2 = point_cloud_t_2[useful_mask]
                r_t_1 = r_t_1[useful_mask]
                r_t_2 = r_t_2[useful_mask]
                gt_labels = gt_labels[useful_mask]
                vlm_labels = vlm_labels[useful_mask]

                acc = 0
                if len(vlm_labels) > 0:
                    acc = np.sum(vlm_labels == gt_labels) / len(vlm_labels)
                    print(f"vlm label acc: {acc}")
                else:
                    print("no vlm label")
                self.vlm_label_acc = acc
                return sa_t_1, sa_t_2, r_t_1, r_t_2, point_cloud_t_1, point_cloud_t_2, gt_labels, vlm_labels
            else:
                sa_t_1 = sa_t_1[useful_mask]
                sa_t_2 = sa_t_2[useful_mask]
                r_t_1 = r_t_1[useful_mask]
                r_t_2 = r_t_2[useful_mask]
                img_t_1 = img_t_1[useful_mask]
                img_t_2 = img_t_2[useful_mask]
                gt_labels = gt_labels[useful_mask]
                vlm_labels = vlm_labels[useful_mask]
                
                acc = 0
                if len(vlm_labels) > 0:
                    acc = np.sum(vlm_labels == gt_labels) / len(vlm_labels)
                    print(f"vlm label acc: {acc}")
                else:
                    print("no vlm label")
                self.vlm_label_acc = acc
                return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, gt_labels, vlm_labels
        else:
            gpt_two_image_paths = []
            combined_images_list = []
            file_path = os.path.abspath(__file__)
            dir_path = os.path.dirname(file_path)
            save_path = os.path.join(dir_path, "data", "gpt_query_image", self.env_name, time_string)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            useful_indices = []
            for idx, (im1, im2) in enumerate(zip(img_t_1, img_t_2)):
                im1 = robust_fix_image_array(im1)
                im2 = robust_fix_image_array(im2)
                combined_image = np.concatenate([im1, im2], axis=1)
                combined_images_list.append(combined_image)
                try:
                    combined_image_pil = Image.fromarray(combined_image)
                except Exception as e:
                    print("Error converting combined image to PIL, combined_image.shape =", combined_image.shape)
                    raise e
                first_image_save_path = os.path.join(save_path, f"first_{idx:06}.png")
                second_image_save_path = os.path.join(save_path, f"second_{idx:06}.png")
                Image.fromarray(robust_fix_image_array(im1)).save(first_image_save_path)
                Image.fromarray(robust_fix_image_array(im2)).save(second_image_save_path)
                gpt_two_image_paths.append([first_image_save_path, second_image_save_path])
                diff = np.linalg.norm(im1 - im2)
                if diff < 1e-3:
                    useful_indices.append(0)
                else:
                    useful_indices.append(1)
            if self.vlm == 'gpt4v_two_image':
                from vlms.gpt4_infer import gpt4v_infer_2
                vlm_labels = []
                for idx, (img_path_1, img_path_2) in enumerate(gpt_two_image_paths):
                    print(f"querying vlm {idx}/{len(gpt_two_image_paths)}")
                    query_prompt = gpt_free_query_env_prompts[self.env_name]
                    summary_prompt = gpt_summary_env_prompts[self.env_name]
                    res = gpt4v_infer_2(query_prompt, summary_prompt, img_path_1, img_path_2)
                    try:
                        label_res = int(res)
                    except:
                        label_res = -1
                    vlm_labels.append(label_res)
                    time.sleep(0.1)
            elif self.vlm == 'gemini_single_prompt':
                vlm_labels = []
                for idx, (im1, im2) in enumerate(zip(img_t_1, img_t_2)):
                    res = gemini_query_1([
                        gemini_free_query_prompt1,
                        Image.fromarray(robust_fix_image_array(im1)),
                        gemini_free_query_prompt2,
                        Image.fromarray(robust_fix_image_array(im2)),
                        gemini_single_query_env_prompts[self.env_name],
                    ])
                    try:
                        if "-1" in res:
                            res = -1
                        elif "0" in res:
                            res = 0
                        elif "1" in res:
                            res = 1
                        else:
                            res = -1
                    except:
                        res = -1
                    vlm_labels.append(res)
            elif self.vlm == "gemini_free_form":
                vlm_labels = []
                for idx, (im1, im2) in enumerate(zip(img_t_1, img_t_2)):
                    res = gemini_query_2([
                        gemini_free_query_prompt1,
                        Image.fromarray(robust_fix_image_array(im1)),
                        gemini_free_query_prompt2,
                        Image.fromarray(robust_fix_image_array(im2)),
                        gemini_free_query_env_prompts[self.env_name]
                    ],
                        gemini_summary_env_prompts[self.env_name]
                    )
                    try:
                        res = int(res)
                        if res not in [0, 1, -1]:
                            res = -1
                    except:
                        res = -1
                    vlm_labels.append(res)
            else:
                vlm_labels = []
            vlm_labels = np.array(vlm_labels).reshape(-1, 1)
            good_idx = (vlm_labels != -1).flatten()
            useful_indices = (np.array(useful_indices) == 1).flatten()
            good_idx = np.logical_and(good_idx, useful_indices)
            gt_labels = gt_labels[good_idx]
            vlm_labels = vlm_labels[good_idx]
            combined_images_list = np.array(combined_images_list)[good_idx]
            sa_t_1 = sa_t_1[good_idx]
            sa_t_2 = sa_t_2[good_idx]
            r_t_1 = r_t_1[good_idx]
            r_t_2 = r_t_2[good_idx]
            img_t_1 = img_t_1[good_idx]
            img_t_2 = img_t_2[good_idx]
            if self.data_type == "pointcloud":
                point_cloud_t_1 = point_cloud_t_1[good_idx]
                point_cloud_t_2 = point_cloud_t_2[good_idx]
            if self.flip_vlm_label:
                vlm_labels = 1 - vlm_labels
            if self.train_times % self.save_query_interval == 0 or 'gpt4v' in self.vlm:
                save_path = os.path.join(self.log_dir, "vlm_label_set")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(os.path.join(save_path, f"{time_string}.pkl"), "wb") as f:
                    pkl.dump([
                        combined_images_list, gt_labels, vlm_labels,
                        sa_t_1, sa_t_2, r_t_1, r_t_2
                    ], f, protocol=pkl.HIGHEST_PROTOCOL)
            acc = 0
            if len(vlm_labels) > 0:
                acc = np.sum(vlm_labels == gt_labels) / len(vlm_labels)
                print(f"vlm label acc: {acc}")
            else:
                print("no vlm label")
            self.vlm_label_acc = acc
            if self.data_type == "image":
                return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, gt_labels, vlm_labels
            else:
                return sa_t_1, sa_t_2, r_t_1, r_t_2, point_cloud_t_1, point_cloud_t_2, gt_labels, vlm_labels

    def get_label_from_cached_states(self):
        """
        从缓存标签文件中读取标签数据，避免重复调用 VLM 接口。

        返回：
            缓存的标签数据及对应的查询数据。
        """
        if self.read_cache_idx >= len(self.all_cached_labels):
            return None, None, None, None, None, []
        with open(self.all_cached_labels[self.read_cache_idx], 'rb') as f:
            data = pkl.load(f)
        self.read_cache_idx += 1
        combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = data
        return combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2

    # =============================================================================
    # 以下为不同采样策略，均利用优化后的 get_queries 和 get_label 函数实现高效采样
    # =============================================================================
    def kcenter_sampling(self):
        """
        K-Center Sampling 策略：使用状态-动作数据或点云数据，基于距离选择最具代表性的样本。

        返回：
            int: 成功写入缓冲区的标签数量。
        """
        num_init = self.mb_size * self.large_batch
        if self.data_type == "pointcloud":
            _, _, r_t_1, r_t_2, pc_t_1, pc_t_2 = self.data_handler.get_queries(mb_size=num_init)
            temp_pc = np.concatenate([pc_t_1.reshape(pc_t_1.shape[0], -1),
                                      pc_t_2.reshape(pc_t_2.shape[0], -1)], axis=1)
            max_len = self.capacity if self.data_handler.buffer_full else self.data_handler.buffer_index
            tot_pc = self.data_handler.buffer_seg1[:max_len].reshape(max_len, -1)
            selected_index = KCenterGreedy(temp_pc, tot_pc, self.mb_size)
            r_t_1 = r_t_1[selected_index]
            r_t_2 = r_t_2[selected_index]
            pc_t_1 = pc_t_1[selected_index]
            pc_t_2 = pc_t_2[selected_index]
            _, _, r_t_1, r_t_2, pc_t_1, pc_t_2, labels = self.get_label(None, None, r_t_1, r_t_2,
                                                                           point_cloud_t_1=pc_t_1, point_cloud_t_2=pc_t_2)
            if len(labels) > 0:
                self.data_handler.put_queries(pc_t_1, pc_t_2, labels)
            return len(labels)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2, extra1, extra2 = self.data_handler.get_queries(mb_size=num_init)
            temp_sa_t_1 = sa_t_1[:, :self.ds]
            temp_sa_t_2 = sa_t_2[:, :self.ds]
            temp_sa = np.concatenate([temp_sa_t_1, temp_sa_t_2], axis=1)
            max_len = self.capacity if self.data_handler.buffer_full else self.data_handler.buffer_index
            tot_sa_1 = self.data_handler.buffer_seg1[:max_len].reshape(max_len, -1)
            tot_sa_2 = self.data_handler.buffer_seg2[:max_len].reshape(max_len, -1)
            tot_sa = np.concatenate([tot_sa_1, tot_sa_2], axis=1)
            selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
            r_t_1 = r_t_1[selected_index]
            sa_t_1 = sa_t_1[selected_index]
            r_t_2 = r_t_2[selected_index]
            sa_t_2 = sa_t_2[selected_index]
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1=extra1, img_t_2=extra2)
            if len(labels) > 0:
                self.data_handler.put_queries(sa_t_1, sa_t_2, labels)
            return len(labels)
    
    def kcenter_disagree_sampling(self):
        """
        K-Center Disagree Sampling 策略：
        基于模型预测不一致性选择样本进行标签生成。
        针对点云数据，由于状态-动作向量数据通常为 None，因此直接调用 uniform_sampling 策略。

        返回：
            int: 成功写入缓冲区的标签数量。
        """
        if self.data_type == "pointcloud":
            return self.uniform_sampling()
        sa_t_1, sa_t_2, r_t_1, r_t_2, extra1, extra2 = self.data_handler.get_queries(mb_size=self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1 = r_t_1[top_k_index]
        sa_t_1 = sa_t_1[top_k_index]
        r_t_2 = r_t_2[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1=extra1, img_t_2=extra2
        )
        if len(labels) > 0:
            self.data_handler.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    def kcenter_entropy_sampling(self):
        """
        K-Center Entropy Sampling 策略：
        基于预测熵选择样本进行标签生成。
        针对点云数据，同样直接采用 uniform_sampling 策略。

        返回：
            int: 成功写入缓冲区的标签数量。
        """
        if self.data_type == "pointcloud":
            return self.uniform_sampling()
        num_init = self.mb_size * self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2, extra1, extra2 = self.data_handler.get_queries(mb_size=num_init)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1 = r_t_1[top_k_index]
        sa_t_1 = sa_t_1[top_k_index]
        r_t_2 = r_t_2[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2,
                                                               img_t_1=extra1, img_t_2=extra2)
        if len(labels) > 0:
            self.data_handler.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    def uniform_sampling(self):
        """
        Uniform Sampling 策略：
        根据是否使用图像奖励、点云奖励或状态-动作向量奖励，
        分别采样查询对并生成标签，最后将查询对及标签写入缓冲区。

        返回：
            int: 成功写入缓冲区的标签数量。
        """
        if not self.vlm_label:
            if self.data_type == "pointcloud":
                sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2 = self.data_handler.get_queries(mb_size=self.mb_size)
                sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, labels = self.get_label(
                    sa_t_1, sa_t_2, r_t_1, r_t_2,
                    point_cloud_t_1=pc_t_1, point_cloud_t_2=pc_t_2
                )
                if len(labels) > 0:
                    self.data_handler.put_queries(pc_t_1, pc_t_2, labels)
            elif not self.image_reward:
                sa_t_1, sa_t_2, r_t_1, r_t_2 = self.data_handler.get_queries(mb_size=self.mb_size)
                sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
                    sa_t_1, sa_t_2, r_t_1, r_t_2
                )
                if len(labels) > 0:
                    self.data_handler.put_queries(sa_t_1, sa_t_2, labels)
            elif self.data_type == "image":
                sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2 = self.data_handler.get_queries(mb_size=self.mb_size)
                sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels = self.get_label(
                    sa_t_1, sa_t_2, r_t_1, r_t_2,
                    img_t_1=img_t_1, img_t_2=img_t_2
                )
                if len(labels) > 0:
                    self.data_handler.put_queries(img_t_1, img_t_2, labels)
        else:
            if self.cached_label_path is None:
                if self.vlm == 'pointllm_two_image':
                    if self.data_type == "pointcloud":
                        sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2 = self.data_handler.get_queries(mb_size=self.mb_size)
                        sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, labels, vlm_labels = self.get_label(
                            sa_t_1, sa_t_2, r_t_1, r_t_2,
                            img_t_1=img_t_1, img_t_2=img_t_2,
                            point_cloud_t_1=pc_t_1, point_cloud_t_2=pc_t_2
                        )
                        if len(vlm_labels) > 0:
                            self.data_handler.put_queries(pc_t_1, pc_t_2, vlm_labels)
                    elif self.data_type == "image":
                        sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2 = self.data_handler.get_queries(mb_size=self.mb_size)
                        sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels = self.get_label(
                            sa_t_1, sa_t_2, r_t_1, r_t_2,
                            img_t_1=img_t_1, img_t_2=img_t_2,
                            point_cloud_t_1=pc_t_1, point_cloud_t_2=pc_t_2
                        )
                        if len(vlm_labels) > 0:
                            self.data_handler.put_queries(img_t_1, img_t_2, vlm_labels)
                else:
                    if self.data_type == "pointcloud":
                        sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2 = self.data_handler.get_queries(mb_size=self.mb_size)
                        sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels = self.get_label(
                            sa_t_1, sa_t_2, r_t_1, r_t_2,
                            img_t_1=img_t_1, img_t_2=img_t_2,
                            point_cloud_t_1=pc_t_1, point_cloud_t_2=pc_t_2
                        )
                        if len(vlm_labels) > 0:
                            self.data_handler.put_queries(pc_t_1, pc_t_2, vlm_labels)
                    elif self.data_type == "image":
                        sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2 = self.data_handler.get_queries(mb_size=self.mb_size)
                        sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels = self.get_label(
                            sa_t_1, sa_t_2, r_t_1, r_t_2,
                            img_t_1=img_t_1, img_t_2=img_t_2,
                            point_cloud_t_1=pc_t_1, point_cloud_t_2=pc_t_2
                        )
                        if len(vlm_labels) > 0:
                            self.data_handler.put_queries(img_t_1, img_t_2, vlm_labels)
            else:
                combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_label_from_cached_states()
                num, height, width, _ = combined_images_list.shape
                img_t_1 = combined_images_list[:, :, :width // 2, :]
                img_t_2 = combined_images_list[:, :, width // 2:, :]
                labels = vlm_labels
                if len(labels) > 0:
                    self.data_handler.put_queries(img_t_1, img_t_2, labels)

        return len(labels if not self.vlm_label else vlm_labels)

    def disagreement_sampling(self):
        """
        Disagreement Sampling 策略：基于模型预测不一致性选择样本进行标签生成。

        返回：
            int: 成功写入缓冲区的标签数量。
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2, extra1, extra2 = self.data_handler.get_queries(mb_size=self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1 = r_t_1[top_k_index]
        sa_t_1 = sa_t_1[top_k_index]
        r_t_2 = r_t_2[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2,
                                                               img_t_1=extra1, img_t_2=extra2)
        if len(labels) > 0:
            self.data_handler.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)