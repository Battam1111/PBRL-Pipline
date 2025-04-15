#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_PEBBLE.py
===============

本文件实现PEBBLE算法的训练流程，支持强化学习过程中同时利用奖励模型进行奖励函数的学习。
在本版本中，针对奖励模型使用图像数据与点云数据两种情况进行了适配：
- 当配置参数 reward_data_type=="image" 时，训练过程中使用图像数据（例如经过resize、归一化后的RGB图像）作为奖励模型的输入；
- 当 reward_data_type=="pointcloud" 时，训练过程中使用点云数据作为奖励模型的输入（例如环境通过 render(mode='pointcloud') 得到的数据）。

作者：您的姓名
日期：2023-…（更新日期）
"""

import numpy as np
import torch
import os
import time
import pickle as pkl
from collections import deque

from logger import Logger            # 日志记录模块
from replay_buffer import ReplayBuffer  # 回放缓冲区模块
from reward_model import RewardModel    # 奖励模型模块（已经适配点云支持）
from reward_model_score import RewardModelScore  # 基于分数的奖励模型模块
from prompt import clip_env_prompts     # 用于设置CLIP提示的环境变量
# 数据保存相关模块（图像、点云）
from dataCollection.DataSaver.image_saver import ImageSaver  
from dataCollection.DataSaver.pointcloud_saver import PointCloudSaver

import utils       # 实用函数库
import hydra       # 配置管理库
from PIL import Image  # 图像处理
from vlms.blip_infer_2 import blip2_image_text_matching  # BLIP2图像-文本匹配
from vlms.clip_infer import clip_infer_score as clip_image_text_matching  # CLIP图像-文本匹配
from vlms.pointllm_infer import init_model   # 初始化PointLLM模型
import cv2         # 计算机视觉库

class Workspace(object):
    def __init__(self, cfg):
        # 获取当前工作目录
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg  # 配置文件对象

        # 设置CLIP提示（用于某些VLM模型）
        self.cfg.prompt = clip_env_prompts[cfg.env]
        self.cfg.clip_prompt = clip_env_prompts[cfg.env]
        self.reward = self.cfg.reward  # 奖励类型

        # 初始化日志记录器
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name
        )
        
        # 如果使用 pointllm 两图模式，则初始化 PointLLM 模型，并强制设置 image_reward 为 True
        if self.cfg.vlm == 'pointllm_two_image' and self.cfg.vlm_label == 1:
            init_model()
            print("PointLLM model initialized.")
            # 实际使用图像奖励模型（但奖励数据可以由点云提供，见下文reward_data_type参数）
            self.cfg.image_reward = True

        # 设置随机种子
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False  # 成功记录标志

        current_file_path = os.path.dirname(os.path.realpath(__file__))
        # 将 prompt.py 文件复制到日志目录（记录配置）
        os.system("cp {}/prompt.py {}/".format(current_file_path, self.logger._log_dir))
        
        # 创建环境（支持多种环境类型）
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        elif cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
            self.env = utils.make_classic_control_env(cfg)
        elif 'softgym' in cfg.env:
            self.env = utils.make_softgym_env(cfg)
        else:
            self.env = utils.make_env(cfg)
        
        # 设置代理参数维度（观察、动作空间）
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        # 实例化代理
        self.agent = hydra.utils.instantiate(cfg.agent)
        
        # 根据环境配置设置图像尺寸与缩放因子
        image_height = image_width = cfg.image_size
        self.resize_factor = 1
        if "sweep" in cfg.env or 'drawer' in cfg.env or "soccer" in cfg.env:
            image_height = image_width = 300 
        if "Rope" in cfg.env:
            image_height = image_width = 240
            self.resize_factor = 3
        elif "Water" in cfg.env:
            image_height = image_width = 360
            self.resize_factor = 2
        if "CartPole" in cfg.env:
            image_height = image_width = 200
        if "Cloth" in cfg.env:
            image_height = image_width = 360
            
        self.image_height = image_height
        self.image_width = image_width

        # 初始化图像保存器
        # self.image_saver = ImageSaver(
        #     task=cfg.env,
        #     output_dir="/home/star/Yanjun/RL-VLM-F/test/images",
        # )

        # 初始化点云保存器
        # self.pointcloud_saver = PointCloudSaver(
        #     task=cfg.env,
        #     output_dir="/home/star/Yanjun/RL-VLM-F/data/pointclouds",
        #     second_output_dir="/home/star/Yanjun/RL-VLM-F/data/DensePointClouds",
        # )

        # 根据配置参数 reward_data_type 决定回放缓冲区存储方式：
        # 如果 reward_data_type 为 "image"，则启用图像存储；
        # 如果 reward_data_type 为 "pointcloud"，则启用点云存储。
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            # 根据具体情况可以调整容量，这里保持一致
            int(cfg.replay_buffer_capacity) if cfg.reward_data_type == "pointcloud" else 200000,
            self.device,
            store_image=True,
            store_point_cloud=(cfg.reward_data_type == "pointcloud" or self.cfg.vlm == 'pointllm_two_image'),
            image_size=image_height
        )
        
        # 日志相关变量初始化
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # -------------------------------
        # 实例化奖励模型
        # 传入参数中新增 reward_data_type 和 point_cloud_num_points，用于支持点云数据
        # -------------------------------
        reward_model_class = RewardModel
        if self.reward == 'learn_from_preference':
            reward_model_class = RewardModel
        elif self.reward == 'learn_from_score':
            reward_model_class = RewardModelScore
        
        self.reward_model = reward_model_class(
            # PEBBLE原始参数
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            capacity=cfg.max_feedback * 2,
            
            # VLM相关参数
            vlm_label=cfg.vlm_label,
            vlm=cfg.vlm,
            env_name=cfg.env,
            clip_prompt=clip_env_prompts[cfg.env],
            log_dir=self.logger._log_dir,
            flip_vlm_label=cfg.flip_vlm_label,
            cached_label_path=cfg.cached_label_path,

            # 基于图像的奖励模型参数
            image_reward=cfg.image_reward,
            image_height=image_height,
            image_width=image_width,
            resize_factor=self.resize_factor,
            resnet=cfg.resnet,
            conv_kernel_sizes=cfg.conv_kernel_sizes,
            conv_strides=cfg.conv_strides,
            conv_n_channels=cfg.conv_n_channels,

            # 新增：数据类型和点云相关参数
            data_type=self.cfg.reward_data_type,   # "image" 或 "pointcloud"
            point_cloud_num_points=self.cfg.point_cloud_num_points,      # 点云数据中点的数量
        )
        
        # 加载已有奖励模型（若配置了加载路径）
        if self.cfg.reward_model_load_dir != "None":
            print("loading reward model at {}".format(self.cfg.reward_model_load_dir))
            self.reward_model.load(self.cfg.reward_model_load_dir, 1000000)
                
        # 加载已有代理模型（若配置了加载路径）
        if self.cfg.agent_model_load_dir != "None":
            print("loading agent model at {}".format(self.cfg.agent_model_load_dir))
            self.agent.load(self.cfg.agent_model_load_dir, 1000000)
        
    def evaluate(self, save_additional=False):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)

        all_ep_infos = []
        for episode in range(self.cfg.num_eval_episodes):
            print("evaluating episode {}".format(episode))
            images = []
            obs = self.env.reset()
            if "metaworld" in self.cfg.env:
                obs = obs[0]

            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            ep_info = []
            rewards = []
            t_idx = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                try:
                    obs, reward, done, extra = self.env.step(action)
                except:
                    obs, reward, terminated, truncated, extra = self.env.step(action)
                    done = terminated or truncated
                ep_info.append(extra)
                rewards.append(reward)
                if "metaworld" in self.cfg.env:
                    render_image = self.env.render()
                    if self.cfg.mode != 'eval':
                        render_image = render_image[::-1, :, :]
                        if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                            render_image = render_image[100:400, 100:400, :]
                    else:
                        render_image = render_image[::-1, :, :]
                elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                    render_image = self.env.render(mode='rgb_array')
                else:
                    render_image = self.env.render(mode='rgb_array')

                if 'softgym' not in self.cfg.env:
                    images.append(render_image)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                    
                t_idx += 1
                if self.cfg.mode == 'eval' and t_idx > 50:
                    break
                    
            all_ep_infos.append(ep_info)
            if 'softgym' in self.cfg.env:
                images = self.env.video_frames
                
            save_gif_path = os.path.join(save_gif_dir, 'step{:07}_episode{:02}_{}.gif'.format(self.step, episode, round(true_episode_reward, 2)))
            utils.save_numpy_as_gif(np.array(images), save_gif_path)
            if save_additional:
                save_image_dir = os.path.join(self.logger._log_dir, 'eval_images')
                if not os.path.exists(save_image_dir):
                    os.makedirs(save_image_dir)
                for i, image in enumerate(images):
                    save_image_path = os.path.join(save_image_dir, 'step{:07}_episode{:02}_{}.png'.format(self.step, episode, i))
                    Image.fromarray(image).save(save_image_path)
                save_reward_path = os.path.join(self.logger._log_dir, "eval_reward")
                if not os.path.exists(save_reward_path):
                    os.makedirs(save_reward_path)
                with open(os.path.join(save_reward_path, "step{:07}_episode{:02}.pkl".format(self.step, episode)), "wb") as f:
                    pkl.dump(rewards, f)
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate = (success_rate / self.cfg.num_eval_episodes) * 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward, self.step)
        for key, value in extra.items():
            self.logger.log('eval/' + key, value, self.step)

        if self.log_success:
            self.logger.log('eval/success_rate', success_rate, self.step)
            self.logger.log('train/true_episode_success', success_rate, self.step)
            
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
        # 获取反馈数量
        labeled_queries = 0 
        if first_flag == 1:
            # 首次反馈使用随机采样
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            # 根据配置选择不同采样方式
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        # 更新反馈计数
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        total_acc = 0
        if self.labeled_feedback > 0:
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    self.reward_model.train()
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    self.reward_model.train()
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break
        
        if self.reward == 'learn_from_preference':
            print("Reward function is updated!! ACC: " + str(total_acc))
        elif self.reward == 'learn_from_score':
            print("Reward function is updated!! MSE: " + str(total_acc))
        return total_acc, self.reward_model.vlm_label_acc

    def run(self):
        model_save_dir = os.path.join(self.work_dir, "models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # 初始化训练参数
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0

        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()

        interact_count = 0
        reward_learning_acc = 0
        vlm_acc = 0
        eval_cnt = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    self.logger.log('train/reward_learning_acc', reward_learning_acc, self.step)
                    self.logger.log('train/vlm_acc', vlm_acc, self.step)
                    for key, value in extra.items():
                        self.logger.log('train/' + key, value, self.step)
                    start_time = time.time()
                    if "Cloth" in self.cfg.env:
                        self.logger.dump(self.step, save=((self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps)))
                    else:
                        self.logger.dump(self.step, save=((self.step > self.cfg.num_seed_steps)))
                        
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    eval_cnt += 1

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)

                obs = self.env.reset()
                if "metaworld" in self.cfg.env:
                    obs = obs[0]
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

                traj_images = []
                ep_info = []

            # 采集阶段：若处于种子阶段，则随机动作，否则由代理产生
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # 更新阶段：根据训练进程执行不同更新策略
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                print("finished unsupervised exploration!!")
                if self.reward in ['learn_from_preference', 'learn_from_score']:
                    if self.cfg.reward_schedule == 1:
                        frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                        if frac == 0:
                            frac = 0.01
                    elif self.cfg.reward_schedule == 2:
                        frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                    else:
                        frac = 1
                    self.reward_model.change_batch(frac)
                    new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                    self.reward_model.set_teacher_thres_skip(new_margin)
                    self.reward_model.set_teacher_thres_equal(new_margin)
                    reward_learning_acc, vlm_acc = self.learn_reward(first_flag=1)
                    self.reward_model.eval()
                    self.replay_buffer.relabel_with_predictor(self.reward_model)
                    self.reward_model.train()
                self.agent.reset_critic()
                self.agent.update_after_reset(self.replay_buffer, self.logger, self.step, gradient_update=self.cfg.reset_update, policy_update=True)
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                if self.total_feedback < self.cfg.max_feedback and self.reward in ['learn_from_preference', 'learn_from_score']:
                    if interact_count == self.cfg.num_interact:
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                        reward_learning_acc, vlm_acc = self.learn_reward()
                        self.reward_model.eval()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        self.reward_model.train()
                        interact_count = 0
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
            elif self.step > self.cfg.num_seed_steps:
                if self.step % 1000 == 0:
                    print("unsupervised exploration!!")
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, gradient_update=1, K=self.cfg.topK)
            try:
                next_obs, reward, done, extra = self.env.step(action)
            except:
                next_obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated
            ep_info.append(extra)

            # ===== 核心修改：获取图像和点云数据 =====
            render_image = None
            point_cloud = None

            # 始终获取图像数据
            if "metaworld" in self.cfg.env:
                render_image = self.env.render()
                render_image = render_image[::-1, :, :]
                if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                    render_image = render_image[100:400, 100:400, :]
            elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                render_image = self.env.render(mode='rgb_array')
            elif 'softgym' in self.cfg.env:
                render_image = self.env.render(mode='rgb_array', hide_picker=True)
            else:
                render_image = self.env.render(mode='rgb_array')

            # 当使用 pointllm_two_image 模式时，同时获取点云数据
            if self.cfg.vlm == 'pointllm_two_image' or self.cfg.reward_data_type == "pointcloud":
                point_cloud = self.env.render(mode='pointcloud')

            # 保存图像数据（保存器相关代码可根据需要启用）
            # self.image_saver.save_data(render_image)
            # self.pointcloud_saver.save_data(point_cloud, reward)

            if self.cfg.image_reward and render_image is not None:
                if 'Water' not in self.cfg.env and 'Rope' not in self.cfg.env:
                    render_image = cv2.resize(render_image, (self.image_height, self.image_width))
                traj_images.append(render_image)

            # ===== 奖励估计部分 =====
            if self.reward in ['learn_from_preference', 'learn_from_score']:
                if self.cfg.reward_data_type == "pointcloud":
                    # 使用点云作为奖励模型输入
                    # 为防止单样本归一化层统计不稳定，先切换奖励模型到评估模式
                    self.reward_model.eval()
                    if point_cloud is not None:
                        # 注意：此处 point_cloud 通常形状为 (N, 6)，取第一个样本，reshape 为 (1, point_cloud_num_points, 6)
                        pc = point_cloud[0]
                        pc = pc.reshape(1, self.cfg.point_cloud_num_points, 6)
                        reward_hat = self.reward_model.r_hat(pc)
                    else:
                        reward_hat = reward
                    # 计算完毕后恢复模型到训练模式
                    self.reward_model.train()
                elif not self.cfg.image_reward:
                    # 使用状态-动作向量作为输入（纯 MLP）
                    self.reward_model.eval()
                    reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
                    self.reward_model.train()
                else:
                    # 使用图像作为输入
                    image = render_image.transpose(2, 0, 1).astype(np.float32) / 255.0
                    image = image[:, ::self.resize_factor, ::self.resize_factor]
                    image = image.reshape(1, 3, image.shape[1], image.shape[2])
                    self.reward_model.eval()
                    reward_hat = self.reward_model.r_hat(image)
                    self.reward_model.train()
            elif self.reward == 'blip2_image_text_matching':
                query_image = render_image
                query_prompt = clip_env_prompts[self.cfg.env] 
                reward_hat = blip2_image_text_matching(query_image, query_prompt) * 2 - 1
                if self.cfg.flip_vlm_label:
                    reward_hat = -reward_hat
            elif self.reward == 'clip_image_text_matching':
                query_image = render_image
                query_prompt = clip_env_prompts[self.cfg.env] 
                reward_hat = clip_image_text_matching(query_image, query_prompt) * 2 - 1
                if self.cfg.flip_vlm_label:
                    reward_hat = -reward_hat
            elif self.reward == 'gt_task_reward':
                reward_hat = reward
            elif self.reward == 'sparse_task_reward':
                reward_hat = extra['success']
            else:
                reward_hat = reward

            # 处理回合结束标志
            done = float(done)
            if 'softgym' not in self.cfg.env:
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            else:
                done_no_max = done

            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
            
            # 添加奖励训练数据（同时传入图像和点云数据，内部会根据 data_type 进行处理）
            if self.reward in ['learn_from_preference', 'learn_from_score']:
                self.reward_model.add_data(obs, action, reward, done, img=render_image, point_cloud=point_cloud[0] if point_cloud is not None else None)

            # 将数据添加到回放缓冲区
            if (self.cfg.reward_data_type == "image" or self.cfg.reward_data_type == "pointcloud") and (self.reward not in ["gt_task_reward", "sparse_task_reward"]):
                # 如果使用图像数据作为奖励输入，则传入预处理后的图像数据
                self.replay_buffer.add(
                    obs, action, reward_hat, next_obs, done, done_no_max,
                    image=render_image[::self.resize_factor, ::self.resize_factor, :] if render_image is not None else None,
                    point_cloud=point_cloud[0] if point_cloud is not None else None
                )
            else:
                # 其它情况，均不传入图像与点云数据
                self.replay_buffer.add(
                    obs, action, reward_hat, next_obs, done, done_no_max,
                    image=None,
                    point_cloud=None
                )


            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            
            # 定期保存模型
            if self.step % self.cfg.save_interval == 0 and self.step > 0:
                self.agent.save(model_save_dir, self.step)
                self.reward_model.save(model_save_dir, self.step)
            
        # 最后保存一次模型
        self.agent.save(model_save_dir, self.step)
        self.reward_model.save(model_save_dir, self.step)



        
@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    # 手动设置参数
    

    # 设置环境名称
    cfg.env = 'metaworld_soccer-v2'  # 使用元世界足球环境
    # cfg.env = 'metaworld_drawer-open-v2' # 使用元世界抽屉打开环境
    # cfg.env = 'metaworld_door-open-v2'  # 使用元世界开门环境
    # cfg.env = 'metaworld_disassemble-v2' # 使用元世界拆卸环境
    
    # cfg.env = 'metaworld_handle-pull-side-v2' # 使用元世界拉手环境
    # cfg.env = 'metaworld_peg-insert-side-v2' # 使用元世界插销环境

    # cfg.env = 'metaworld_hand-insert-v2' # 使用元世界手插入环境（有问题）
    # cfg.env = 'metaworld_sweep-into-v2' # 使用元世界扫描环境（有问题）
    # cfg.env = 'metaworld_shelf-place-v2' # 使用元世界架子放置环境(用不了)
    
    # 设置随机种子，确保实验的可重复性
    cfg.seed = 0  
    
    # 设置实验名称，用于日志记录和文件保存路径
    cfg.exp_name = 'reproduce'  
    
    # 设置奖励类型为基于偏好学习
    cfg.reward = 'learn_from_preference'
    
    # 设置视觉语言模型（VLM）的相关参数
    cfg.vlm_label = 0  # 使用VLM标签(0/1)
    # cfg.vlm = 'pointllm_two_image'  # 使用名为`pointllm_two_image`的VLM模型
    cfg.vlm = ''  # 不使用vlm
    # cfg.vlm = 'gpt4v_two_image'  # 另一种可选的VLM模型（已注释）
    
    # 图像奖励相关配置
    cfg.image_reward = 1  # 启用基于图像的奖励
    
    # 奖励学习的相关参数
    cfg.reward_batch = 40  # 奖励模型每次更新的批量大小
    cfg.segment = 1  # 奖励模型学习的片段长度（设置为1以最小化开销）
    cfg.teacher_eps_mistake = 0  # 教师错误的概率
    cfg.reward_update = 5  # 奖励模型更新频率

    # cfg.num_interact = 4000  # 与环境交互的总步数

    # cfg.num_interact = 1000  # 缩小与环境交互的步数
    # cfg.max_feedback = 20000  # 最大用户反馈次数
    cfg.reward_lr = 1e-4  # 奖励模型的学习率

    # 强化学习代理参数
    cfg.agent.params.actor_lr = 0.0003  # 策略网络的学习率
    cfg.agent.params.critic_lr = 0.0003  # 值函数网络的学习率
    cfg.gradient_update = 1  # 每次只进行1次梯度更新
    cfg.activation = 'tanh'  # 使用tanh作为激活函数
    
    # 无监督学习和总训练步数
    # cfg.num_unsup_steps = 9000  # 无监督学习步数
    # cfg.num_train_steps = 100000  # 总训练步数
    
    # 强化学习训练的批量大小
    # cfg.agent.params.batch_size = 512  
    cfg.agent.params.batch_size = 128  # 缩小为128
    
    # 值函数网络的结构配置
    cfg.double_q_critic.params.hidden_dim = 256  # 隐藏层维度
    cfg.double_q_critic.params.hidden_depth = 3  # 隐藏层深度（层数）
    
    # 策略网络的结构配置
    cfg.diag_gaussian_actor.params.hidden_dim = 256  # 隐藏层维度
    cfg.diag_gaussian_actor.params.hidden_depth = 3  # 隐藏层深度（层数）

    # 反馈类型与教师策略参数
    cfg.feed_type = 0  # 设置反馈类型
    cfg.teacher_beta = -1  # 教师策略参数，用于控制偏好
    cfg.teacher_gamma = 1  # 教师的折扣因子
    cfg.teacher_eps_skip = 0  # 教师跳过反馈的概率
    cfg.teacher_eps_equal = 0  # 教师提供中性反馈的概率
    
    # 评估参数
    cfg.num_eval_episodes = 1  # 每次评估的回合数
    # cfg.cached_label_path = 'data/cached_labels/Soccer/seed_1/'  # 缓存的标签路径（已注释）

    # 初始化工作空间并开始训练
    workspace = Workspace(cfg)  # 基于配置初始化工作空间

    # 如果模式为评估模式
    if cfg.mode == 'eval':
        workspace.evaluate(save_additional=cfg.save_images)  # 进行评估，是否保存额外数据由`cfg.save_images`决定
        exit()  # 退出程序

    # 启动训练过程
    workspace.run()

# 主函数入口
if __name__ == '__main__':
    main()