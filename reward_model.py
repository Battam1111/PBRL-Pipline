import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time


import asyncio
from PIL import Image
import datetime
import pickle as pkl
import random
import cv2
# reward model架构修改的情况
# from simple_point_net import PointNet  # <-- 已不再使用

from prompt import (
    gemini_free_query_env_prompts, gemini_summary_env_prompts,
    gemini_free_query_prompt1, gemini_free_query_prompt2,
    gemini_single_query_env_prompts,
    gpt_free_query_env_prompts, gpt_summary_env_prompts,
    pointllm_free_query_env_prompts, pointllm_summary_env_prompts,
    pointllm_free_query_prompt1, pointllm_free_query_prompt2,
)
from vlms.gemini_infer import gemini_query_2, gemini_query_1
from conv_net import CNN, fanin_init

from vlms.pointllm_infer import pointllm_query_1, pointllm_query_2

device = 'cuda'

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

def gen_image_net(image_height, image_width, 
                  conv_kernel_sizes=[5, 3, 3 ,3], 
                  conv_n_channels=[16, 32, 64, 128], 
                  conv_strides=[3, 2, 2, 2]):
    conv_args = dict(
        kernel_sizes=conv_kernel_sizes,
        n_channels=conv_n_channels,
        strides=conv_strides,
        output_size=1,
    )
    conv_kwargs = dict(
        hidden_sizes=[],  # linear layers after conv
        batch_norm_conv=False,
        batch_norm_fc=False,
    )

    return CNN(
        **conv_args,
        paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        input_height=image_height,
        input_width=image_width,
        input_channels=3,
        init_w=1e-3,
        hidden_init=fanin_init,
        **conv_kwargs
    )

def gen_image_net2():
    from torchvision.models.resnet import ResNet
    from torchvision.models.resnet import BasicBlock

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    return model

def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index

def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device),
                            dim=-1,
                            p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, 
                 max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 
                 # vlm related params
                 vlm_label=True,
                 env_name="CartPole-v1",
                 vlm="gemini_free_form",
                 clip_prompt=None,
                 log_dir=None,
                 flip_vlm_label=False,
                 save_query_interval=25,
                 cached_label_path=None,

                 # image based reward
                 reward_model_layers=3,
                 reward_model_H=256,
                 image_reward=True,
                 image_height=128,
                 image_width=128,
                 resize_factor=1,
                 resnet=False,
                 conv_kernel_sizes=[5, 3, 3, 3],
                 conv_n_channels=[16, 32, 64, 128],
                 conv_strides=[3, 2, 2, 2],
                 **kwargs
                ):
        
        # VLM 相关
        self.vlm_label = vlm_label
        self.env_name = env_name
        self.vlm = vlm
        self.clip_prompt = clip_prompt
        self.vlm_label_acc = 0
        self.log_dir = log_dir
        self.flip_vlm_label = flip_vlm_label
        self.train_times = 0
        self.save_query_interval = save_query_interval

        # 一些基础参数
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        
        self.capacity = int(capacity)
        self.reward_model_layers = reward_model_layers
        self.reward_model_H = reward_model_H
        self.image_reward = image_reward
        self.resnet = resnet
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_n_channels = conv_n_channels
        self.conv_strides = conv_strides

        # 点云只用于“打标签”，不做训练
        self.N = 8192
        self.point_cloud_dim = self.N * 6  
        self.point_cloud_inputs = []

        # ————————————————————————————————————————
        # 根据是否使用图像reward，初始化存储 buffer_seg1, buffer_seg2
        # ————————————————————————————————————————
        if not image_reward:
            # 非图像reward，则存 (s,a) 向量
            self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
            self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        else:
            # 无论是否 vlm == 'pointllm_two_image'，这里都统一按图像大小存储
            # 不再存储点云形状到 buffer_seg，因为训练只用图像
            assert self.size_segment == 1
            self.buffer_seg1 = np.empty((self.capacity, 1, image_height, image_width, 3), dtype=np.uint8)
            self.buffer_seg2 = np.empty((self.capacity, 1, image_height, image_width, 3), dtype=np.uint8)
            self.image_height = image_height
            self.image_width = image_width
            self.resize_factor = resize_factor

        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        # 训练过程中的数据结构
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size

        # 根据是否使用图像reward来设定 batch size
        if not image_reward:
            self.train_batch_size = 128
        else:
            if not self.resnet:
                self.train_batch_size = 64
            else:
                self.train_batch_size = 32

        self.CEloss = nn.CrossEntropyLoss()

        # 教师参数
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin

        # 处理 cached_label_path
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        if cached_label_path is not None:
            self.cached_label_path = "{}/{}".format(dir_path, cached_label_path)
        else:
            self.cached_label_path = None
            
        self.read_cache_idx = 0
        if self.cached_label_path is not None:
            all_cached_labels = sorted(os.listdir(self.cached_label_path))
            self.all_cached_labels = [os.path.join(self.cached_label_path, x) for x in all_cached_labels]
        else:
            self.all_cached_labels = []

        # 构建 ensemble 模型
        self.construct_ensemble()

    def eval(self,):
        for i in range(self.de):
            self.ensemble[i].eval()

    def train(self,):
        for i in range(self.de):
            self.ensemble[i].train()
    
    def softXEnt_loss(self, input, target):
        logprobs = nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size * new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    # ————————————————————————————————————————
    # 构建网络 —— 不再使用 PointNet
    # ————————————————————————————————————————
    def construct_ensemble(self):
        for i in range(self.de):
            if not self.image_reward:
                # 原先的纯 MLP：处理 obs + action
                model = nn.Sequential(*gen_net(
                    in_size=self.ds + self.da, 
                    out_size=1, 
                    H=self.reward_model_H, 
                    n_layers=self.reward_model_layers, 
                    activation=self.activation
                )).float().to(device)

            elif self.vlm == 'pointllm_two_image':
                # 当配置为 pointllm_two_image 时，也用图像网络（不再使用 PointNet）
                if not self.resnet:
                    model = gen_image_net(
                        self.image_height, self.image_width,
                        self.conv_kernel_sizes, self.conv_n_channels, self.conv_strides
                    ).float().to(device)
                else:
                    model = gen_image_net2().float().to(device)

            else:
                # 普通图像模式
                if not self.resnet:
                    model = gen_image_net(
                        self.image_height, self.image_width,
                        self.conv_kernel_sizes, self.conv_n_channels, self.conv_strides
                    ).float().to(device)
                else:
                    model = gen_image_net2().float().to(device)

            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        # 优化器
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    # ————————————————————————————————————————
    # 采集数据，存储在 self.inputs 等结构中
    # 注意：点云只在打标签时使用，不进入训练
    # ————————————————————————————————————————
    def add_data(self, obs, act, rew, done, img=None, point_cloud=None):
        """
        添加新的数据到训练数据中(仅用于形成轨迹/打标签)。
        obs: (obs_dim,)
        act: (action_dim,)
        rew: float
        done: bool
        img: (H, W, 3) or None
        point_cloud: (N, 6) or None
        """
        sa_t = np.concatenate([obs, act], axis=-1)
        flat_input = sa_t.reshape(1, self.da + self.ds)

        r_t = np.array(rew).reshape(1, 1)
        if img is not None:
            flat_img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        if point_cloud is not None:
            # 点云只存起来用于后续打标签
            point_cloud = point_cloud.astype(np.float32)

            # 做安全检查与采样
            if point_cloud.ndim > 2:
                point_cloud = point_cloud[0]
            elif point_cloud.ndim == 1:
                point_cloud = point_cloud.reshape(-1, 6)

            flat_point_cloud = point_cloud.reshape(1, -1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(r_t)
            if img is not None:
                self.img_inputs.append(flat_img)
            if point_cloud is not None:
                self.point_cloud_inputs.append(flat_point_cloud)
        elif done:
            if 'Cloth' not in self.env_name:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], r_t])
                if img is not None:
                    self.img_inputs[-1] = np.concatenate([self.img_inputs[-1], flat_img], axis=0)
                if point_cloud is not None:
                    self.point_cloud_inputs[-1] = np.concatenate([self.point_cloud_inputs[-1], flat_point_cloud], axis=0)

                # FIFO
                if len(self.inputs) > self.max_size:
                    self.inputs = self.inputs[1:]
                    self.targets = self.targets[1:]
                    if img is not None:
                        self.img_inputs = self.img_inputs[1:]
                    if point_cloud is not None:
                        self.point_cloud_inputs = self.point_cloud_inputs[1:]
                # 开始新的轨迹
                self.inputs.append([])
                self.targets.append([])
                if img is not None:
                    self.img_inputs.append([])
                if point_cloud is not None:
                    self.point_cloud_inputs.append([])

            else:
                # ClothFold 只有一步
                self.inputs.append([flat_input])
                self.targets.append([r_t])
                if img is not None:
                    self.img_inputs.append([flat_img])
                if point_cloud is not None:
                    self.point_cloud_inputs.append([flat_point_cloud])
                if len(self.inputs) > self.max_size:
                    self.inputs = self.inputs[1:]
                    self.targets = self.targets[1:]
                    if img is not None:
                        self.img_inputs = self.img_inputs[1:]
                    if point_cloud is not None:
                        self.point_cloud_inputs = self.point_cloud_inputs[1:]
        else:
            # 还在当前轨迹
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = r_t
                if img is not None:
                    self.img_inputs[-1] = flat_img
                if point_cloud is not None:
                    self.point_cloud_inputs[-1] = flat_point_cloud
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], r_t])
                if img is not None:
                    self.img_inputs[-1] = np.concatenate([self.img_inputs[-1], flat_img], axis=0)
                if point_cloud is not None:
                    self.point_cloud_inputs[-1] = np.concatenate([self.point_cloud_inputs[-1], flat_point_cloud], axis=0)

    def add_data_batch(self, obses, rewards):
        # 用于一次性添加多条
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])

    # ————————————————————————————————————————
    # 以下是不变的评估/打标签/查询逻辑
    # ————————————————————————————————————————
    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        return F.softmax(r_hat, dim=-1)[:, 0]
    
    def p_hat_entropy(self, x_1, x_2, member=-1):
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        if isinstance(x, torch.Tensor):
            x_tensor = x.float().to(device)
        else:
            x_tensor = torch.from_numpy(x).float().to(device)
        output = self.ensemble[member](x_tensor)
        return output

    def r_hat(self, x):
        # 将 ensemble 中各成员结果求平均
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)

    def r_hat_batch(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(),
                '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(file_dir, model_dir)
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        batch_size = 256
        num_epochs = int(np.ceil(max_len / batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch + 1) * batch_size
            if last_index > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch * batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch * batch_size:last_index]
            labels = self.buffer_label[epoch * batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                r_hat1 = self.r_hat_member(sa_t_1, member=member).sum(axis=1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member).sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)

    # ————————————————————————————————————————
    # 以下是从轨迹中随机抽取对比片段的逻辑
    # ————————————————————————————————————————
    def get_queries(self, mb_size=20):
        """
        从存储的轨迹中获取查询对 (sa_t_1, sa_t_2)、(r_t_1, r_t_2)
        以及对应的图像/点云数据。
        经过本次修改后：
        - 当 self.vlm == 'pointllm_two_image' 时，
        不仅返回 point_cloud_t_1, point_cloud_t_2，
        也同时返回 img_t_1, img_t_2（如果已存过图像）。
        """
        # 1) 判断当前输入数据集长度
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        # 2) 将轨迹数据转为 numpy 方便索引
        train_inputs = np.array(self.inputs[:max_len])   # shape: (max_len, traj_len, ds+da)
        train_targets = np.array(self.targets[:max_len]) # shape: (max_len, traj_len, 1)
        
        # 3) 如果需要 VLM 或图像reward，取出图像/点云
        if self.vlm_label or self.image_reward:
            if self.vlm == 'pointllm_two_image' and self.vlm_label:
                # 同时取点云数据与图像数据（需确保 self.img_inputs 里也存了图像）
                train_point_clouds = np.array(self.point_cloud_inputs[:max_len])
                train_images = np.array(self.img_inputs[:max_len])
                # 如果您的某些环境下需要 squeeze(1)（如 Cloth），可按需处理：
                if 'Cloth' in self.env_name:
                    train_images = train_images.squeeze(1)
            else:
                # 非 pointllm_two_image 情况，如普通图像模式
                train_images = np.array(self.img_inputs[:max_len])
                if 'Cloth' in self.env_name:
                    train_images = train_images.squeeze(1)

        # 4) 随机选择两条轨迹索引
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)

        sa_t_1 = train_inputs[batch_index_1]  # (mb_size, traj_len, ds+da)
        r_t_1 = train_targets[batch_index_1]  # (mb_size, traj_len, 1)
        sa_t_2 = train_inputs[batch_index_2]
        r_t_2 = train_targets[batch_index_2]

        # 5) 若需要图像或点云，则同样随机取
        if self.vlm_label or self.image_reward:
            if self.vlm == 'pointllm_two_image' and self.vlm_label:
                point_cloud_t_1 = train_point_clouds[batch_index_1]  # (mb_size, traj_len, N*6?)
                point_cloud_t_2 = train_point_clouds[batch_index_2]
                img_t_1 = train_images[batch_index_1]                # (mb_size, traj_len, H, W, 3)?
                img_t_2 = train_images[batch_index_2]
            else:
                img_t_1 = train_images[batch_index_1]                # (mb_size, traj_len, H, W, 3)
                img_t_2 = train_images[batch_index_2]

        # 6) 将 (mb_size, traj_len, ds+da) 改成 (mb_size*traj_len, ds+da)
        #   同理，r_t_1, r_t_2 也 reshape
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])

        if self.vlm_label or self.image_reward:
            if self.vlm == 'pointllm_two_image' and self.vlm_label:
                # point_cloud
                point_cloud_t_1 = point_cloud_t_1.reshape(-1, point_cloud_t_1.shape[-1])
                point_cloud_t_2 = point_cloud_t_2.reshape(-1, point_cloud_t_2.shape[-1])
                # images
                img_t_1 = img_t_1.reshape(-1, img_t_1.shape[2], img_t_1.shape[3], img_t_1.shape[4])
                img_t_2 = img_t_2.reshape(-1, img_t_2.shape[2], img_t_2.shape[3], img_t_2.shape[4])
            else:
                # 只处理图像
                img_t_1 = img_t_1.reshape(-1, img_t_1.shape[2], img_t_1.shape[3], img_t_1.shape[4])
                img_t_2 = img_t_2.reshape(-1, img_t_2.shape[2], img_t_2.shape[3], img_t_2.shape[4])

        # 7) 处理时间片段（若 size_segment > 1，需要随机截取 segment）
        time_index = np.array([list(range(i * len_traj, i * len_traj + self.size_segment)) 
                            for i in range(mb_size)])
        if 'Cloth' not in self.env_name:
            random_idx_1 = np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)
            random_idx_2 = np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)
            time_index_1 = time_index + random_idx_1
            time_index_2 = time_index + random_idx_2
        else:
            time_index_1 = time_index
            time_index_2 = time_index

        # 8) 根据索引取出 segment
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)  # (mb_size, size_segment, ds+da)
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)

        # 同理取出图像或点云
        if self.vlm_label or self.image_reward:
            if self.vlm == 'pointllm_two_image' and self.vlm_label:
                point_cloud_t_1 = np.take(point_cloud_t_1, time_index_1, axis=0).squeeze(1)
                point_cloud_t_2 = np.take(point_cloud_t_2, time_index_2, axis=0).squeeze(1)
                img_t_1 = np.take(img_t_1, time_index_1, axis=0)
                img_t_2 = np.take(img_t_2, time_index_2, axis=0)
                
                # 拼接图像 (参考原有逻辑，如果 size_segment>1，需要把多张合并)
                batch_size, horizon, h, w, _ = img_t_1.shape
                transposed_images = np.transpose(img_t_1, (0, 2, 1, 3, 4))
                img_t_1 = transposed_images.reshape(batch_size, h, horizon * w, 3)
                transposed_images = np.transpose(img_t_2, (0, 2, 1, 3, 4))
                img_t_2 = transposed_images.reshape(batch_size, h, horizon * w, 3)
            else:
                # 普通图像模式
                img_t_1 = np.take(img_t_1, time_index_1, axis=0)
                img_t_2 = np.take(img_t_2, time_index_2, axis=0)
                batch_size, horizon, h, w, _ = img_t_1.shape
                transposed_images = np.transpose(img_t_1, (0, 2, 1, 3, 4))
                img_t_1 = transposed_images.reshape(batch_size, h, horizon * w, 3)
                transposed_images = np.transpose(img_t_2, (0, 2, 1, 3, 4))
                img_t_2 = transposed_images.reshape(batch_size, h, horizon * w, 3)

        # 9) 最终返回
        #   - 如果既没有 vlm_label 也没有 image_reward，则只返回 (sa_t_1, sa_t_2, r_t_1, r_t_2)
        #   - 否则根据是否 pointllm_two_image
        if not self.vlm_label and not self.image_reward:
            return sa_t_1, sa_t_2, r_t_1, r_t_2
        else:
            if self.vlm == 'pointllm_two_image' and self.vlm_label:
                # 返回：状态动作、奖励、点云、以及图像
                return sa_t_1, sa_t_2, r_t_1, r_t_2, point_cloud_t_1, point_cloud_t_2, img_t_1, img_t_2
            else:
                # 普通图像模式，只返回 (img_t_1, img_t_2)
                return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2


    def put_queries(self, sa_t_1, sa_t_2, labels):
        """
        将打好标签的查询片段存入 buffer_seg1, buffer_seg2 中（用于训练）。
        """
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample

        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index

            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - maximum_index
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            # 如果是图像reward，就需要在前面再加一维 (1, ...)
            if self.image_reward:
                # 去除对 pointllm_two_image 的特殊分支，统一对图像做 reshape
                sa_t_1 = sa_t_1.reshape(sa_t_1.shape[0], 1, sa_t_1.shape[1], sa_t_1.shape[2], sa_t_1.shape[3])
                sa_t_2 = sa_t_2.reshape(sa_t_2.shape[0], 1, sa_t_2.shape[1], sa_t_2.shape[2], sa_t_2.shape[3])

            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    def get_label(self, 
              sa_t_1, sa_t_2, 
              r_t_1, r_t_2,
              img_t_1=None, img_t_2=None,
              point_cloud_t_1=None, point_cloud_t_2=None):
        """
        给定两个片段，先根据真实reward算出“理性标签”，再根据 VLM 做二次修正。
        修改后要点：
        1) pointllm_two_image 模式下，除了处理点云对比拿标签之外，也统一返回图像数据 (img_t_1, img_t_2)；
        2) 其它 VLM 模式下，仅返回图像；
        3) 若不使用图像奖励 & 不使用 VLM，则只返回 (sa_t_1, sa_t_2, r_t_1, r_t_2, labels)。
        """

        # ------------------------------------------------
        # A. 根据真实 reward 先做“理性”标签
        # ------------------------------------------------
        sum_r_t_1 = np.sum(r_t_1, axis=1)  # (batch,)
        sum_r_t_2 = np.sum(r_t_2, axis=1)  # (batch,)

        # 1) teacher_thres_skip：过滤掉 reward 过低片段
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            valid_mask = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if np.sum(valid_mask) == 0:
                return None, None, None, None, []
            # 筛选有效样本
            sa_t_1 = sa_t_1[valid_mask]
            sa_t_2 = sa_t_2[valid_mask]
            r_t_1 = r_t_1[valid_mask]
            r_t_2 = r_t_2[valid_mask]
            sum_r_t_1 = sum_r_t_1[valid_mask]
            sum_r_t_2 = sum_r_t_2[valid_mask]
            if img_t_1 is not None:
                img_t_1 = img_t_1[valid_mask]
            if img_t_2 is not None:
                img_t_2 = img_t_2[valid_mask]
            if point_cloud_t_1 is not None:
                point_cloud_t_1 = point_cloud_t_1[valid_mask]
            if point_cloud_t_2 is not None:
                point_cloud_t_2 = point_cloud_t_2[valid_mask]

        # 2) 判断等价
        margin_mask = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)

        # 3) 按照“伪地面真值”生成 label
        #    - gamma折扣处理
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for idx in range(seg_size - 1):
            temp_r_t_1[:, :idx + 1] *= self.teacher_gamma
            temp_r_t_2[:, :idx + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        #    - teacher_beta随机化
        if self.teacher_beta > 0:
            # 用 softmax 后进行随机
            r_hat = np.stack([sum_r_t_1, sum_r_t_2], axis=-1)  # (batch,2)
            r_hat_tensor = torch.from_numpy(r_hat).float().to(device) * self.teacher_beta
            prob_2 = F.softmax(r_hat_tensor, dim=-1)[:, 1]  # x_2 > x_1 的概率
            random_draw = torch.bernoulli(prob_2).int().cpu().numpy().reshape(-1, 1)
            labels = random_draw  # 0 or 1
        else:
            # 不随机化
            rational_labels = (sum_r_t_1 < sum_r_t_2).astype(int)  # 0 or 1
            labels = rational_labels.reshape(-1, 1)

        # 4) 加噪声
        noise_mask = (np.random.rand(labels.shape[0]) <= self.teacher_eps_mistake)
        labels[noise_mask] = 1 - labels[noise_mask]

        # 5) 等价情况贴上 -1
        labels[margin_mask] = -1

        # ------------------------------------------------
        # B. 若无需 VLM 标注 => 直接返回
        # ------------------------------------------------
        if not self.vlm_label:
            # 不使用 VLM => 仅返回普通五元组
            # （若使用图像reward，也会进到这里, 
            #   但 self.vlm_label = False 说明不需要二次修改标签）
            if not self.image_reward:
                # 返回: sa_t_1, sa_t_2, r_t_1, r_t_2, labels
                return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
            else:
                # 返回: sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels
                return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels

        # ------------------------------------------------
        # C. 需要 VLM 标注 => 二次修改 label
        # ------------------------------------------------
        # 生成对应的 VLM label (vlm_labels)，并过滤掉无效样本
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

        if self.vlm == 'pointllm_two_image':
            # 点云对比
            vlm_labels = []
            useful_indices = []
            for idx, (pc1, pc2) in enumerate(zip(point_cloud_t_1, point_cloud_t_2)):
                dist = np.linalg.norm(pc1 - pc2)
                if dist < 1e-3:
                    # 点云几乎相同 => 跳过
                    useful_indices.append(0)
                    vlm_labels.append(-1)
                else:
                    useful_indices.append(1)
                    # 调用 pointllm_query_2
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
                        res_int = int(res)
                        if res_int not in [0, 1, -1]:
                            res_int = -1
                    except:
                        res_int = -1
                    vlm_labels.append(res_int)

            vlm_labels = np.array(vlm_labels).reshape(-1, 1)
            useful_mask = (np.array(useful_indices) == 1) & (vlm_labels.reshape(-1) != -1)

            # 根据 mask 过滤
            sa_t_1 = sa_t_1[useful_mask]
            sa_t_2 = sa_t_2[useful_mask]
            r_t_1 = r_t_1[useful_mask]
            r_t_2 = r_t_2[useful_mask]
            labels = labels[useful_mask]
            rational_labels = (sum_r_t_1 < sum_r_t_2).astype(int)[useful_mask]  # for debug
            vlm_labels = vlm_labels[useful_mask]
            point_cloud_t_1 = point_cloud_t_1[useful_mask]
            point_cloud_t_2 = point_cloud_t_2[useful_mask]
            if img_t_1 is not None:
                img_t_1 = img_t_1[useful_mask]
            if img_t_2 is not None:
                img_t_2 = img_t_2[useful_mask]

            if self.flip_vlm_label:
                vlm_labels = 1 - vlm_labels

            # 需要保存标签
            if self.train_times % self.save_query_interval == 0:
                save_path = os.path.join(self.log_dir, "vlm_label_set")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(f"{save_path}/{time_string}.pkl", "wb") as f:
                    pkl.dump([
                        point_cloud_t_1, point_cloud_t_2,
                        rational_labels, vlm_labels,
                        sa_t_1, sa_t_2, r_t_1, r_t_2
                    ], f, protocol=pkl.HIGHEST_PROTOCOL)

            # 计算准确率 (仅供 debug)
            acc = 0
            if len(vlm_labels) > 0:
                acc = np.mean((vlm_labels.flatten() == rational_labels).astype(np.float32))
                print(f"[pointllm_two_image] VLM 标签准确率: {acc:.4f}")
            else:
                print("[pointllm_two_image] 没有有效的 VLM 标签")

            self.vlm_label_acc = acc

            # 最终返回
            #   => (sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2, labels, vlm_labels)
            #   其中 img_t_1, img_t_2 供后续“写入 buffer”使用
            return sa_t_1, sa_t_2, r_t_1, r_t_2, point_cloud_t_1, point_cloud_t_2, img_t_1, img_t_2, labels, vlm_labels


        else:
            # 其它 VLM 逻辑
            gpt_two_image_paths = []
            combined_images_list = []

            file_path = os.path.abspath(__file__)
            dir_path = os.path.dirname(file_path)
            save_path = "{}/data/gpt_query_image/{}/{}".format(dir_path, self.env_name, time_string)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            useful_indices = []
            for idx, (im1, im2) in enumerate(zip(img_t_1, img_t_2)):
                combined_image = np.concatenate([im1, im2], axis=1)
                combined_images_list.append(combined_image)
                combined_image = Image.fromarray(combined_image)
                
                first_image_save_path = os.path.join(save_path, f"first_{idx:06}.png")
                second_image_save_path = os.path.join(save_path, f"second_{idx:06}.png")
                Image.fromarray(im1).save(first_image_save_path)
                Image.fromarray(im2).save(second_image_save_path)
                gpt_two_image_paths.append([first_image_save_path, second_image_save_path])

                diff = np.linalg.norm(im1 - im2)
                if diff < 1e-3:
                    useful_indices.append(0)
                else:
                    useful_indices.append(1)

            # 针对 gpt4v_two_image / gemini 等多种情况
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
                        Image.fromarray(im1),
                        gemini_free_query_prompt2,
                        Image.fromarray(im2),
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
                        Image.fromarray(im1),
                        gemini_free_query_prompt2,
                        Image.fromarray(im2),
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
                # 默认情况
                vlm_labels = []

            vlm_labels = np.array(vlm_labels).reshape(-1, 1)
            good_idx = (vlm_labels != -1).flatten()
            useful_indices = (np.array(useful_indices) == 1).flatten()
            good_idx = np.logical_and(good_idx, useful_indices)

            sa_t_1 = sa_t_1[good_idx]
            sa_t_2 = sa_t_2[good_idx]
            r_t_1 = r_t_1[good_idx]
            r_t_2 = r_t_2[good_idx]
            rational_labels = rational_labels[good_idx]
            vlm_labels = vlm_labels[good_idx]
            combined_images_list = np.array(combined_images_list)[good_idx]
            img_t_1 = img_t_1[good_idx]
            img_t_2 = img_t_2[good_idx]

            if self.flip_vlm_label:
                vlm_labels = 1 - vlm_labels

            if self.train_times % self.save_query_interval == 0 or 'gpt4v' in self.vlm:
                save_path = os.path.join(self.log_dir, "vlm_label_set")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open("{}/{}.pkl".format(save_path, time_string), "wb") as f:
                    pkl.dump([
                        combined_images_list, rational_labels, vlm_labels,
                        sa_t_1, sa_t_2, r_t_1, r_t_2
                    ], f, protocol=pkl.HIGHEST_PROTOCOL)

            acc = 0
            if len(vlm_labels) > 0:
                acc = np.sum(vlm_labels == rational_labels) / len(vlm_labels)
                print(f"vlm label acc: {acc}")
            else:
                print("no vlm label")

            self.vlm_label_acc = acc

            # 如果不是图像奖励 => 返回  (sa_t_1, sa_t_2, r_t_1, r_t_2, labels, vlm_labels)
            if not self.image_reward:
                return sa_t_1, sa_t_2, r_t_1, r_t_2, labels, vlm_labels
            else:
                # 若是图像奖励 => (sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels)
                return sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels

    # ————————————————————————————————————————
    # 不同查询策略
    # ————————————————————————————————————————
    def kcenter_sampling(self):
        num_init = self.mb_size * self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=num_init)
        
        # 做KCenter
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        
        # 打标签
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    
    def kcenter_disagree_sampling(self):
        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=num_init)
        
        # 根据不一致度选TopK
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # KCenter
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    
    def kcenter_entropy_sampling(self):
        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=num_init)
        
        # 根据熵选TopK
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # KCenter
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    
    def uniform_sampling(self):
        """
        修改后示例：
        1) 无论是否 pointllm_two_image，只要有 image_reward，就返回 img_t_1, img_t_2 并写入 buffer。
        2) 若既无 image_reward 也无 vlm_label，保持原逻辑只返回 (sa_t_1, sa_t_2, r_t_1, r_t_2) 并 put_queries。
        3) 若有 vlm_label & pointllm_two_image，则同时获取 pc_t & img_t，并用 pc_t 打标签，但最后写 buffer 依旧是图像数据。
        """

        # 1) 根据是否 vlm_label / image_reward / cached_label_path 等条件，调用 get_queries & get_label
        if not self.vlm_label:
            # 不使用 VLM 标签
            if not self.image_reward:
                # A. 无图像奖励、无VLM => get_queries 返回 (sa_t_1, sa_t_2, r_t_1, r_t_2)
                sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=self.mb_size)
                # get_label 返回 (sa_t_1, sa_t_2, r_t_1, r_t_2, labels)
                sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
            else:
                # B. 有图像奖励、无VLM => get_queries 返回 (sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2)
                sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2 = self.get_queries(mb_size=self.mb_size)
                # get_label 返回 (sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels)
                sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels = self.get_label(
                    sa_t_1, sa_t_2, r_t_1, r_t_2,
                    img_t_1=img_t_1, 
                    img_t_2=img_t_2
                )
        else:
            # 使用 VLM 标签
            #   可能是 pointllm_two_image 或 其他 gemini / gpt4v ...
            if self.cached_label_path is None:
                # 在线查询
                if self.vlm == 'pointllm_two_image':
                    # C. pointllm_two_image => get_queries 返回 (sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2)
                    sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2 = self.get_queries(mb_size=self.mb_size)
                    # get_label 返回 (sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2, labels, vlm_labels)
                    sa_t_1, sa_t_2, r_t_1, r_t_2, pc_t_1, pc_t_2, img_t_1, img_t_2, gt_labels, vlm_labels = self.get_label(
                        sa_t_1, sa_t_2, r_t_1, r_t_2,
                        point_cloud_t_1=pc_t_1,
                        point_cloud_t_2=pc_t_2,
                        img_t_1=img_t_1,
                        img_t_2=img_t_2
                    )
                else:
                    # D. 其他 VLM 模式 => get_queries 返回 (sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2)
                    sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2 = self.get_queries(mb_size=self.mb_size)
                    # get_label => (sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, labels, vlm_labels)
                    sa_t_1, sa_t_2, r_t_1, r_t_2, img_t_1, img_t_2, gt_labels, vlm_labels = self.get_label(
                        sa_t_1, sa_t_2, r_t_1, r_t_2,
                        img_t_1=img_t_1, 
                        img_t_2=img_t_2
                    )
            else:
                # E. 从缓存读取
                if self.read_cache_idx < len(self.all_cached_labels):
                    if self.vlm == 'pointllm_two_image':
                        # 缓存里存的也要保证能够同时返回 pc + 图像(如 combined_images_list)
                        pc_t_1, pc_t_2, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_label_from_cached_states()
                        # 注意：若还需要 img_t_1, img_t_2，也得保证 get_label_from_cached_states 里有
                        # 暂未实现（不过目前也用不到）
                        pass
                    else:
                        combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_label_from_cached_states()
                        # 同理，如果是 gemini / gpt4v 等，需要在 combined_images_list 里切分出 img_t_1, img_t_2
                        num, height, width, _ = combined_images_list.shape
                        img_t_1 = combined_images_list[:, :, :width // 2, :]
                        img_t_2 = combined_images_list[:, :, width // 2:, :]
                        # 若需要 resize
                        if 'Rope' not in self.env_name and 'Water' not in self.env_name:
                            resized_img_t_1 = np.zeros((num, self.image_height, self.image_width, 3), dtype=np.uint8)
                            resized_img_t_2 = np.zeros((num, self.image_height, self.image_width, 3), dtype=np.uint8)
                            for idx in range(len(img_t_1)):
                                resized_img_t_1[idx] = cv2.resize(img_t_1[idx], (self.image_height, self.image_width))
                                resized_img_t_2[idx] = cv2.resize(img_t_2[idx], (self.image_height, self.image_width))
                            img_t_1 = resized_img_t_1
                            img_t_2 = resized_img_t_2
                else:
                    vlm_labels = []

            # 这里最终标签全都放在labels里
            labels = vlm_labels

        # 2) 将得到的 labels 放入 buffer
        #    先判断是否有有效数据
        if len(labels) > 0:
            # 情况 1：无图像奖励 & 非 pointllm_two_image => 纯向量写 buffer
            if not self.image_reward and self.vlm != 'pointllm_two_image':
                self.put_queries(sa_t_1, sa_t_2, labels)

            # 情况 2：有图像奖励 => 需要将 img_t_1, img_t_2 下采样后写 buffer
            else:
                # 这里将 pointllm_two_image 情况也统一起来，只要“image_reward=True”，就写图像
                #   => 所以必须保证在上方 get_label 中已经获取到 img_t_1, img_t_2
                #   => 并且在 pointllm_two_image 下 get_label 的返回值里确实含有 img_t_1, img_t_2
                sa_t_1 = img_t_1[:, ::self.resize_factor, ::self.resize_factor, :]
                sa_t_2 = img_t_2[:, ::self.resize_factor, ::self.resize_factor, :]
                self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)


    def get_label_from_cached_states(self):
        if self.read_cache_idx >= len(self.all_cached_labels):
            return None, None, None, None, None, []
        with open(self.all_cached_labels[self.read_cache_idx], 'rb') as f:
            data = pkl.load(f)
        self.read_cache_idx += 1

        # data 格式: [combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2]
        combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = data
        return combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2
    
    def disagreement_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]

        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    
    def entropy_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size]

        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)

    # ————————————————————————————————————————
    # 训练 reward
    # ————————————————————————————————————————
    def train_reward(self):
        """
        训练奖励模型 (CE Loss)，对 <seg1, seg2> 做二分类。
        """
        self.train_times += 1
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index

        # 每个 ensemble 成员对应一个随机洗牌
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss_all = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member == 0:
                    total += labels.size(0)

                # 若是图像（含 pointllm_two_image 模式），统一按图像处理
                if self.image_reward:
                    sa_t_1 = np.transpose(sa_t_1, (0, 1, 4, 2, 3))  # (B,1,H,W,3) -> (B,1,3,H,W)
                    sa_t_2 = np.transpose(sa_t_2, (0, 1, 4, 2, 3))
                    sa_t_1 = sa_t_1.astype(np.float32) / 255.0
                    sa_t_2 = sa_t_2.astype(np.float32) / 255.0
                    sa_t_1 = sa_t_1.squeeze(1)  # (B,3,H,W)
                    sa_t_2 = sa_t_2.squeeze(1)

                sa_t_1 = torch.from_numpy(sa_t_1).float().to(device)
                sa_t_2 = torch.from_numpy(sa_t_2).float().to(device)

                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)

                # 非图像模式，就 sum(axis=1) ；图像模式因为输出单值，也可以 sum(axis=1) 不过是 scalar
                if not self.image_reward:
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)

                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                curr_loss = self.CEloss(r_hat, labels)
                loss_all += curr_loss

                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss_all.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total
        torch.cuda.empty_cache()
        return ensemble_acc
    
    def train_soft_reward(self):
        """
        处理等价标签(-1)的 soft CE。
        """
        self.train_times += 1
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index

        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss_all = 0.0

            last_index = (epoch + 1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member == 0:
                    total += labels.size(0)

                if self.image_reward:
                    sa_t_1 = np.transpose(sa_t_1, (0, 1, 4, 2, 3))
                    sa_t_2 = np.transpose(sa_t_2, (0, 1, 4, 2, 3))
                    sa_t_1 = sa_t_1.astype(np.float32) / 255.0
                    sa_t_2 = sa_t_2.astype(np.float32) / 255.0
                    sa_t_1 = sa_t_1.squeeze(1)
                    sa_t_2 = sa_t_2.squeeze(1)

                sa_t_1 = torch.from_numpy(sa_t_1).float().to(device)
                sa_t_2 = torch.from_numpy(sa_t_2).float().to(device)

                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                if not self.image_reward:
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)

                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # 等价用 0.5
                uniform_index = (labels == -1)
                labels[uniform_index] = 0  # 先改成0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5

                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss_all += curr_loss

                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss_all.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total
        torch.cuda.empty_cache()
        return ensemble_acc
