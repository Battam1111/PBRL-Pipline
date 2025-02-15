import numpy as np
import torch
import utils

def robust_fix_image_array(img):
    """
    确保图像数组为 3 维 (H, W, C) 格式。
    如果检测到额外的单一维度（例如形状 (1, H, W, C) 或 (1,1,H,W,C)），则移除这些单一维度。
    若仍然多余，则尝试取第一个样本作为代表。
    
    参数：
        img (np.array): 待处理的图像数组。
    返回：
        np.array: 处理后的图像数组，形状为 (H, W, C) 或 (H, W)。
    """
    # 逐步去除前导的单一维度
    while img.ndim > 3 and img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    # 如果仍有多余维度（注意保留最后一个颜色通道维度），尝试对非最后一维进行 squeeze
    if img.ndim > 3:
        new_dims = []
        for i, d in enumerate(img.shape):
            # 保留最后一维（颜色通道）
            if i == img.ndim - 1:
                new_dims.append(d)
            else:
                if d != 1:
                    new_dims.append(d)
        if len(new_dims) == 3:
            img = img.reshape(new_dims)
    # 若依然多余，则直接取第一个元素（例如：如果出现 (N, H, W, C)，则选择第一个样本）
    if img.ndim > 3:
        img = img[0]
    return img

class ReplayBuffer(object):
    """存储环境转移数据的回放缓冲区"""
    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        device,
        window=1,
        store_image=False,
        image_size=300,
        store_point_cloud=False,
        point_cloud_size=8192
    ):
        """
        参数：
            obs_shape: 观测空间的形状（例如 (11,)）
            action_shape: 动作空间的形状（例如 (4,)）
            capacity: 缓冲区容量
            device: PyTorch 设备（cuda 或 cpu）
            window: 保留参数（目前未使用）
            store_image: 是否存储图像数据
            image_size: 如果存储图像，则图像的高宽（正方形）
            store_point_cloud: 是否存储点云数据
            point_cloud_size: 点云中点的数量（每个点包含 x, y, z, r, g, b 六个分量）
        """
        self.capacity = capacity
        self.device = device

        # 如果观测只有一维（如状态向量），使用 float32；否则（如图像）使用 uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.window = window
        self.store_image = store_image
        self.store_point_cloud = store_point_cloud

        # 如果需要存储图像，则初始化形状为 (capacity, image_size, image_size, 3) 的数组
        self.images = np.empty((capacity, image_size, image_size, 3), dtype=np.uint8)

        # 如果需要存储点云，则初始化形状为 (capacity, point_cloud_size, 6) 的数组
        if self.store_point_cloud:
            self.point_clouds = np.empty((capacity, point_cloud_size, 6), dtype=np.float32)

        self.idx = 0          # 当前插入索引
        self.full = False     # 标记缓冲区是否已满

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, image=None, point_cloud=None):
        """
        向缓冲区添加一条经验数据 (obs, action, reward, next_obs, done, ...)。
        如果 store_image 为 True 且传入 image，则存储图像；若传入 point_cloud 则存储点云数据。
        """
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        if image is not None and self.store_image:
            np.copyto(self.images[self.idx], image)

        if point_cloud is not None and self.store_point_cloud:
            point_cloud = point_cloud.astype(np.float32)
            np.copyto(self.point_clouds[self.idx], point_cloud)

        # 环形缓冲
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or (self.idx == 0)

    def relabel_with_predictor(self, predictor):
        """
        使用奖励模型 predictor 对回放缓冲区中的经验重新估计奖励。
        注意：根据 predictor 的数据类型选择合适的输入。
            - 如果 predictor.data_type == "pointcloud" 且缓冲区存储了点云数据，则使用 self.point_clouds；
            - 如果 predictor.data_type == "image"，则使用 self.images；
            - 否则，使用 obs+action 拼接作为输入。
        """
        if hasattr(predictor, 'data_type') and predictor.data_type == "pointcloud" and self.store_point_cloud:
            batch_size = 32  # 点云数据较大，使用较小的 batch_size
        elif self.store_image:
            batch_size = 32
        else:
            batch_size = 200

        total_samples = self.idx if not self.full else self.capacity
        total_iter = (total_samples + batch_size - 1) // batch_size

        for i in range(total_iter):
            start = i * batch_size
            end = min(start + batch_size, total_samples)
            if hasattr(predictor, 'data_type') and predictor.data_type == "pointcloud" and self.store_point_cloud:
                inputs = self.point_clouds[start:end]
            elif self.store_image:
                imgs = self.images[start:end]
                # 转换为 (B, 3, H, W) 并归一化
                inputs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0
            else:
                obses = self.obses[start:end]
                acts = self.actions[start:end]
                inputs = np.concatenate([obses, acts], axis=-1)
            pred_reward = predictor.r_hat_batch(inputs)
            pred_reward = pred_reward.reshape(-1, 1)
            self.rewards[start:end] = pred_reward

        torch.cuda.empty_cache()

    def sample(self, batch_size, return_images=False, return_point_clouds=False):
        """
        随机采样 batch_size 条经验数据，用于训练。
        可选返回图像或点云数据。
        """
        max_size = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_size, size=batch_size)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)
        result = [obses, actions, rewards, next_obses, not_dones, not_dones_no_max]
        if return_images and self.store_image:
            imgs = torch.as_tensor(self.images[idxs], device=self.device).float() / 255.0
            result.append(imgs)
        if return_point_clouds and self.store_point_cloud:
            pcs = torch.as_tensor(self.point_clouds[idxs], device=self.device).float()
            result.append(pcs)
        return tuple(result)

    def sample_state_ent(self, batch_size, return_images=False, return_point_clouds=False):
        """
        用于采样状态熵（或其他探测目的），同时返回全部存储的观测 full_obs。
        """
        max_size = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_size, size=batch_size)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)
        full_obs = self.obses if self.full else self.obses[:self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device).float()
        result = [obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max]
        if return_images and self.store_image:
            imgs = torch.as_tensor(self.images[idxs], device=self.device).float() / 255.0
            result.append(imgs)
        if return_point_clouds and self.store_point_cloud:
            pcs = torch.as_tensor(self.point_clouds[idxs], device=self.device).float()
            result.append(pcs)
        return tuple(result)
