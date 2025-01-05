import numpy as np
import torch
import utils

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
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
            obs_shape: 观测空间的形状 (例如 (11,) )
            action_shape: 动作空间的形状 (例如 (4,) )
            capacity: 缓冲区大小
            device: PyTorch 设备 (cuda/cpu)
            window: 暂时保留（没有实际作用）
            store_image: 是否存储图像
            image_size: 如果存储图像，则图像的高宽
            store_point_cloud: 是否存储点云
            point_cloud_size: 点云数量 (N)，默认为 8192
        """
        self.capacity = capacity
        self.device = device

        # 若 obs 只有一维(典型就是状态向量)，则用 float32；否则(典型是图像)用 uint8
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

        # 如果需要存图像，则初始化一个 (capacity, H, W, 3) 的缓冲
        self.images = np.empty((capacity, image_size, image_size, 3), dtype=np.uint8)

        # 如果需要存点云，则初始化 (capacity, point_cloud_size, 6) 的缓冲
        # 其中每个点有 (x, y, z, r, g, b) 六个分量
        if self.store_point_cloud:
            self.point_clouds = np.empty((capacity, point_cloud_size, 6), dtype=np.float32)

        self.idx = 0          # 缓冲区当前插入位置
        self.full = False     # 标志位：缓冲区是否已填满

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, image=None, point_cloud=None):
        """
        向缓冲区中添加一条经验 (obs, action, reward, next_obs, done...)。
        如果 store_image 为 True 并且传入了 image，就存储该图像；同理对点云也如此。
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
            # 将 point_cloud 存进来
            # point_cloud 形状应为 (N, 6)，与初始化的 (capacity, N, 6) 对应
            point_cloud = point_cloud.astype(np.float32)
            np.copyto(self.point_clouds[self.idx], point_cloud)

        # 环形索引
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or (self.idx == 0)

    def relabel_with_predictor(self, predictor):
        """
        使用给定的奖励模型 predictor，重新为回放缓冲区中存储的 (obs, action) 或图像 
        估计奖励并赋值给 self.rewards。
        
        注意：当 store_image = True 时，用图像计算 reward；
              否则用 obs+action 计算 reward；
              如果只是存了 point_cloud 并没开启 store_image，则仍用 obs+action。
              point_cloud 在此处仅用于打标签，不参加 reward 估计。
        """
        # 按固定 batch_size 分批处理
        if self.store_image:
            batch_size = 32
        else:
            # 若没有图像模式，则使用较大 batch_size
            batch_size = 200
        
        total_samples = self.idx if not self.full else self.capacity
        total_iter = (total_samples + batch_size - 1) // batch_size  # ceil 整除

        for i in range(total_iter):
            start = i * batch_size
            end = min(start + batch_size, total_samples)

            # 构造输入
            if self.store_image:
                # 如果存了图像，就用图像作为输入
                imgs = self.images[start:end]
                imgs = np.transpose(imgs, (0, 3, 1, 2))  # (B, H, W, 3) -> (B, 3, H, W)
                imgs = imgs.astype(np.float32) / 255.0
                inputs = imgs
            else:
                # 否则用 (obs+action) 作为输入
                obses = self.obses[start:end]
                acts = self.actions[start:end]
                inputs = np.concatenate([obses, acts], axis=-1)  # (B, obs_dim + act_dim)

            # 调用 predictor
            pred_reward = predictor.r_hat_batch(inputs)
            pred_reward = pred_reward.reshape(-1, 1)

            self.rewards[start:end] = pred_reward

        # 清理显存缓存
        torch.cuda.empty_cache()

    def sample(self, batch_size, return_images=False, return_point_clouds=False):
        """
        随机取 batch_size 条数据用于训练。
        可选地返回图像或点云。
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
            # 返回图像 (B, H, W, 3)，在使用时可以再做 transpose
            imgs = torch.as_tensor(self.images[idxs], device=self.device).float() / 255.0
            result.append(imgs)

        if return_point_clouds and self.store_point_cloud:
            pcs = torch.as_tensor(self.point_clouds[idxs], device=self.device).float()
            result.append(pcs)

        return tuple(result)

    def sample_state_ent(self, batch_size, return_images=False, return_point_clouds=False):
        """
        sample_state_ent 用于做状态熵或类似探测时会用到。
        多返回一个 full_obs 。
        """
        max_size = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_size, size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        # 取出所有已存 obs 组成 full_obs
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device).float()

        result = [obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max]

        if return_images and self.store_image:
            imgs = torch.as_tensor(self.images[idxs], device=self.device).float() / 255.0
            result.append(imgs)

        if return_point_clouds and self.store_point_cloud:
            pcs = torch.as_tensor(self.point_clouds[idxs], device=self.device).float()
            result.append(pcs)

        return tuple(result)
