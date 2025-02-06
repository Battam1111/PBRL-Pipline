# -*- coding: utf-8 -*-
"""
工具模块：包含多因素距离计算和 MD5 哈希计算等通用函数

核心功能：
  1. multi_factor_distance：计算样本间的多因素距离，综合 embedding 差异、reward 差异以及时间差异
  2. md5_hash_for_array：对 numpy 数组计算 MD5 哈希，并截取前 8 位，用于文件名中的简短标识
"""

import hashlib
import numpy as np

def multi_factor_distance(
    emb_new: np.ndarray,
    emb_old: np.ndarray,
    reward_new: float,
    reward_old: float,
    time_new: float,
    time_old: float,
    alpha: float = 1.0,
    beta: float  = 0.0,
    gamma: float = 0.0
) -> float:
    """
    计算多因素距离，将 embedding 差、reward 差以及 time 差综合计算得到距离：
    
        distance = sqrt( alpha * ||emb_new - emb_old||^2 +
                         beta * (reward_new - reward_old)^2 +
                         gamma * (time_new - time_old)^2 )
    
    参数说明：
      - emb_new, emb_old: 样本的 embedding 向量（numpy 数组）
      - reward_new, reward_old: 样本对应的 reward 数值
      - time_new, time_old: 样本插入时的时间（或顺序值）
      - alpha, beta, gamma: 分别对应 embedding、reward、time 在距离计算中的权重。
    
    默认 alpha=1.0, beta=0.0, gamma=0.0，即只考虑 embedding 差异。
    
    返回：
      综合计算后的距离（浮点数）
    """
    diff_emb = emb_new - emb_old
    # 计算 embedding 的欧氏距离
    dist_emb = np.linalg.norm(diff_emb)
    
    r_diff = reward_new - reward_old
    t_diff = time_new - time_old
    
    value = alpha * (dist_emb ** 2) + beta * (r_diff ** 2) + gamma * (t_diff ** 2)
    return np.sqrt(value)


def md5_hash_for_array(arr: np.ndarray) -> str:
    """
    对给定的 numpy 数组计算 MD5 哈希，并截取前 8 位 hex 值。
    用于生成文件名中的简短标识。如果 arr 为 None，则返回 "nohash"。
    
    参数：
      - arr: numpy 数组
    
    返回：
      前 8 位 MD5 哈希字符串
    """
    if arr is None:
        return "nohash"
    raw_bytes = arr.flatten().tobytes()
    md5_val = hashlib.md5(raw_bytes).hexdigest()
    return md5_val[:8]
