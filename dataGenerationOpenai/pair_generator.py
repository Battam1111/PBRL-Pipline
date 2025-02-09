#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pair_generator.py

本模块负责基于加载的样本数据生成样本对（配对），支持以下策略：
  1. 跨 bin 配对（不同 bin 之间）
  2. 邻近 bin 配对（bin 差值为 1）
  3. 同 bin 内基于 embedding 距离（选距离最小和最大的 topN 对）
  4. 同 bin 随机采样配对

同时支持去重、限制单个样本在所有配对中出现次数等功能。
"""

import random
import math
from typing import List, Tuple, Dict
import numpy as np
from utils import log

def embedding_distance(embA: List[float], embB: List[float]) -> float:
    """
    计算两个 embedding 向量的欧氏距离
    """
    arrA = np.array(embA, dtype=np.float32)
    arrB = np.array(embB, dtype=np.float32)
    return float(np.linalg.norm(arrA - arrB))

class PairGenerator:
    def __init__(self, samples_dict: Dict[int, dict]):
        """
        初始化 PairGenerator 实例
        :param samples_dict: 样本字典，键为 sample_id，值为样本信息（包含 bin_id、embedding 等）
        """
        self.samples_dict = samples_dict
        self.bin_map = self._group_by_bin()

    def _group_by_bin(self) -> Dict[int, List[int]]:
        """
        将样本根据 bin_id 进行分组
        """
        bin_map: Dict[int, List[int]] = {}
        for sid, info in self.samples_dict.items():
            b = info.get("bin_id", -1)
            bin_map.setdefault(b, []).append(sid)
        return bin_map

    def generate_pairs(self,
                       max_pairs: int = None,
                       random_seed: int = 42,
                       usage_limit: int = 50,
                       topN_intra_bin: int = 5,
                       allow_duplicates: bool = False,
                       preserve_strategy: bool = False,
                       strict_deduplicate: bool = True) -> List[Tuple[int, int, str]]:
        """
        综合多种策略生成样本对。
        :param max_pairs: 最多生成多少对；若为 None 则不限制
        :param random_seed: 随机种子
        :param usage_limit: 每个样本在所有配对中最多出现次数
        :param topN_intra_bin: 同 bin 内基于 embedding 距离时，每种（最小/最大）选取几对
        :param allow_duplicates: 是否允许同一对重复出现
        :param preserve_strategy: 是否保留相同样本对不同策略标签（若 True，则同一对不同标签均保留）
        :param strict_deduplicate: 若为 True，则完全按 (min(a,b), max(a,b)) 去重，忽略策略标签
        :return: 列表，每项为 (sampleA, sampleB, strategyTag)
        """
        random.seed(random_seed)
        candidate_pairs = []

        # 策略1：跨 bin 配对（保留一定比例）
        cross_pairs = self._generate_cross_bin_pairs(ratio=0.4)
        candidate_pairs.extend(cross_pairs)

        # 策略2：邻近 bin 配对
        neighbor_pairs = self._generate_neighbor_bin_pairs(ratio=0.3)
        candidate_pairs.extend(neighbor_pairs)

        # 策略3：同 bin 内基于 embedding 距离（选择最小与最大）
        intra_embed_pairs = self._generate_intra_bin_pairs_by_embedding(topN=topN_intra_bin)
        candidate_pairs.extend(intra_embed_pairs)

        # 策略4：同 bin 随机采样配对
        intra_random_pairs = self._generate_random_intra_bin_pairs(ratio=0.2)
        candidate_pairs.extend(intra_random_pairs)

        if not candidate_pairs:
            log("[PairGenerator] 未生成任何候选配对。")
            return []

        # 去重处理
        if not allow_duplicates:
            if strict_deduplicate:
                unique_pairs = self._deduplicate_pairs(candidate_pairs)
            else:
                if preserve_strategy:
                    unique_pairs = self._deduplicate_pairs_with_strategy(candidate_pairs)
                else:
                    unique_pairs = self._deduplicate_pairs(candidate_pairs)
        else:
            unique_pairs = candidate_pairs

        # 限制单个样本出现次数
        limited_pairs = self._apply_usage_limit(unique_pairs, usage_limit)

        # 若指定 max_pairs，则截断或扩充（允许复制）
        if max_pairs is not None:
            if len(limited_pairs) > max_pairs:
                random.shuffle(limited_pairs)
                limited_pairs = limited_pairs[:max_pairs]
            else:
                if allow_duplicates and len(limited_pairs) < max_pairs:
                    limited_pairs = self._duplicate_fill(limited_pairs, needed=max_pairs)
                    random.shuffle(limited_pairs)
                    limited_pairs = limited_pairs[:max_pairs]

        log(f"[PairGenerator] 最终生成样本对数：{len(limited_pairs)}")
        return limited_pairs

    def _generate_cross_bin_pairs(self, ratio: float = 0.4) -> List[Tuple[int, int, str]]:
        """生成所有不同 bin 间的样本对，并随机保留一定比例"""
        pairs = []
        bin_ids = sorted(self.bin_map.keys())
        if len(bin_ids) < 2:
            return pairs
        for i in range(len(bin_ids)):
            for j in range(i + 1, len(bin_ids)):
                binA, binB = bin_ids[i], bin_ids[j]
                listA = self.bin_map.get(binA, [])
                listB = self.bin_map.get(binB, [])
                if not listA or not listB:
                    continue
                current_pairs = [(a, b, f"cross_bin_{binA}_{binB}") for a in listA for b in listB]
                random.shuffle(current_pairs)
                keep_num = int(math.ceil(len(current_pairs) * ratio))
                pairs.extend(current_pairs[:keep_num])
        return pairs

    def _generate_neighbor_bin_pairs(self, ratio: float = 0.3) -> List[Tuple[int, int, str]]:
        """生成数值相邻（差值为 1）的 bin 间样本对"""
        pairs = []
        bin_ids = sorted(self.bin_map.keys())
        for i in range(len(bin_ids)):
            for j in range(i + 1, len(bin_ids)):
                binA, binB = bin_ids[i], bin_ids[j]
                if abs(binA - binB) == 1:
                    listA = self.bin_map.get(binA, [])
                    listB = self.bin_map.get(binB, [])
                    if not listA or not listB:
                        continue
                    current_pairs = [(a, b, f"neighbor_bin_{binA}_{binB}") for a in listA for b in listB]
                    random.shuffle(current_pairs)
                    keep_num = int(math.ceil(len(current_pairs) * ratio))
                    pairs.extend(current_pairs[:keep_num])
        return pairs

    def _generate_intra_bin_pairs_by_embedding(self, topN: int = 5) -> List[Tuple[int, int, str]]:
        """在同一 bin 内，根据 embedding 欧氏距离选取最小和最大的配对各 topN 对"""
        pairs = []
        for bin_id, sids in self.bin_map.items():
            if len(sids) < 2:
                continue
            emb_list = [(sid, self.samples_dict[sid].get("embedding", [])) for sid in sids]
            emb_list = [x for x in emb_list if x[1]]
            if len(emb_list) < 2:
                continue
            dist_pairs = []
            for i in range(len(emb_list)):
                for j in range(i + 1, len(emb_list)):
                    sidA, embA = emb_list[i]
                    sidB, embB = emb_list[j]
                    dist = embedding_distance(embA, embB)
                    dist_pairs.append((sidA, sidB, dist))
            if not dist_pairs:
                continue
            dist_pairs.sort(key=lambda x: x[2])
            n = min(topN, len(dist_pairs))
            top_small = dist_pairs[:n]
            top_large = dist_pairs[-n:]
            pairs.extend([(sa, sb, f"intra_bin_{bin_id}_small") for (sa, sb, _) in top_small])
            pairs.extend([(sa, sb, f"intra_bin_{bin_id}_large") for (sa, sb, _) in top_large])
        return pairs

    def _generate_random_intra_bin_pairs(self, ratio: float = 0.2) -> List[Tuple[int, int, str]]:
        """在同一 bin 内，随机采样一定比例的样本对"""
        pairs = []
        for bin_id, sids in self.bin_map.items():
            if len(sids) < 2:
                continue
            all_pairs = []
            for i in range(len(sids)):
                for j in range(i + 1, len(sids)):
                    all_pairs.append((sids[i], sids[j], f"intra_bin_random_{bin_id}"))
            random.shuffle(all_pairs)
            keep_num = int(math.ceil(len(all_pairs) * ratio))
            pairs.extend(all_pairs[:keep_num])
        return pairs

    def _deduplicate_pairs(self, pairs: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """根据 (min(a,b), max(a,b)) 去重，不考虑策略标签"""
        seen = set()
        unique = []
        for a, b, tag in pairs:
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                unique.append((a, b, tag))
        return unique

    def _deduplicate_pairs_with_strategy(self, pairs: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """根据 (min(a,b), max(a,b), tag) 去重，保留同一对但策略标签不同的情况"""
        seen = set()
        unique = []
        for a, b, tag in pairs:
            key = (min(a, b), max(a, b), tag)
            if key not in seen:
                seen.add(key)
                unique.append((a, b, tag))
        return unique

    def _apply_usage_limit(self, pairs: List[Tuple[int, int, str]], usage_limit: int) -> List[Tuple[int, int, str]]:
        """限制每个样本在所有配对中出现的最大次数"""
        usage_count = {}
        limited = []
        for a, b, tag in pairs:
            if usage_count.get(a, 0) < usage_limit and usage_count.get(b, 0) < usage_limit:
                limited.append((a, b, tag))
                usage_count[a] = usage_count.get(a, 0) + 1
                usage_count[b] = usage_count.get(b, 0) + 1
        return limited

    def _duplicate_fill(self, pairs: List[Tuple[int, int, str]], needed: int) -> List[Tuple[int, int, str]]:
        """当生成的配对数量不足时，允许重复复制填充到指定数量"""
        result = pairs.copy()
        if not result:
            return result
        i = 0
        dupe_id = 1
        while len(result) < needed:
            a, b, tag = result[i % len(result)]
            new_tag = f"{tag}(DUPE_{dupe_id})"
            result.append((a, b, new_tag))
            i += 1
            dupe_id += 1
        return result
