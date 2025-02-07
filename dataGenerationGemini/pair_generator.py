#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pair_generator.py

本模块负责生成样本对，支持以下策略：
  1. 跨 bin 配对：随机配对不同 bin 中的样本。
  2. 邻近 bin 配对：对相邻 bin 中的样本进行随机配对。
  3. 同一 bin 内基于 embedding 距离配对：选择 embedding 距离最小和最大的若干对。

同时，对生成的样本对进行去重，并限制每个样本出现次数，确保任务数据多样且均衡。
"""

import random
import math
from typing import List, Tuple, Dict
import numpy as np

def embedding_distance(embA: List[float], embB: List[float]) -> float:
    """
    计算两个 embedding 向量的欧氏距离。

    参数：
      embA, embB: 两个 embedding 向量（列表）。
    返回：
      欧氏距离（浮点数）。
    """
    arrA = np.array(embA, dtype=np.float32)
    arrB = np.array(embB, dtype=np.float32)
    diff = arrA - arrB
    return float(np.sqrt(np.dot(diff, diff)))

class PairGenerator:
    def __init__(self, samples_dict: Dict[int, dict]):
        """
        初始化 PairGenerator 实例。

        参数：
          samples_dict: 样本数据字典，键为 sample_id，值为样本信息。
        """
        self.samples_dict = samples_dict

    def generate_pairs(self,
                       max_pairs: int = None,
                       random_seed: int = 42,
                       usage_limit: int = 50,
                       topN_intra_bin: int = 5) -> List[Tuple[int, int, str]]:
        """
        生成样本对列表，采用多种配对策略混合生成。

        参数：
          max_pairs: 最终生成的样本对数量上限。
          random_seed: 随机种子，用于结果复现。
          usage_limit: 每个样本在所有对中最多出现次数。
          topN_intra_bin: 同一 bin 内选择的 topN 对数。
        返回：
          样本对列表，每个元素为 (sampleA, sampleB, 标签)。
        """
        random.seed(random_seed)
        bin_map: Dict[int, List[int]] = {}
        for sid, info in self.samples_dict.items():
            b = info.get("bin_id", -1)
            bin_map.setdefault(b, []).append(sid)
        if not bin_map:
            print("[PairGenerator] 未能构建 bin_map，返回空列表。")
            return []

        all_bins = sorted(bin_map.keys())
        raw_pairs = []

        # 策略1：跨 bin 配对（最小 bin 与最大 bin）
        if len(all_bins) >= 2:
            bmin, bmax = all_bins[0], all_bins[-1]
            raw_pairs.extend(self._generate_bin_pairs_random(bin_map, bmin, bmax, "cross_bin_minmax", ratio=0.4))

        # 策略2：邻近 bin 配对
        if len(all_bins) > 1:
            for i in range(len(all_bins) - 1):
                binA, binB = all_bins[i], all_bins[i+1]
                raw_pairs.extend(self._generate_bin_pairs_random(bin_map, binA, binB, f"neighbor_bin_{binA}_{binB}", ratio=0.3))

        # 策略3：同一 bin 内基于 embedding 距离配对
        for b_id in all_bins:
            sids_in_bin = bin_map[b_id]
            if len(sids_in_bin) < 2:
                continue
            raw_pairs.extend(self._generate_intra_bin_pairs_by_embedding(sids_in_bin, topN_intra_bin, b_id))

        if not raw_pairs:
            print("[PairGenerator] 未生成任何样本对。")
            return []

        final_pairs = self._apply_usage_and_dedup(raw_pairs, usage_limit)

        if max_pairs is not None:
            if len(final_pairs) > max_pairs:
                random.shuffle(final_pairs)
                final_pairs = final_pairs[:max_pairs]
            elif len(final_pairs) < max_pairs:
                final_pairs = self._duplicate_fill(final_pairs, needed=max_pairs)
                if len(final_pairs) > max_pairs:
                    random.shuffle(final_pairs)
                    final_pairs = final_pairs[:max_pairs]
        return final_pairs

    def _generate_bin_pairs_random(self,
                                   bin_map: Dict[int, List[int]],
                                   binA: int,
                                   binB: int,
                                   pair_tag: str,
                                   ratio: float) -> List[Tuple[int, int, str]]:
        listA = bin_map.get(binA, [])
        listB = bin_map.get(binB, [])
        if not listA or not listB:
            return []
        pairs = [(a, b, pair_tag) for a in listA for b in listB if a != b]
        random.shuffle(pairs)
        keep_num = int(math.ceil(len(pairs) * ratio))
        return pairs[:keep_num]

    def _generate_intra_bin_pairs_by_embedding(self, sids_in_bin: List[int], topN: int, bin_id: int) -> List[Tuple[int, int, str]]:
        emb_list = [(sid, self.samples_dict[sid].get("embedding", []))
                    for sid in sids_in_bin if self.samples_dict[sid].get("embedding")]
        if len(emb_list) < 2:
            return []
        dist_pairs = []
        for i in range(len(emb_list)):
            for j in range(i + 1, len(emb_list)):
                sidA, embA = emb_list[i]
                sidB, embB = emb_list[j]
                dist = embedding_distance(embA, embB)
                dist_pairs.append((sidA, sidB, dist))
        if not dist_pairs:
            return []
        dist_pairs.sort(key=lambda x: x[2])
        top_small = dist_pairs[:min(topN, len(dist_pairs))]
        top_large = dist_pairs[-min(topN, len(dist_pairs)):]
        results = [(sa, sb, f"intra_bin_{bin_id}_small") for (sa, sb, _) in top_small]
        results += [(sa, sb, f"intra_bin_{bin_id}_large") for (sa, sb, _) in top_large]
        return results

    def _apply_usage_and_dedup(self, raw_pairs: List[Tuple[int, int, str]], usage_limit: int) -> List[Tuple[int, int, str]]:
        seen = set()
        deduped = []
        for (a, b, tag) in raw_pairs:
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                deduped.append((a, b, tag))
        usage_count = {}
        final_pairs = []
        for (a, b, tag) in deduped:
            if usage_count.get(a, 0) < usage_limit and usage_count.get(b, 0) < usage_limit:
                final_pairs.append((a, b, tag))
                usage_count[a] = usage_count.get(a, 0) + 1
                usage_count[b] = usage_count.get(b, 0) + 1
        return final_pairs

    def _duplicate_fill(self, pairs: List[Tuple[int, int, str]], needed: int) -> List[Tuple[int, int, str]]:
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
