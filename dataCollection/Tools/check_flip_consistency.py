#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_flip_consistency.py

本脚本用于对比「翻转前(Baseline)」与「翻转后(Flipped)」两份合并结果文件的数据，
并且在决策(0/1/-1)一致性判断中应用【翻转逻辑】：
- 在翻转情况下，若B文件中decision='0'，则F文件中需='1'才视为一致(反之亦然)，
- 若B文件中decision='-1'，则F文件也需='-1'才算一致。
这样可以体现「Situation顺序被对调后」的大模型回答是否符合「同一结论」。

假设：
  - B文件(翻转前)路径 => dataCollection/Dataset/metaworld_soccer-v2_merged_B.json
  - F文件(翻转后)路径 => dataCollection/Dataset/metaworld_soccer-v2_merged_F.json

主要流程：
1) 读取 B 文件 => b_map: { custom_id -> {decision, analysis, cloud_info} }
2) 读取 F 文件 => f_map: { custom_id -> {decision, analysis, cloud_info} }
3) 对公共 custom_id，应用 “翻转一致性规则” 判断决策是否一致
   (0 <-> 1, -1 <-> -1)
4) 输出统计结果(见 report_results)

使用方法：
   python check_flip_consistency.py

你可在 main() 中自行修改 B_FILE 与 F_FILE 的路径。
"""

import os
import json

def load_merged_json(file_path: str):
    """
    从指定的 merged JSON 文件中读取数据, 返回:
       { custom_id -> { "decision":"...", "analysis":"...", "cloud_info":[...] } }

    其中 "cloud_info" 可做更深入检查, 本脚本主要关注 "decision" & "analysis".
    """
    if not os.path.isfile(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"[错误] 无法解析 JSON: {e}")
            return {}

    # data 应该是一个 list, 每个元素形如:
    # {
    #   "custom_id":"...",
    #   "cloud_info": [...],
    #   "analysis":"(文本)",
    #   "decision":"0"/"1"/"-1"
    # }
    # 我们构建一个 map => cid->(dic)
    result_map = {}
    line_count = 0
    if isinstance(data, list):
        for item in data:
            cid = item.get("custom_id","")
            if cid:
                line_count += 1
                dec = item.get("decision","")
                ana = item.get("analysis","")
                cinfo= item.get("cloud_info", [])
                result_map[cid] = {
                    "decision": dec,
                    "analysis": ana,
                    "cloud_info": cinfo
                }

    print(f"[load_merged_json] 从 {file_path} 读取成功, 共 {line_count} 条记录.")
    return result_map


def flip_decision_equivalent(decB: str, decF: str) -> bool:
    """
    在「翻转场景」下，判断 baseline决策 decB 与 flipped决策 decF 是否表示同一个结论.

    规则:
     - 若 B='0' => F='1' 才算一致 (因为原先S1更好, 翻转后S1成为旧S2)
     - 若 B='1' => F='0' 才算一致
     - 若 B='-1' => F='-1' 才算一致
     - 其他情况 => 不一致

    注: decB/decF 都可为空字符串或不在['0','1','-1']之列, 则直接返回False.
    """
    # 标准集合
    valid_set = {'0','1','-1'}
    if (decB not in valid_set) or (decF not in valid_set):
        return False

    if decB == '-1' and decF == '-1':
        return True
    if decB == '0' and decF == '1':
        return True
    if decB == '1' and decF == '0':
        return True

    return False


def compare_two_maps(b_map: dict, f_map: dict):
    """
    对比两个映射(b_map vs f_map), 并使用 "flip_decision_equivalent" 进行决策一致性判断.

    每个映射: { cid -> {"decision":"...","analysis":"...","cloud_info":[...]} }.

    返回:
      - count_B, count_F
      - missing_in_B, missing_in_F
      - common_cids
      - decision_match_count
      - decision_diff_count
      - decision_diffs: [(cid, decB, decF)]
      - analysis_same_count
      - analysis_diff_count
      - analysis_diff_cids
    """
    b_cids = set(b_map.keys())
    f_cids = set(f_map.keys())

    missing_in_B = sorted(list(f_cids - b_cids))  # 只在F中出现
    missing_in_F = sorted(list(b_cids - f_cids))  # 只在B中出现
    common_cids  = sorted(list(b_cids & f_cids))

    decision_match_count = 0
    decision_diff_count  = 0
    decision_diffs       = []

    analysis_same_count=0
    analysis_diffs=[]

    for cid in common_cids:
        decB = b_map[cid]["decision"]
        decF = f_map[cid]["decision"]

        # 用自定义规则判断翻转一致性
        if flip_decision_equivalent(decB, decF):
            decision_match_count += 1
        else:
            decision_diff_count += 1
            decision_diffs.append((cid, decB, decF))

        # 分析文本是否完全相同(可选)
        anaB = b_map[cid]["analysis"].strip()
        anaF = f_map[cid]["analysis"].strip()
        if anaB == anaF:
            analysis_same_count += 1
        else:
            analysis_diffs.append(cid)

    results = {
        "count_B": len(b_map),
        "count_F": len(f_map),
        "missing_in_B": missing_in_B,
        "missing_in_F": missing_in_F,
        "common_cids": common_cids,
        "decision_match_count": decision_match_count,
        "decision_diff_count":  decision_diff_count,
        "decision_diffs":       decision_diffs,
        "analysis_same_count":  analysis_same_count,
        "analysis_diff_count":  len(common_cids) - analysis_same_count,
        "analysis_diff_cids":   analysis_diffs
    }
    return results

def report_results(results: dict):
    """
    根据 compare_two_maps() 的结果字典, 输出可读的对比报告.

    包括:
      - B, F 文件各自记录数
      - missing_in_B, missing_in_F
      - 公共 custom_id 数
      - decision一致/不一致数量及具体列表
      - analysis完全相同/不同计数及列表
    """
    print("\n==== [翻转一致性检查] 对比报告 ====\n")
    cB = results["count_B"]
    cF = results["count_F"]
    print(f"Baseline文件记录数: {cB}, Flipped文件记录数: {cF}")

    missingB = results["missing_in_B"]
    missingF = results["missing_in_F"]
    print(f"\n仅在F中出现(不在B中)的 custom_id 数: {len(missingB)}")
    if missingB:
        print("  =>", missingB)

    print(f"\n仅在B中出现(不在F中)的 custom_id 数: {len(missingF)}")
    if missingF:
        print("  =>", missingF)

    common_cids = results["common_cids"]
    print(f"\n公共 custom_id 数: {len(common_cids)}")

    dmatch = results["decision_match_count"]
    ddiff  = results["decision_diff_count"]
    print(f"\n决策(Decision)翻转一致数: {dmatch}, 不一致数: {ddiff}")
    # if ddiff>0:
    #     print("以下 custom_id 的决策不一致(翻转规则下):")
    #     for (cid, decB, decF) in results["decision_diffs"]:
    #         print(f"  - {cid}: B={decB}, F={decF}")

    # asame = results["analysis_same_count"]
    # adiff= results["analysis_diff_count"]
    # print(f"\n回答文本(analysis)完全一致数: {asame}, 不一致数: {adiff}")
    # if adiff>0:
    #     print("以下 custom_id 的回答文本有差异:")
    #     diff_list = results["analysis_diff_cids"]
    #     for cid in diff_list:
    #         print(f"  - {cid}")

    print("\n==== [翻转一致性检查] 报告结束 ====\n")

def main():
    """
    主函数:
      1) 指定 Baseline 文件 (B_FILE)
      2) 指定 Flipped 文件 (F_FILE)
      3) 读取 => map
      4) compare => 生成结果
      5) report => 打印报告

    你可在此调整 B_FILE, F_FILE 路径, 实现对其他环境的检查.
    """

    # B_FILE = "dataCollection/Dataset/metaworld_soccer-v2_merged_B.json"
    # F_FILE = "dataCollection/Dataset/metaworld_soccer-v2_merged_F.json"

    # B_FILE = "dataCollection/Dataset/metaworld_drawer-open-v2_merged_B.json"
    # F_FILE = "dataCollection/Dataset/metaworld_drawer-open-v2_merged_F.json"

    # B_FILE = "dataCollection/Dataset/metaworld_door-open-v2_merged_B.json"
    # F_FILE = "dataCollection/Dataset/metaworld_door-open-v2_merged_F.json"

    B_FILE = "dataCollection/Dataset/metaworld_disassemble-v2_merged_B.json"
    F_FILE = "dataCollection/Dataset/metaworld_disassemble-v2_merged_F.json"

    # 1) 加载
    b_map = load_merged_json(B_FILE)
    f_map = load_merged_json(F_FILE)

    # 2) 对比(翻转逻辑)
    results = compare_two_maps(b_map, f_map)

    # 3) 输出报告
    report_results(results)

if __name__=="__main__":
    main()
