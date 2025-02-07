#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transform_api_data.py

本脚本用于将「生成的输入文件」(batch_input_all.jsonl) 与「模型返回的输出文件」
(例如 gemini_results.jsonl 或 batch_output_merged.jsonl) 合并，得到一个包含全部信息的
JSON 数组，便于后续数据分析与应用。

主要步骤：
1) 解析输入文件 (parse_input_file)：
   - 逐行读取 JSON 数据，提取 custom_id 及 body.contents 中用户输入的文本内容。
   - 用户文本包含两个部分：Situation 1 与 Situation 2，对应多个图像 URL。
   - 对每个 URL，取其文件名后调用 parse_filename()，利用正则表达式解析文件名中的关键信息：
         idx, bin, r, t, emb, view
     注意：经过图像拼接后生成的文件，其文件名不再包含 "_view…" 部分，本正则表达式支持可选的 view 字段。
   - 将解析结果保存为字典，结构为：
         { custom_id -> [ {situation, idx, bin, r, t, emb, view, url}, ... ] }

2) 解析输出文件 (parse_output_file)：
   - 逐行读取输出 JSONL 文件，提取 custom_id 以及模型返回回答文本。
   - 考虑到目前返回格式中，模型回答文本直接存于 response.text 字段，
     同时为了兼容旧格式（可能存在 response.body.choices），本函数会自动判断。
   - 调用 parse_decision() 从回答文本中解析出最终决策： '0' / '1' / '-1'。
   - 最终结果保存为字典：
         { custom_id -> { "analysis": 模型回答文本, "decision": '0'/'1'/'-1' } }

3) 合并并保存 (merge_and_save)：
   - 以 custom_id 为键，对输入与输出数据进行对齐合并；若某侧数据缺失，则相应字段为空字符串。
   - 生成一个包含所有记录的列表，每个记录格式为：
         {
           "custom_id": "xxx",
           "cloud_info": [
              { "situation": 1, "idx": ..., "bin": ..., "r": ..., "t": ..., "emb": "...", "view": "...", "url": "..." },
              { "situation": 2, ... },
              ...
           ],
           "analysis": "模型回答原文",
           "decision": "0/1/-1"
         }
   - 将最终结果写入目标输出文件 merged_file 中，并按 custom_id 进行排序。

使用方式：
   python transform_api_data.py

请根据实际情况在 main() 函数中修改以下路径：
   - input_file_in  => 指向 batch_input_all.jsonl（输入任务文件）
   - input_file_out => 指向模型返回的输出文件（如 gemini_results.jsonl）
   - merged_file    => 合并后输出文件的路径
"""

import os
import re
import json
import sys

# =============================================================================
# 正则表达式定义：
# 支持两种文件名格式：
# 1) 原始视角图像文件名：例如
#    pc_000253_bin_2_r_4.61_t_253.00_emb_add61616_view2.jpg
# 2) 拼接后图像文件名：例如
#    pc_000253_bin_2_r_4.61_t_253.00_emb_add61616.jpg
# view 部分为可选，使用 (?:_view(\w+))? 进行匹配。
# =============================================================================
FILENAME_REGEX = r"^pc_(\d+)_bin_(\d+)_r_([\d.]+)_t_([\d.]+)_emb_([a-zA-Z0-9]+)(?:_view(\w+))?\.jpg$"

def parse_decision(content: str) -> str:
    """
    从模型回答文本 content 中提取最终决策结果： '0', '1', 或 '-1'。
    
    考虑到回答文本可能采用加粗、列表、换行或空格分隔等多种格式，本函数尝试以下匹配策略：
      1. 匹配 Markdown 加粗格式，如 **0**、**1**、**-1**
      2. 匹配列表格式，如 - '0'、- '1'、- '-1'
      3. 匹配换行后直接给出的数字，如 "\n0", "\n1", "\n-1"
      4. 匹配空格分隔，如 " 0", " 1", " -1"
      
    如果均未匹配到，则返回空字符串 ""。

    :param content: 模型返回的回答文本
    :return: 提取到的决策结果，若无法确定则返回 ""
    """
    lower_text = content.lower()

    # 优先匹配 Markdown 加粗格式
    if '**0**' in lower_text and '**1**' not in lower_text and '**-1**' not in lower_text:
        return '0'
    if '**1**' in lower_text and '**-1**' not in lower_text and '**0**' not in lower_text:
        return '1'
    if '**-1**' in lower_text:
        return '-1'

    # 匹配列表格式
    if "- '0'" in lower_text:
        return '0'
    if "- '1'" in lower_text:
        return '1'
    if "- '-1'" in lower_text:
        return '-1'

    # 匹配换行后直接给出的数字
    if "\n0" in lower_text:
        return '0'
    if "\n1" in lower_text:
        return '1'
    if "\n-1" in lower_text:
        return '-1'

    # 尝试空格分隔查找
    if " 0" in lower_text:
        return '0'
    if " 1" in lower_text:
        return '1'
    if " -1" in lower_text:
        return '-1'

    return ""

def parse_filename(filename: str) -> dict:
    """
    给定一个文件名（不含路径），例如：
      pc_000253_bin_2_r_4.61_t_253.00_emb_add61616_view2.jpg
    或经过拼接后生成的文件名：
      pc_000253_bin_2_r_4.61_t_253.00_emb_add61616.jpg
    利用正则表达式 FILENAME_REGEX 提取关键信息，包括：
       idx, bin, r, t, emb, view
    其中 view 为可选字段，若不存在则返回空字符串 ""。

    返回格式为字典，如：
      {
         "idx": 253,
         "bin": 2,
         "r": 4.61,
         "t": 253.00,
         "emb": "add61616",
         "view": "view2"   # 若不存在则为 ""
      }
    若匹配失败，则返回空字典 {}。

    :param filename: 文件名字符串（不含路径）
    :return: 解析后的关键信息字典
    """
    m = re.match(FILENAME_REGEX, filename)
    if not m:
        return {}
    idx_str, bin_str, r_str, t_str, emb_str, view_str = m.groups()

    try:
        idx_val = int(idx_str)
    except Exception:
        idx_val = None

    try:
        bin_val = int(bin_str)
    except Exception:
        bin_val = None

    try:
        r_val = float(r_str)
    except Exception:
        r_val = None

    try:
        t_val = float(t_str)
    except Exception:
        t_val = None

    return {
        "idx": idx_val,
        "bin": bin_val,
        "r": r_val,
        "t": t_val,
        "emb": emb_str,
        "view": view_str if view_str is not None else ""
    }

def parse_input_file(input_file_in: str) -> dict:
    """
    解析输入文件 (batch_input_all.jsonl)，逐行读取 JSON 数据，
    提取并处理每行内容，得到如下结构：
       { custom_id -> [ {situation, idx, bin, r, t, emb, view, url}, ... ] }

    具体流程：
      1. 对每行 JSON 数据，提取 custom_id 和 body.contents 中用户输入的文本内容。
         注意：新版输入文件中，用户文本存储在 body.contents 数组中，
         通常取第一个元素或唯一元素的 parts 列表中第一个部分的 text。
      2. 利用正则表达式提取 "Situation 1:" 和 "Situation 2:" 两部分的内容。
         如果 Situation 2 中包含 "Objective:" 则只取其之前的内容。
      3. 将两部分内容分别按空格拆分，得到 URL 列表。
      4. 对每个 URL，使用 os.path.basename 获取文件名，并调用 parse_filename() 解析文件名中的信息。
      5. 将解析结果构造为列表，每个元素为字典：
             { "situation": 1或2, "idx": ..., "bin": ..., "r": ..., "t": ..., "emb": ..., "view": ..., "url": ... }
         并以 custom_id 为键保存至结果字典中。

    :param input_file_in: 输入文件路径 (batch_input_all.jsonl)
    :return: 解析后的数据字典，键为 custom_id，值为关键信息列表
    """
    if not os.path.isfile(input_file_in):
        print(f"[错误] 找不到输入文件: {input_file_in}")
        return {}

    result_map = {}
    line_count = 0
    matched_count = 0

    with open(input_file_in, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[警告] 行 {line_count} JSON解析失败: {e}")
                continue

            # 获取 custom_id
            cid = obj.get("custom_id", "")
            if not cid:
                continue

            # 获取 body.contents 数组中用户输入的文本内容
            body = obj.get("body", {})
            contents = body.get("contents", [])
            if not contents:
                continue

            # 假定用户文本存放在第一个元素的 parts 列表中
            user_text = ""
            try:
                parts = contents[0].get("parts", [])
                if parts:
                    user_text = parts[0].get("text", "")
            except Exception as e:
                print(f"[警告] custom_id {cid} 提取用户文本失败: {e}")
                continue

            # 利用正则提取 "Situation 1:" 与 "Situation 2:" 部分
            # 使用 DOTALL 模式允许跨行匹配，忽略大小写
            pattern_sit = re.compile(r"Situation\s+1\s*:\s*(.*?)\s*Situation\s+2\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)
            mm = pattern_sit.search(user_text)
            if not mm:
                print(f"[警告] custom_id {cid} 未能匹配 Situation 部分")
                continue
            sit1_str = mm.group(1).strip()
            sit2_str = mm.group(2).strip()
            # 如果 Situation 2 中包含 "Objective:"，则只取其前部分
            idx_obj = sit2_str.lower().find("objective:")
            if idx_obj >= 0:
                sit2_str = sit2_str[:idx_obj].strip()

            # 将两部分内容分别按空格拆分为 URL 列表
            s1_urls = [u.strip() for u in sit1_str.split() if u.strip()]
            s2_urls = [u.strip() for u in sit2_str.split() if u.strip()]

            parsed_list = []
            # 处理 Situation 1 中的 URL
            for u in s1_urls:
                fname = os.path.basename(u)
                fdict = parse_filename(fname)
                # 如果解析失败，仍保留 URL 信息
                item = {
                    "situation": 1,
                    "idx": fdict.get("idx"),
                    "bin": fdict.get("bin"),
                    "r": fdict.get("r"),
                    "t": fdict.get("t"),
                    "emb": fdict.get("emb", ""),
                    "view": fdict.get("view", ""),
                    "url": u
                }
                parsed_list.append(item)
            # 处理 Situation 2 中的 URL
            for u in s2_urls:
                fname = os.path.basename(u)
                fdict = parse_filename(fname)
                item = {
                    "situation": 2,
                    "idx": fdict.get("idx"),
                    "bin": fdict.get("bin"),
                    "r": fdict.get("r"),
                    "t": fdict.get("t"),
                    "emb": fdict.get("emb", ""),
                    "view": fdict.get("view", ""),
                    "url": u
                }
                parsed_list.append(item)

            result_map[cid] = parsed_list
            matched_count += 1

    print(f"[INFO] 输入文件解析完成：共 {line_count} 行，其中 {matched_count} 行成功解析；总 custom_id 数量 = {len(result_map)}")
    return result_map

def parse_output_file(input_file_out: str) -> dict:
    """
    解析输出文件 (例如 gemini_results.jsonl)，逐行读取 JSON 数据，
    提取 custom_id 以及模型返回的回答文本，并利用 parse_decision() 解析出最终决策结果。

    本函数兼容以下返回格式：
      - 新版格式：模型回答文本直接存于 response.text 字段
      - 旧版格式：模型回答文本存于 response.body.choices[0].message.content 字段

    最终返回结构为：
         { custom_id -> { "analysis": 模型回答文本, "decision": "0"/"1"/"-1" } }

    :param input_file_out: 输出文件路径
    :return: 解析后的结果字典
    """
    if not os.path.isfile(input_file_out):
        print(f"[错误] 找不到输出文件: {input_file_out}")
        return {}

    result_map = {}
    line_count = 0
    valid_count = 0

    with open(input_file_out, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[警告] 输出文件第 {line_count} 行 JSON解析失败: {e}")
                continue

            cid = obj.get("custom_id", "")
            if not cid:
                continue

            # 尝试新版格式：response.text
            content = obj.get("response", {}).get("text", "")
            if not content:
                # 若新版格式未命中，则尝试旧版格式：response.body.choices[0].message.content
                response_obj = obj.get("response", {})
                body_obj = response_obj.get("body", {})
                choices = body_obj.get("choices", [])
                if choices:
                    try:
                        content = choices[0]["message"]["content"]
                    except Exception:
                        content = ""
            if not content:
                continue

            decision = parse_decision(content)
            result_map[cid] = {
                "analysis": content,
                "decision": decision
            }
            valid_count += 1

    print(f"[INFO] 输出文件解析完成：共 {line_count} 行，其中 {valid_count} 行成功解析.")
    return result_map

def merge_and_save(input_map: dict, output_map: dict, merged_file: str) -> None:
    """
    将输入数据 (input_map) 与输出数据 (output_map) 按 custom_id 进行合并，
    生成包含全部信息的记录数组。每条记录的格式如下：
      {
        "custom_id": "xxx",
        "cloud_info": [
          { "situation": 1, "idx": ..., "bin": ..., "r": ..., "t": ..., "emb": "...", "view": "...", "url": "..." },
          { "situation": 2, ... },
          ...
        ],
        "analysis": "模型回答原文",
        "decision": "0/1/-1"
      }
    最后将合并结果写入 merged_file 文件，写出的内容为格式化后的 JSON 数组，
    并按照 custom_id 进行升序排序。

    :param input_map: 输入文件解析得到的数据字典 { custom_id -> [ {...}, ... ] }
    :param output_map: 输出文件解析得到的数据字典 { custom_id -> { "analysis": ..., "decision": ... } }
    :param merged_file: 合并结果输出文件的路径
    """
    merged_list = []
    # 以输入文件中的 custom_id 为主进行合并，若输出数据中不存在则置为空字符串
    for cid, cloud_info_list in input_map.items():
        out = output_map.get(cid, {})
        analysis = out.get("analysis", "")
        decision = out.get("decision", "")
        item = {
            "custom_id": cid,
            "cloud_info": cloud_info_list,
            "analysis": analysis,
            "decision": decision
        }
        merged_list.append(item)

    # 按 custom_id 升序排序（假设 custom_id 中包含数值部分）
    merged_list.sort(key=lambda x: x["custom_id"])

    os.makedirs(os.path.dirname(merged_file), exist_ok=True)
    try:
        with open(merged_file, "w", encoding="utf-8") as outf:
            json.dump(merged_list, outf, ensure_ascii=False, indent=2)
        print(f"[完成] 合并输出文件生成：{merged_file}；共 {len(merged_list)} 条记录.")
    except Exception as e:
        print(f"[错误] 写入合并文件失败: {e}")

def main():
    """
    主函数：
      1) 指定输入文件 (batch_input_all.jsonl) 与输出文件 (模型返回的 JSONL 文件) 的路径，
         以及最终合并后输出的目标文件路径 merged_file。
      2) 调用 parse_input_file() 和 parse_output_file() 分别解析输入与输出文件。
      3) 调用 merge_and_save() 将解析结果按 custom_id 合并并写入最终文件。

    请根据实际情况修改以下路径：
         - input_file_in: 指向 batch_input_all.jsonl（输入任务文件）
         - input_file_out: 指向模型返回的输出文件（例如 gemini_results.jsonl）
         - merged_file: 合并后输出文件的路径
    """
    # 修改以下路径为实际文件路径
    input_file_in = "dataCollection/Dataset/metaworld_soccer-v2/20250207-132604/batch_input_all.jsonl"
    input_file_out = "dataCollection/Dataset/metaworld_soccer-v2/20250207-132604/gemini_results.jsonl"
    merged_file = "dataCollection/Dataset/metaworld_soccer-v2_merged_F.json"

    # 解析输入文件，获得 custom_id -> [cloud_info] 数据结构
    input_map = parse_input_file(input_file_in)
    # 解析输出文件，获得 custom_id -> {analysis, decision} 数据结构
    output_map = parse_output_file(input_file_out)
    # 合并输入与输出数据，并保存最终结果
    merge_and_save(input_map, output_map, merged_file)

if __name__ == "__main__":
    main()
