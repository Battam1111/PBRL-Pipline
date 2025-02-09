#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transform_api_data.py

本脚本用于将“生成的输入文件”（例如 batch_input_all.jsonl）与“模型返回的输出文件”
（例如 direct_api_results.jsonl 或 batch_output_merged.jsonl）合并为一个包含全部信息的 JSON 数组，
便于后续数据分析与应用。

主要处理步骤：
  1) 解析输入文件（parse_input_file）：
     - 逐行读取输入 JSONL 文件，提取每行中的 custom_id 以及请求内容；
     - 针对新版 JSONL 文件中用户消息的 content 字段为列表，遍历该列表，
       根据消息类型分别提取文本信息与 image_url；
     - 采用“当前情景”标记法：当文本中出现 "Situation 1:" 或 "Situation 2:" 时，
       切换当前情景；遇到 image_url 类型项时，将其 URL归入当前情景（若当前情景为空则按出现顺序归类）；
     - 对每个 image_url，调用 os.path.basename() 得到文件名，并利用 parse_filename() 提取关键信息：
           idx, bin, r, t, emb, view
       其中 view 为可选字段（若未拼接则返回空字符串）。
     - 最终以 custom_id 为键构造字典，每个 custom_id 对应一个列表，
       列表中每个元素为字典格式，包含 { "situation": 1 或 2, "idx": ..., "bin": ..., "r": ..., "t": ..., "emb": "...", "view": "...", "url": "..." }。
       
  2) 解析输出文件（parse_output_file）：
     - 逐行读取输出 JSONL 文件，解析每行 JSON 数据；
     - 兼容 Batch 模式与 Direct 模式两种返回格式，从 response 中提取回答文本，
       回答文本通常位于 response.body.choices[0].message.content 或 response.choices[0].message.content；
     - 调用 parse_decision() 从回答文本中提取最终决策结果： '0'、'1' 或 '-1'（若无法解析，则返回空字符串）；
     - 最终构造字典： { custom_id -> { "analysis": 模型回答原文, "decision": "0" 或 "1" 或 "-1" } }。
     
  3) 合并并保存（merge_and_save）：
     - 以 custom_id 为键，对输入数据和输出数据进行合并；
     - 每条合并记录格式为：
           {
             "custom_id": "xxx",
             "cloud_info": [ record1, record2, ... ],
             "analysis": "模型回答原文",
             "decision": "0" 或 "1" 或 "-1"
           }
     - 将所有记录组成列表，并按 custom_id 排序后写入目标输出文件（格式化为 JSON 数组）。
     
使用方式：
   python transform_api_data.py

请根据实际情况在 main() 函数中修改以下文件路径：
   - input_file_in  => 生成的输入文件（例如 batch_input_all.jsonl）
   - input_file_out => 模型返回的输出文件（例如 direct_api_results.jsonl 或 batch_output_merged.jsonl）
   - merged_file    => 合并后输出文件的路径
"""

import os
import re
import json
import sys

# -----------------------------------------------------------------------------
# 更新正则表达式：支持最新项目中两种文件名格式（包含 _sid 部分，可选 _view 部分）
# 例如：pc_005692_bin_2_r_3.56_t_6304.00_emb_e2a0ed5a_sid5692.jpg
# -----------------------------------------------------------------------------
FILENAME_REGEX = r"^pc_(\d+)_bin_(\d+)_r_([\d.]+)_t_([\d.]+)_emb_([a-zA-Z0-9]+)(?:_sid\d+)?(?:_view(\w+))?\.jpg$"


def parse_decision(content: str) -> str:
    """
    从模型回答文本 content 中提取最终决策结果，返回 '0', '1' 或 '-1'。

    解析策略（按优先级顺序）：
      0. 如果存在 "Conclusion"（或 "**Conclusion**"）关键词，则只在其后部分查找决策数值，
         并优先匹配形如 "Conclusion: -1" 或 "**Conclusion**: 1" 的格式。
      1. 整行匹配：如果某一行完全由决策数字构成，则视为最终决策。
      2. Markdown 加粗格式匹配：例如 **-1**。
      3. 列表格式匹配：要求行以 "-" 开头，且 "-" 后必须有空格，避免将 "-1" 中的 "-" 误认为 bullet 标记。
      4. 空格分隔匹配：匹配独立数字，要求前后为空白、标点或字符串边界。

    在每个步骤中，若匹配到多个但不一致的结果，则返回空字符串 "" 表示不明确。

    :param content: 模型返回的回答文本
    :return: 提取到的决策结果，或 ""（表示无法确定）
    """
    # 如果文本中存在 "conclusion" 关键词，则限定只在其后的区域中查找（忽略大小写）
    lower_content = content.lower()
    idx = lower_content.rfind("conclusion")
    if idx != -1:
        region = content[idx:]
    else:
        region = content

    # Step 0: 优先尝试从 Conclusion 上下文中直接提取决策数字
    # 支持格式例如： **Conclusion**: -1、Conclusion: 1 等
    conclusion_regex = re.compile(r'(?i)(?:\*\*?conclusion\*\*?)[\s:：]*((?:-1)|[01])')
    concl_matches = conclusion_regex.findall(region)
    if concl_matches:
        unique = set(concl_matches)
        if len(unique) == 1:
            return unique.pop()
        else:
            return ""

    # Step 1: 整行匹配：整行仅包含决策数字（允许前后有空白）
    line_regex = re.compile(r'(?m)^\s*((?:-1)|[01])\s*$')
    line_matches = line_regex.findall(region)
    if line_matches:
        unique = set(line_matches)
        if len(unique) == 1:
            return unique.pop()
        else:
            return ""

    # Step 2: Markdown 加粗格式匹配，如 **-1**
    md_regex = re.compile(r'\*\*\s*((?:-1)|[01])\s*\*\*')
    md_matches = md_regex.findall(region)
    if md_matches:
        unique = set(md_matches)
        if len(unique) == 1:
            return unique.pop()
        else:
            return ""

    # Step 3: 列表格式匹配：行以 "-" 开头，后跟至少一个空格，再匹配决策数字
    bullet_regex = re.compile(r'(?m)^\s*-\s+["\']?\s*((?:-1)|[01])\s*["\']?')
    bullet_matches = bullet_regex.findall(region)
    if bullet_matches:
        unique = set(bullet_matches)
        if len(unique) == 1:
            return unique.pop()
        else:
            return ""

    # Step 4: 空格分隔匹配：要求前后为空白、标点或字符串边界
    space_regex = re.compile(r'(?<!\S)((?:-1)|[01])(?=$|\s|[.,;:!?])')
    space_matches = space_regex.findall(region)
    if space_matches:
        unique = set(space_matches)
        if len(unique) == 1:
            return unique.pop()
        else:
            return ""

    return ""




def parse_filename(filename: str) -> dict:
    """
    解析给定文件名（不含路径）的关键信息。支持两种格式：
      1) 原始视角图像文件名，例如：
         pc_003145_bin_1_r_2.57_t_3410.00_emb_dbc1ca39_sid3145.jpg
      2) 拼接后图像文件名，例如：
         pc_003145_bin_1_r_2.57_t_3410.00_emb_dbc1ca39.jpg
         或包含 _view 后缀，如：
         pc_003145_bin_1_r_2.57_t_3410.00_emb_dbc1ca39_sid3145_viewA.jpg
         
    利用正则表达式 FILENAME_REGEX 提取以下字段：
       idx, bin, r, t, emb, view（若 view 部分不存在，则返回空字符串）
    
    若匹配失败，则返回空字典 {}。
    
    :param filename: 文件名字符串（不含路径）
    :return: 包含解析结果的字典
             示例： { "idx": 5692, "bin": 2, "r": 3.56, "t": 6304.00, "emb": "e2a0ed5a", "view": "" }
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
    解析输入文件（例如 batch_input_all.jsonl），逐行读取 JSON 数据，
    提取每个请求的 custom_id 与用户消息中包含的图像 URL 信息。
    
    针对新版 JSONL 文件：
      - 用户消息部分的 content 为列表，可能包含文本和 image_url 两种类型；
      - 采用“当前情景”标记法：当文本中出现 "Situation 1:" 或 "Situation 2:" 时更新当前情景；
      - 遇到 image_url 类型项时，将其中的 URL 归入当前情景；
      - 若当前情景无法判断，则按出现顺序：第一个归为情景 1，第二个归为情景 2，后续默认归为情景 2；
      
    对于每个 image_url：
      - 调用 os.path.basename() 获取文件名；
      - 调用 parse_filename() 解析文件名，提取 idx, bin, r, t, emb, view 信息；
      - 将当前记录构造为字典： { "situation": 1或2, "idx": ..., "bin": ..., "r": ..., "t": ..., "emb": "...", "view": "...", "url": "..." }
      
    最终返回字典： { custom_id -> [record1, record2, ...] }
    
    :param input_file_in: 输入文件路径（例如 batch_input_all.jsonl）
    :return: 解析后的数据字典
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

            cid = obj.get("custom_id", "")
            if not cid:
                continue

            body = obj.get("body", {})
            messages = body.get("messages", [])
            if len(messages) < 2:
                continue

            # 获取用户消息部分（通常为最后一条消息），注意 content 为列表
            user_msg = messages[-1].get("content", [])
            if not isinstance(user_msg, list):
                # 兼容旧格式，将非列表内容当作纯文本
                user_content = str(user_msg)
                user_msg = [{"type": "text", "text": user_content}]

            # 采用当前情景标记法
            current_situation = None
            extracted = []
            for item in user_msg:
                typ = item.get("type", "").lower()
                if typ == "text":
                    text = item.get("text", "")
                    # 检查是否包含情景标记，忽略大小写
                    if "situation 1" in text.lower():
                        current_situation = 1
                    elif "situation 2" in text.lower():
                        current_situation = 2
                    # 其它文本信息暂不处理
                elif typ == "image_url":
                    # 获取 image_url 中的 URL，注意兼容字典格式
                    url = ""
                    # 支持两种格式：直接 url 字符串或字典格式 {"url": "..."}
                    if isinstance(item.get("image_url"), dict):
                        url = item.get("image_url", {}).get("url", "").strip()
                    elif isinstance(item.get("image_url"), str):
                        url = item.get("image_url").strip()
                    if not url:
                        continue
                    if current_situation is None:
                        # 如果未能从文本中判断情景，则按出现顺序判断：第一个为情景1，第二个为情景2
                        if len(extracted) == 0:
                            current_situation = 1
                        elif len(extracted) == 1:
                            current_situation = 2
                        else:
                            current_situation = 2
                    # 从 URL 中提取文件名，并调用 parse_filename() 解析关键信息
                    fname = os.path.basename(url)
                    file_info = parse_filename(fname)
                    if not file_info:
                        file_info = {}
                    record = {
                        "situation": current_situation,
                        "idx": file_info.get("idx"),
                        "bin": file_info.get("bin"),
                        "r": file_info.get("r"),
                        "t": file_info.get("t"),
                        "emb": file_info.get("emb", ""),
                        "view": file_info.get("view", ""),
                        "url": url
                    }
                    extracted.append(record)
            # 如果通过 image_url 未能提取到任何记录，则尝试从全部文本中提取（兼容旧格式）
            if not extracted:
                full_text = ""
                for item in user_msg:
                    if item.get("type", "").lower() == "text":
                        full_text += item.get("text", "") + " "
                pattern = re.compile(r"Situation\s+1\s*:\s*(.*?)\s*Situation\s+2\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)
                mm = pattern.search(full_text)
                if mm:
                    s1_str = mm.group(1).strip()
                    s2_str = mm.group(2).strip()
                    s1_urls = s1_str.split()
                    s2_urls = s2_str.split()
                    for u in s1_urls:
                        fname = os.path.basename(u)
                        file_info = parse_filename(fname)
                        record = {
                            "situation": 1,
                            "idx": file_info.get("idx"),
                            "bin": file_info.get("bin"),
                            "r": file_info.get("r"),
                            "t": file_info.get("t"),
                            "emb": file_info.get("emb", ""),
                            "view": file_info.get("view", ""),
                            "url": u
                        }
                        extracted.append(record)
                    for u in s2_urls:
                        fname = os.path.basename(u)
                        file_info = parse_filename(fname)
                        record = {
                            "situation": 2,
                            "idx": file_info.get("idx"),
                            "bin": file_info.get("bin"),
                            "r": file_info.get("r"),
                            "t": file_info.get("t"),
                            "emb": file_info.get("emb", ""),
                            "view": file_info.get("view", ""),
                            "url": u
                        }
                        extracted.append(record)
            result_map[cid] = extracted
            matched_count += 1

    print(f"[INFO] 输入文件解析完成：共 {line_count} 行，其中成功解析 {matched_count} 行；总 custom_id 数量 = {len(result_map)}")
    return result_map

def parse_output_file(input_file_out: str) -> dict:
    """
    解析输出文件（例如 batch_output_merged.jsonl 或 direct_api_results.jsonl），逐行读取 JSON 数据，
    提取每条记录的 custom_id 以及模型返回的回答文本，并利用 parse_decision() 提取决策结果。
    
    为兼容不同调用模式下的返回格式：
      - 优先尝试从 response.body.choices[0].message.content 中获取回答文本；
      - 若未成功，则尝试从 response.choices[0].message.content 中获取；
    
    最终返回字典： { custom_id -> { "analysis": 回答文本, "decision": "0"/"1"/"-1" } }
    
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
                print(f"[警告] 输出文件行 {line_count} JSON解析失败: {e}")
                continue

            cid = obj.get("custom_id", "")
            if not cid:
                continue

            response_obj = obj.get("response", {})
            # 尝试从 body 中获取（Batch 模式）
            body_obj = response_obj.get("body", {})
            choices = body_obj.get("choices", [])
            if not choices:
                # 兼容 Direct 模式
                choices = response_obj.get("choices", [])
            if not choices:
                continue

            content = ""
            try:
                content = choices[0]["message"]["content"]
            except Exception:
                pass
            if not content:
                continue

            decision = parse_decision(content)
            result_map[cid] = {
                "analysis": content,
                "decision": decision
            }
            valid_count += 1

    print(f"[INFO] 输出文件解析完成：共 {line_count} 行，其中成功解析 {valid_count} 行。")
    return result_map

def merge_and_save(input_map: dict, output_map: dict, merged_file: str) -> None:
    """
    以 custom_id 为键将输入数据与输出数据进行合并，
    生成包含全部信息的记录数组，每条记录格式为：
         {
           "custom_id": "xxx",
           "cloud_info": [ { "situation": 1, "idx": ..., "bin": ..., "r": ..., "t": ..., "emb": "...", "view": "...", "url": "..." }, ... ],
           "analysis": "模型回答原文",
           "decision": "0"/"1"/"-1"
         }
    合并后按照 custom_id 排序，并写入目标输出文件（格式化为 JSON 数组）。
    
    :param input_map: 由 parse_input_file() 得到的数据字典 { custom_id -> [ {...}, ... ] }
    :param output_map: 由 parse_output_file() 得到的数据字典 { custom_id -> { "analysis": ..., "decision": ... } }
    :param merged_file: 合并结果输出文件的路径
    """
    merged_list = []
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

    # 按 custom_id 进行排序（可根据需要调整排序规则）
    merged_list.sort(key=lambda x: x["custom_id"])

    os.makedirs(os.path.dirname(merged_file), exist_ok=True)
    with open(merged_file, "w", encoding="utf-8") as outf:
        json.dump(merged_list, outf, ensure_ascii=False, indent=2)

    print(f"[完成] 合并输出文件生成：{merged_file}；共 {len(merged_list)} 条记录。")

def main():
    """
    主函数：
      1. 根据实际情况指定输入文件和输出文件的路径：
            - input_file_in  : 生成的输入文件（例如 batch_input_all.jsonl）
            - input_file_out : 模型返回的输出文件（例如 direct_api_results.jsonl 或 batch_output_merged.jsonl）
            - merged_file    : 最终合并后输出文件的保存路径
      2. 分别调用 parse_input_file() 和 parse_output_file() 解析数据，
         然后调用 merge_and_save() 生成最终合并结果。
    """
    # 示例文件路径，请根据实际情况进行修改
    input_file_in = "dataCollection/Dataset/metaworld_soccer-v2/20250209-144201/batch_input_all.jsonl"
    input_file_out = "dataCollection/Dataset/metaworld_soccer-v2/20250209-144201/batch_output_merged.jsonl"
    merged_file = "dataCollection/Dataset/metaworld_soccer-v2_merged_F.json"

    print("[INFO] 开始解析输入文件……")
    input_map = parse_input_file(input_file_in)
    
    print("[INFO] 开始解析输出文件……")
    output_map = parse_output_file(input_file_out)
    
    print("[INFO] 开始合并数据并写入文件……")
    merge_and_save(input_map, output_map, merged_file)

if __name__ == "__main__":
    main()
