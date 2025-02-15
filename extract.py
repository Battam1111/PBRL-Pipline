import re

def extract_label_from_response(response) -> int:
    """
    从模型返回的响应文本中提取决策标签，支持多种格式：
      1. 如果文本中存在 "Conclusion:"（或其变体）后跟决策数字，则优先采用。
      2. 搜索 Markdown 加粗格式（如 **0**），并采用最后一次出现的匹配结果。
      3. 检查文本末尾非空行是否单独为决策标签（去除可能的 Markdown 包裹）。
      4. 扫描列表项格式（例如 "- 0" 或 "* 0"），取最后一次匹配。
      5. 最后，在全文中查找孤立数字，采用最后一个匹配项。
      
    仅接受的决策标签为 0、1 或 -1；若均无法明确提取，则返回 -1。
    
    参数:
      response: 模型返回的响应内容（任意类型，最终会转换为字符串）
    
    返回:
      int: 解析得到的决策标签（0、1 或 -1）
    """
    # 1. 预处理：确保响应为字符串并去除首尾空白
    text = str(response).strip()
    
    # 策略 1：在 "Conclusion:" 上下文中寻找标签（忽略大小写）
    conclusion_pattern = re.compile(r'(?i)conclusion\s*[:：]\s*([01]|-1)')
    m = conclusion_pattern.search(text)
    if m:
        candidate = m.group(1)
        if candidate in {"0", "1", "-1"}:
            return int(candidate)
    
    # 策略 2：搜索 Markdown 加粗格式（例如 **0**）
    # 当存在多个匹配时，取最后一次出现的——通常模型的最终答案在文本末尾
    markdown_pattern = re.compile(r'\*\*\s*([01]|-1)\s*\*\*')
    markdown_matches = markdown_pattern.findall(text)
    if markdown_matches:
        candidate = markdown_matches[-1]
        if candidate in {"0", "1", "-1"}:
            return int(candidate)
    
    # 策略 3：检查文本末尾的非空行
    # 通常最终决策会独立占据一行，可能还带有 Markdown 包裹
    lines = text.splitlines()
    for line in reversed(lines):
        stripped_line = line.strip()
        if not stripped_line:
            continue
        # 如果行以 Markdown 标记包裹，则去除两侧的星号
        if stripped_line.startswith("**") and stripped_line.endswith("**"):
            stripped_line = stripped_line.strip("*").strip()
        # 如果这一行恰好为有效决策数字，则直接返回
        if stripped_line in {"0", "1", "-1"}:
            return int(stripped_line)
        # 尝试精确匹配单一数字（例如 " 0 "）
        if re.fullmatch(r'([01]|-1)', stripped_line):
            return int(stripped_line)
    
    # 策略 4：搜索列表项格式，如 "- 0" 或 "* 0"（多行匹配）
    list_pattern = re.compile(r'^[\-\*]\s+([01]|-1)\s*$', re.MULTILINE)
    list_matches = list_pattern.findall(text)
    if list_matches:
        return int(list_matches[-1])
    
    # 策略 5：最后在全局范围内查找孤立的决策数字，采用最后一次出现
    isolated_pattern = re.compile(r'\b([01]|-1)\b')
    isolated_matches = isolated_pattern.findall(text)
    if isolated_matches:
        return int(isolated_matches[-1])
    
    # 如果所有策略均未能明确解析，则返回默认值 -1
    return -1


# ========================= 示例调用 =========================
if __name__ == '__main__':
    res_text = (
        "1. Situation 1 shows a soccer ball positioned near an angled structure resembling a goal, "
        "with an opening that is slightly ajar but not directly aligned with the goal’s entry.\n\n\n"
        "2. Situation 2 shows the same angled goal structure as in Situation 1, but with the soccer ball "
        "slightly farther from the goal. The ball is positioned at a greater distance from the structure’s opening, "
        "making it harder to access the goal.\n\n\n"
        "3. In terms of achieving the objective of moving the ball into the goal, Situation 1 is better as the ball is closer to the goal.\n\n\n"
        "**0**"
    )
    label = extract_label_from_response(res_text)
    print("Extracted label:", label)
