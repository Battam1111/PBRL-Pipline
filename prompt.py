import copy

clip_env_prompts = {
    "metaworld_sweep-into-v2": "The green cube is in the hole.",  # 未解决，有奖励问题
    "metaworld_drawer-open-v2": "The drawer is opened.",          # 尝试翻转版本
    "metaworld_door-open-v2": "The safe door is opened.",         # 尝试翻转版本
    "metaworld_soccer-v2": "The soccer ball is in the goal.",     # 未解决，有奖励问题

    "metaworld_peg-insert-side-v2": "The peg is inserted into the hole.",
    "metaworld_hand-insert-v2": "The hand is inserted into the hole.",
    "metaworld_shelf-place-v2": "The object is placed on the shelf.",
    "metaworld_disassemble-v2": "The peg is disassembled.",
    "metaworld_handle-pull-side-v2": "The handle is pulled to the side.",

    "CartPole-v1": "pole vertically upright on top of the cart.",
    "softgym_RopeFlattenEasy": "The blue rope is straightened.",
    "softgym_PassWater": "The container, which holds water, is as close to the red circle as possible without causing too many water droplets to spill.",
    "softgym_ClothFoldDiagonal": "The cloth is folded diagonally from top left corner to bottom right corner.",
}

# RL-VLM-F 使用的目标提示
goal_env_prompts = {
    "metaworld_sweep-into-v2": "to minimize the distance between the green cube and the hole",  # 未解决，有奖励问题
    "metaworld_drawer-open-v2": "to open the drawer",           # 尝试翻转版本
    "metaworld_door-open-v2": "to open the safe door",          # 尝试翻转版本
    "metaworld_soccer-v2": "to move the soccer ball into the goal",  # 未解决，有奖励问题

    # 后加的任务
    "metaworld_peg-insert-side-v2": "to insert the peg into the hole",
    "metaworld_hand-insert-v2": "to insert the hand into the hole",
    "metaworld_shelf-place-v2": "to place the object on the shelf",
    "metaworld_disassemble-v2": "to disassemble the peg",
    "metaworld_handle-pull-side-v2": "to pull the handle to the side",

    "CartPole-v1": "to balance the brown pole on the black cart to be upright",
    "softgym_RopeFlattenEasy": "to straighten the blue rope",
    "softgym_PassWater": "to move the container, which holds water, to be as close to the red circle as possible without causing too many water droplets to spill",
    "softgym_ClothFoldDiagonal": "to fold the cloth diagonally from top left corner to bottom right corner",
}

##########################################################################
### 让 Gemini 输出带有两阶段分析的偏好 #################################
##########################################################################

gemini_free_query_prompt1 = """
Consider the following two images:
Image 1:
"""

gemini_free_query_prompt2 = """
Image 2:
"""

gemini_free_query_env_prompts = {}
gemini_free_query_template = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
"""

for env_name, prompt in goal_env_prompts.items():
    gemini_free_query_env_prompts[env_name] = gemini_free_query_template.format(prompt)

# 偏好总结提示
gemini_summary_env_prompts = {}

# 模板 1
gemini_summary_template = """
Based on the text below to the questions:
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
{}

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference.
"""

for env_name, prompt in goal_env_prompts.items():
    gemini_summary_env_prompts[env_name] = gemini_summary_template.format(prompt, "{}")

######################################################################
### 让 Gemini 输出带有单阶段分析的偏好 #################################
######################################################################

gemini_single_query_prompt_template = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference.
"""

gemini_single_query_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    gemini_single_query_env_prompts[env_name] = gemini_single_query_prompt_template.format(prompt)

######################################################################
### 让 Gemini 只输出标签的偏好 #########################################
######################################################################

gemini_single_query_no_analysis_prompt_template = """
The goal is {}. Is the goal better achieved in Image 1 or Image 2?
At the end of the response, reply a single line of: 
0 if the goal is better achieved in Image 1, 
1 if it is better achieved in Image 2, or
-1 if there is no difference or if it is unclear.
"""

### 让 Gemini 输出得分
gemini_score_prompt_start = """
Consider the following image:
"""

gemini_score_template = """
1. What is shown in the image?
2. The goal is {}. On a scale of 0 to 1, the score is 1 if the goal is achieved. What score would you give the image in terms of achieving the goal?
"""

gemini_score_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    gemini_score_env_prompts[env_name] = gemini_score_template.format(prompt)

gemini_score_summary_template = """
Based on the text below to the questions: 
1. What is shown in the image?
2. The goal is {}. On a scale of 0 to 1, the score is 1 if the goal is achieved. What score would you give the image in terms of achieving the goal?
{}

Please reply a single line of the score the text has given.
Reply -1 if the text is unsure.
"""

gemini_score_summary_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    gemini_score_summary_env_prompts[env_name] = gemini_score_summary_template.format(prompt, "{}")





######################################################################
### 为 PointLLM 添加类似 Gemini 的提示 ###################################
######################################################################

# 定义 PointLLM 的提示

# 通用模板定义
pointllm_free_query_prompt1 = """
Consider the following two point clouds:
Point Cloud 1:
"""

pointllm_free_query_prompt2 = """
Point Cloud 2:
"""

pointllm_free_query_env_prompts = {}
pointllm_free_query_template = """
1. What is represented in Point Cloud 1?
2. What is represented in Point Cloud 2?
3. The objective is {}. Is there any difference between Point Cloud 1 and Point Cloud 2 in terms of achieving the objective?
"""

# 根据任务生成模板
for env_name, prompt in goal_env_prompts.items():
    pointllm_free_query_env_prompts[env_name] = pointllm_free_query_template.format(prompt)





# 偏好总结提示模板
pointllm_summary_env_prompts = {}

# 模板优化，明确逻辑层级并简化规则
pointllm_summary_template = """
Based on the text below to the questions:
1. What is represented in Point Cloud 1?
2. What is represented in Point Cloud 2?
3. The objective is {}. Is there any difference between Point Cloud 1 and Point Cloud 2 in terms of achieving the objective?
{}

Is the objective better achieved in Point Cloud 1 or Point Cloud 2?
Reply a single line of 1 if the goal is better achieved in Point Cloud 1, or 2 if it is better achieved in Point Cloud 2.
Reply 0 if the text is unsure or there is no difference.
"""

# 根据环境生成不同目标评估模板
for env_name, prompt in goal_env_prompts.items():
    pointllm_summary_env_prompts[env_name] = pointllm_summary_template.format(prompt, "{}")






######################################################################
### 让 PointLLM 输出带有单阶段分析的偏好 ################################
######################################################################

pointllm_single_query_prompt_template = """
1. What is represented in Point Cloud 1?
2. What is represented in Point Cloud 2?
3. The goal is {}. Is there any difference between Point Cloud 1 and Point Cloud 2 in terms of achieving the goal?

Is the goal better achieved in Point Cloud 1 or Point Cloud 2?
Reply a single line of 0 if the goal is better achieved in Point Cloud 1, or 1 if it is better achieved in Point Cloud 2.
Reply -1 if the text is unsure or there is no difference.
"""

pointllm_single_query_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    pointllm_single_query_env_prompts[env_name] = pointllm_single_query_prompt_template.format(prompt)

######################################################################
### 让 PointLLM 只输出标签的偏好 ########################################
######################################################################

pointllm_single_query_no_analysis_prompt_template = """
The goal is {}. Is the goal better achieved in Point Cloud 1 or Point Cloud 2?
At the end of the response, reply a single line of: 
0 if the goal is better achieved in Point Cloud 1, 
1 if it is better achieved in Point Cloud 2, or
-1 if there is no difference or if it is unclear.
"""

### 让 PointLLM 输出得分
pointllm_score_prompt_start = """
Consider the following point cloud:
"""

pointllm_score_template = """
1. What is represented in the point cloud?
2. The goal is {}. On a scale of 0 to 1, the score is 1 if the goal is achieved. What score would you give the point cloud in terms of achieving the goal?
"""

pointllm_score_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    pointllm_score_env_prompts[env_name] = pointllm_score_template.format(prompt)

pointllm_score_summary_template = """
Based on the text below to the questions: 
1. What is represented in the point cloud?
2. The goal is {}. On a scale of 0 to 1, the score is 1 if the goal is achieved. What score would you give the point cloud in terms of achieving the goal?
{}

Please reply a single line of the score the text has given.
Reply -1 if the text is unsure.
"""

pointllm_score_summary_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    pointllm_score_summary_env_prompts[env_name] = pointllm_score_summary_template.format(prompt, "{}")

######################################################################
### GPT 使用与 Gemini 相同的提示模板 #####################################
######################################################################

gpt_free_query_env_prompts = {}
gpt_free_query_template = copy.deepcopy(gemini_free_query_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_free_query_env_prompts[env_name] = gpt_free_query_template.format(prompt)

gpt_summary_env_prompts = {}
gpt_summary_template = copy.deepcopy(gemini_summary_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_summary_env_prompts[env_name] = gpt_summary_template.format(prompt, "{}")

gpt_score_query_env_prompts = {}
gpt_score_template = copy.deepcopy(gemini_score_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_score_query_env_prompts[env_name] = gpt_score_template.format(prompt)

gpt_score_summary_env_prompts = {}
gpt_score_summary_template = copy.deepcopy(gemini_score_summary_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_score_summary_env_prompts[env_name] = gpt_score_summary_template.format(prompt, "{}")
