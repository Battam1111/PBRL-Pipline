# pointllm_infer.py

import threading
import open3d as o3d
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from PIL import Image

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.data import pc_norm, farthest_point_sample
import traceback  # 导入 traceback 模块

class PointCloudSaver:
    """
    用于保存点云的工具类。
    """

    def __init__(self, filename='/home/star/Yanjun/RL-VLM-F/html/point_cloud-input.ply'):
        """
        初始化 PointCloudSaver 实例。

        参数:
            filename (str): 保存点云文件的路径。
        """
        self.filename = filename


    def save_point_cloud(self, point_cloud):
        threading.Thread(target=self._save_point_cloud_file, args=(point_cloud,)).start()

    def _save_point_cloud_file(self, point_cloud):
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
            o3d.io.write_point_cloud(self.filename, pcd)
        except Exception as e:
            print(f"Error saving point cloud: {e}")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 禁用 PyTorch 的权重初始化，以节省内存
disable_torch_init()

# 配置全局变量
MODEL = None
TOKENIZER = None
POINT_BACKBONE_CONFIG = None
KEYWORDS = None
MM_USE_POINT_START_END = None
CONV_TEMPLATE = None

def init_model(model_name="/home/star/Yanjun/PointLLM/checkpoints/PointLLM_7B_v1.2"):
    """
    初始化 PointLLM 模型和相关配置。
    
    参数：
    - model_name：字符串，模型的路径或名称。
    
    返回：
    - 无，函数会设置全局变量。
    """
    global MODEL, TOKENIZER, POINT_BACKBONE_CONFIG, KEYWORDS, MM_USE_POINT_START_END, CONV_TEMPLATE

    # 测试模型
    # model_name = "YirongSun/pcllm_test1"
    # model_name = "YirongSun/plm_test2"
    model_name = "YirongSun/plm_test2_fixed"

    # 防止重复初始化
    if MODEL is not None and TOKENIZER is not None:
        print("PointLLM model already initialized.")
        return

    # 加载分词器
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)

    # 加载模型，指定使用 bfloat16 精度
    MODEL = PointLLMLlamaForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=False, 
        use_cache=True,
        torch_dtype=torch.bfloat16
    ).to(device)


    # 初始化模型的特殊配置
    MODEL.initialize_tokenizer_point_backbone_config_wo_embedding(TOKENIZER)
    MODEL.eval()

    # 获取模型的配置信息
    MM_USE_POINT_START_END = getattr(MODEL.config, "mm_use_point_start_end", False)
    POINT_BACKBONE_CONFIG = MODEL.get_model().point_backbone_config
    
    CONV_TEMPLATE = conv_templates["vicuna_v1_1"].copy()
    stop_str = CONV_TEMPLATE.sep if CONV_TEMPLATE.sep_style != SeparatorStyle.TWO else CONV_TEMPLATE.sep2
    KEYWORDS = [stop_str]

def pointllm_query_1(query_list, temperature=1.0):
    """
    使用 PointLLM 对单个点云进行推理。

    参数：
    - query_list：列表，包含提示词和点云数据。
    - temperature：浮点数，生成的随机性控制。

    返回：
    - 模型的回复文本，或 -1 如果发生错误。
    """
    if MODEL is None or TOKENIZER is None:
        raise ValueError("Model not initialized. Please call init_model() first.")

    try:
        # 从 query_list 中提取信息
        # 假设 query_list 的结构为：
        # [
        #   "Prompt text",
        #   point_cloud (numpy 数组，形状为 (N, 6))
        # ]
        prompt = query_list[0]
        point_cloud = query_list[1]

        # 获取模型的配置参数
        point_token_len = POINT_BACKBONE_CONFIG['point_token_len']
        default_point_patch_token = POINT_BACKBONE_CONFIG['default_point_patch_token']
        default_point_start_token = POINT_BACKBONE_CONFIG['default_point_start_token']
        default_point_end_token = POINT_BACKBONE_CONFIG['default_point_end_token']

        # 预处理点云
        if point_cloud.shape[0] > 8192:
            point_cloud = farthest_point_sample(point_cloud, 8192)
        point_cloud = pc_norm(point_cloud)
        point_clouds = torch.from_numpy(point_cloud).unsqueeze_(0).to(torch.bfloat16).to(device)
        
        

        # 构建对话
        conv = CONV_TEMPLATE.copy()
        qs = prompt

        if MM_USE_POINT_START_END:
            qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
        else:
            qs = default_point_patch_token * point_token_len + '\n' + qs

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # 编码输入
        inputs = TOKENIZER([prompt_text])
        input_ids = torch.as_tensor(inputs.input_ids).to(device)

        # 设置停止条件
        stopping_criteria = KeywordsStoppingCriteria(KEYWORDS, TOKENIZER, input_ids)
        stop_str = KEYWORDS[0]

        # 模型生成
        with torch.inference_mode():
            output_ids = MODEL.generate(
                input_ids,
                point_clouds=point_clouds,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                max_length=2048,
                top_p=0.95,
                stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        outputs = TOKENIZER.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        return outputs
    except Exception as e:
        print(f"[ERROR in pointllm_query_1] {e}")
        return -1

def pointllm_query_2(query_list, summary_prompt, temperature=1):
    """
    使用 PointLLM 比较两个点云，并对初始响应进行总结。

    参数：
    - query_list：列表，包含提示词和点云数据。
      - query_list[0]: prompt1，例如 "\nConsider the following two point clouds:\nPoint Cloud 1:\n"
      - query_list[1]: pc1，第一个点云数据 (numpy array)
      - query_list[2]: prompt2，例如 "\nPoint Cloud 2:\n"
      - query_list[3]: pc2，第二个点云数据 (numpy array)
      - query_list[4]: main_prompt，例如 "\n1. What is represented in Point Cloud 1?\n2. What is represented in Point Cloud 2?\n3. The goal is to move the soccer ball into the goal. Is there any difference between Point Cloud 1 and Point Cloud 2 in terms of achieving the goal?\n"
    - summary_prompt：字符串，用于对初始响应进行总结的提示词。
    - temperature：浮点数，生成的随机性控制。

    返回：
    - 模型的回复文本，或 -1 如果发生错误。
    """
    if MODEL is None or TOKENIZER is None:
        raise ValueError("Model not initialized. Please call init_model() first.")

    try:
        beg = time.time()

        # 从 query_list 中提取信息
        prompt1 = query_list[0]  # "\nConsider the following two point clouds:\nPoint Cloud 1:\n"
        pc1 = query_list[1]      # 第一个点云数据 (numpy array)
        prompt2 = query_list[2]  # "\nPoint Cloud 2:\n"
        pc2 = query_list[3]      # 第二个点云数据 (numpy array)
        main_prompt = query_list[4]  # 问题列表

        # 获取模型的配置参数
        point_token_len = POINT_BACKBONE_CONFIG['point_token_len']
        default_point_patch_token = POINT_BACKBONE_CONFIG['default_point_patch_token']
        default_point_start_token = POINT_BACKBONE_CONFIG.get('default_point_start_token', "<point_start>")
        default_point_end_token = POINT_BACKBONE_CONFIG.get('default_point_end_token', "<point_end>")

        # 预处理点云1
        if pc1.shape[0] > 8192:
            pc1 = farthest_point_sample(pc1, 8192)
        pc1 = pc_norm(pc1)

        # 预处理点云2
        if pc2.shape[0] > 8192:
            pc2 = farthest_point_sample(pc2, 8192)
        pc2 = pc_norm(pc2)

        # 将 pc1 和 pc2 保持独立，存储在列表中
        point_clouds = torch.stack([
            torch.from_numpy(pc1).to(torch.bfloat16).to(device),
            torch.from_numpy(pc2).to(torch.bfloat16).to(device)
        ], dim=0)


        qs = (
            prompt1 + '\n' +
            default_point_start_token + 
            default_point_patch_token * point_token_len + 
            (default_point_end_token + '\n' if POINT_BACKBONE_CONFIG.get('mm_use_point_start_end', False) else '') +
            prompt2 + '\n' +
            default_point_start_token + 
            default_point_patch_token * point_token_len + 
            (default_point_end_token + '\n' if POINT_BACKBONE_CONFIG.get('mm_use_point_start_end', False) else '') +
            main_prompt
        )

        # print(f"Constructed qs:\n{qs}")  # 调试信息

        # 构建对话模板
        try:
            conv = CONV_TEMPLATE.copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            # print(f"Prompt Text:\n{prompt_text}")  # 调试信息
        except Exception as e:
            print(f"[ERROR in pointllm_query_2 - Building Prompt] {e}")
            print(traceback.format_exc())
            return -1

        # 编码输入
        try:
            inputs = TOKENIZER([prompt_text])
            input_ids = torch.as_tensor(inputs.input_ids).to(device)
            # print(f"Input IDs shape: {input_ids.shape}")  # 调试信息
        except Exception as e:
            print(f"[ERROR in pointllm_query_2 - Tokenization] {e}")
            print(traceback.format_exc())
            return -1

        # 设置停止条件
        try:
            stopping_criteria = KeywordsStoppingCriteria(KEYWORDS, TOKENIZER, input_ids)
            stop_str = KEYWORDS[0]
        except Exception as e:
            print(f"[ERROR in pointllm_query_2 - Setting Stopping Criteria] {e}")
            print(traceback.format_exc())
            return -1

        # 初始生成响应
        try:
            with torch.inference_mode():
                output_ids = MODEL.generate(
                    input_ids,
                    point_clouds=point_clouds,  # 传入列表，分别对应两个点云
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    max_length=2048,
                    top_p=0.95,
                    stopping_criteria=[stopping_criteria]
                )
                # print(f"Output IDs shape: {output_ids.shape}")  # 调试信息
        except Exception as e:
            print(f"[ERROR in pointllm_query_2 - Model Generate] {e}")
            print(traceback.format_exc())
            return -1

        # 解码初始响应
        try:
            input_token_len = input_ids.shape[1]
            initial_response = TOKENIZER.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0].strip()
            if initial_response.endswith(stop_str):
                initial_response = initial_response[:-len(stop_str)].strip()
            # print(f"Initial Response: {initial_response}")  # 调试信息
        except Exception as e:
            print(f"[ERROR in pointllm_query_2 - Decoding Initial Response] {e}")
            print(traceback.format_exc())
            return -1

        # 使用 summary_prompt 对初始响应进行总结，废弃
        # try:
        #     summary_input = summary_prompt.format(initial_response)
        #     conv = CONV_TEMPLATE.copy()
        #     conv.append_message(conv.roles[0], summary_input)
        #     conv.append_message(conv.roles[1], None)
        #     summary_prompt_text = conv.get_prompt()
        #     # print(f"Summary Prompt Text:\n{summary_prompt_text}")  # 调试信息
        # except Exception as e:
        #     print(f"[ERROR in pointllm_query_2 - Building Summary Prompt] {e}")
        #     print(traceback.format_exc())
        #     return -1

        # # 编码总结输入
        # try:
        #     summary_inputs = TOKENIZER([summary_prompt_text])
        #     summary_input_ids = torch.as_tensor(summary_inputs.input_ids).to(device)
        #     # print(f"Summary Input IDs shape: {summary_input_ids.shape}")  # 调试信息
        # except Exception as e:
        #     print(f"[ERROR in pointllm_query_2 - Tokenization for Summary] {e}")
        #     print(traceback.format_exc())
        #     return -1

        # # 总结生成
        # try:
        #     with torch.inference_mode():
        #         summary_output_ids = MODEL.generate(
        #             summary_input_ids,
        #             point_clouds=None,  # 总结阶段不传入点云信息
        #             do_sample=True,
        #             temperature=temperature,
        #             top_k=50,
        #             max_length=512,
        #             top_p=0.95,
        #             stopping_criteria=[stopping_criteria]
        #         )
        #         # print(f"Summary Output IDs shape: {summary_output_ids.shape}")  # 调试信息
        # except Exception as e:
        #     print(f"[ERROR in pointllm_query_2 - Model Generate for Summary] {e}")
        #     print(traceback.format_exc())
        #     return -1

        # # 解码总结响应
        # try:
        #     summary_input_token_len = summary_input_ids.shape[1]
        #     summary_response = TOKENIZER.batch_decode(
        #         summary_output_ids[:, summary_input_token_len:], skip_special_tokens=True
        #     )[0].strip()
        #     if summary_response.endswith(stop_str):
        #         summary_response = summary_response[:-len(stop_str)].strip()
        #     # print(f"Summary Response: {summary_response}")  # 调试信息
        # except Exception as e:
        #     print(f"[ERROR in pointllm_query_2 - Decoding Summary Response] {e}")
        #     print(traceback.format_exc())
        #     return -1

        end = time.time()
        # 可以注释掉
        # print("Time elapsed: ", end - beg)

        # return summary_response
        return initial_response

    except Exception as e:
        print(f"[ERROR in pointllm_query_2 - General] {e}")
        print(traceback.format_exc())
        return -1



# 示例用法
if __name__ == "__main__":
    from prompt import (
        pointllm_free_query_prompt1, pointllm_free_query_prompt2,
        pointllm_free_query_env_prompts, pointllm_summary_env_prompts,
        goal_env_prompts  # 假设已定义
    )
    import numpy as np
    from PIL import Image

    def farthest_point_sample(pc, n_samples):
        # 实现最近点采样或其他采样方法
        # 这里只是一个示例，实际需要根据具体方法实现
        if pc.shape[0] <= n_samples:
            return pc
        indices = np.random.choice(pc.shape[0], n_samples, replace=False)
        return pc[indices]

    def pc_norm(pc):
        # 实现点云归一化
        # 这里只是一个示例，实际需要根据具体方法实现
        centroid = np.mean(pc[:, :3], axis=0)
        pc[:, :3] -= centroid
        scale = np.max(np.linalg.norm(pc[:, :3], axis=1))
        pc[:, :3] /= scale
        return pc

    class PointCloudSaver:
        def save_point_cloud(self, pc):
            # 实现点云保存逻辑
            # 这里只是一个示例，实际需要根据具体方法实现
            pass

    # 假设已初始化 MODEL 和 TOKENIZER
    # 初始化模型和 tokenizer
    # init_model()  # 根据实际情况调用初始化函数

    # 加载点云数据（示例）
    pc1 = np.random.rand(5000, 6).astype(np.float32)  # 示例点云1
    pc2 = np.random.rand(5000, 6).astype(np.float32)  # 示例点云2

    # 设置环境名称和相关提示
    env_name = "metaworld_sweep-into-v2"
    pointllm_free_query_env_prompts = {
        env_name: "The goal is to sweep into v2."
    }
    pointllm_summary_env_prompts = {
        env_name: "Based on the initial response, summarize the preference."
    }

    # 准备提示词
    prompt1 = pointllm_free_query_prompt1
    prompt2 = pointllm_free_query_prompt2
    main_prompt = pointllm_free_query_env_prompts[env_name]
    summary_prompt = pointllm_summary_env_prompts[env_name]

    # 调用 pointllm_query_2
    res = pointllm_query_2(
        [
            prompt1,
            pc1.reshape(-1, 6),
            prompt2,
            pc2.reshape(-1, 6),
            main_prompt
        ],
        summary_prompt,
    )

    print("Summary Response:", res)

