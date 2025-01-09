import os
import json
import base64
import time
import requests
from typing import Dict, Any, List
from PIL import Image
import io

# 配置部分
API_URL = "https://api.openai.com/v1/chat/completions"
BATCH_API_URL = "https://api.openai.com/v1/batches"
FILES_API_URL = "https://api.openai.com/v1/files"

API_KEY = "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"  # 请替换为实际的 API 密钥
MODEL = "gpt-4o-2024-11-20"

# 系统 Prompt，用于指导 GPT-4O 行为
SYSTEM_PROMPT = (
    "You are an AI assistant designed to analyze and compare 3D point clouds (Already rendered as images). "
    "You must strictly follow instructions and adhere to the defined response formats: "
    "1. Provide concise descriptions of image features relevant to the given task. "
    "2. Focus on the objective when performing comparisons. Avoid adding speculative or irrelevant details. "
    "3. Follow the response rules provided in the task. "
    "4. Pay special attention to the presence of robotic grippers or mechanical claws in the images, as they may be crucial for understanding task execution. "
)


# 任务目标提示，每个环境对应一个特定任务描述
objective_env_prompts = {
    "metaworld_drawer-open-v2": "to open the drawer",
    "metaworld_door-open-v2": "to open the safe door",
    "metaworld_soccer-v2": "to move the soccer ball into the goal",
    "metaworld_peg-insert-side-v2": "to insert the peg into the hole",
    "metaworld_disassemble-v2": "to disassemble the peg",
    "metaworld_handle-pull-side-v2": "to pull the handle to the side",
}

# 定义单轮对话模板，用于生成用户请求内容
single_round_template = (
    "Consider the following two images:\n"
    "Image 1:\n{img1}\n"
    "Image 2:\n{img2}\n"
    "Objective: {objective}\n\n"
    "Please answer the following questions one by one:\n"
    "1. What is shown in Image 1?\n"
    "2. What is shown in Image 2?\n"
    "3. Is there any difference between Image 1 and Image 2 in terms of achieving the objective?\n\n"
    "After answering these questions, based on your answers, conclude by replying with a single line:\n"
    "- Reply '0' if the objective is better achieved in Image 1.\n"
    "- Reply '1' if the objective is better achieved in Image 2.\n"
    "- Reply '-1' if you are unsure or there is no difference."
)


# 图像根目录，存放各环境的图像文件夹
IMAGES_ROOT_DIR = "dataCollection/Test"

class GPT4OHandler:
    """
    封装与 GPT-4O 和 Batch API 交互的相关方法，包括发送聊天请求、上传文件、创建与检查批处理等。
    """
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def send_chat_request(self, messages: List[Dict[str, str]], max_tokens: int = 1500) -> Dict[str, Any]:
        """发送单个聊天请求至 GPT-4O，并返回响应的 JSON。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"请求错误：{e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"未知错误：{e}")
            return {"error": str(e)}

    def upload_batch_file(self, file_path: str) -> str:
        """上传 .jsonl 格式的批处理输入文件到 OpenAI，返回文件 ID。"""
        if not file_path.endswith(".jsonl"):
            raise ValueError("上传的文件必须是 .jsonl 格式")

        with open(file_path, "rb") as file:
            response = requests.post(
                FILES_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (os.path.basename(file_path), file, "application/json")},
                data={"purpose": "batch"},
            )
            response.raise_for_status()
            return response.json()["id"]

    def create_batch(self, input_file_id: str) -> Dict[str, Any]:
        """基于上传的文件 ID 创建一个批处理任务。"""
        payload = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }
        response = requests.post(BATCH_API_URL, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """查询指定批处理任务的状态。"""
        response = requests.get(f"{BATCH_API_URL}/{batch_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def download_batch_results(self, output_file_id: str, save_path: str = "batch_output.jsonl"):
        """下载批处理任务的结果并保存到本地文件。"""
        response = requests.get(f"{FILES_API_URL}/{output_file_id}/content", headers=self.headers)
        response.raise_for_status()
        with open(save_path, "w", encoding='utf-8') as file:
            file.write(response.text)

def build_messages(user_prompt: str) -> List[Dict[str, str]]:
    """根据用户提示构建系统和用户消息列表。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

def compress_image_to_base64(image_path: str, target_size=(400, 400), quality=75) -> str:
    """
    读取图像文件，调整大小并转换为 JPEG 格式以压缩数据，然后编码为 Base64 字符串。
    使用 LANCZOS 作为高质量重采样滤波器。
    """
    try:
        with Image.open(image_path) as img:
            # 调整图像大小并使用 LANCZOS 滤波器
            img = img.resize(target_size, Image.LANCZOS)
            buffered = io.BytesIO()
            # 将图像保存为 JPEG 格式
            img.save(buffered, format="JPEG", quality=quality)
            base64_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            # 使用 Markdown 格式嵌入图片
            return f"![image](data:image/jpeg;base64,{base64_image_data})"
    except Exception as e:
        print(f"压缩或编码图像 {image_path} 失败：{e}")
        return ""

def process_environment(env_name: str, objective_prompt: str, gpt_handler: GPT4OHandler):
    """
    单独处理一个环境：检查图像文件、创建批处理任务、轮询状态并下载结果。
    """
    # 定位当前环境的图像文件夹及图像路径
    env_dir = os.path.join(IMAGES_ROOT_DIR, env_name)
    image1_path = os.path.join(env_dir, "image1.png")
    image2_path = os.path.join(env_dir, "image2.png")

    print(f"检查环境 {env_name}: 尝试查找 {image1_path} 和 {image2_path}")
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print(f"环境 {env_name} 缺少必要的图像文件，跳过。")
        return

    # 使用压缩函数减少图像数据量
    img1_base64 = compress_image_to_base64(image1_path)
    img2_base64 = compress_image_to_base64(image2_path)

    if not img1_base64 or not img2_base64:
        print(f"压缩或编码 {env_name} 的图像失败，跳过。")
        return

    user_prompt = single_round_template.format(
        img1=img1_base64,
        img2=img2_base64,
        objective=objective_prompt
    )

    messages = build_messages(user_prompt)

    # 创建以时间戳命名的目录，用于存储当前环境的批处理文件和结果
    base_output_dir = os.path.join("dataCollection", "Dataset", env_name)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    timestamp_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)

    # 为当前环境创建单个请求的批处理输入文件路径
    batch_file_name = os.path.join(timestamp_dir, "batch_input.jsonl")
    request = {
        "custom_id": env_name,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 2000
        }
    }

    # 将请求写入 .jsonl 文件
    with open(batch_file_name, "w", encoding='utf-8') as f:
        f.write(json.dumps(request, ensure_ascii=False) + "\n")

    # 上传文件并创建批处理任务
    try:
        file_id = gpt_handler.upload_batch_file(batch_file_name)
        print(f"{env_name}: 上传批处理文件成功，文件 ID: {file_id}")
        batch = gpt_handler.create_batch(file_id)
        print(f"{env_name}: 创建批处理任务成功，任务 ID: {batch.get('id')}")
    except Exception as e:
        print(f"{env_name}: 批处理创建失败：{e}")
        return

    batch_id = batch.get("id")
    if not batch_id:
        print(f"{env_name}: 未能获取批处理 ID，退出。")
        return

    # 轮询批处理任务状态
    while True:
        try:
            status = gpt_handler.check_batch_status(batch_id)
            print(f"{env_name}: 当前批处理状态: {status.get('status')}")
        except Exception as e:
            print(f"{env_name}: 检查批处理状态失败：{e}")
            return

        if status.get("status") == "completed":
            print(f"{env_name}: 批处理任务已完成！")
            break
        elif status.get("status") == "failed":
            print(f"{env_name}: 批处理任务失败！详情: {status}")
            return
        else:
            print(f"{env_name}: 批处理任务进行中，等待 60 秒后再次检查...")
            time.sleep(60)

    output_file_id = status.get("output_file_id")
    if output_file_id:
        try:
            result_file = os.path.join(timestamp_dir, "batch_output.jsonl")
            gpt_handler.download_batch_results(output_file_id, save_path=result_file)
            print(f"{env_name}: 批处理结果已下载到 {result_file}。")
        except Exception as e:
            print(f"{env_name}: 下载批处理结果失败：{e}")
    else:
        print(f"{env_name}: 未找到输出文件 ID。")

def main():
    gpt_handler = GPT4OHandler(API_URL, API_KEY, MODEL)

    # 依次处理每个环境
    for env_name, objective_prompt in objective_env_prompts.items():
        print(f"开始处理环境：{env_name}")
        process_environment(env_name, objective_prompt, gpt_handler)
        print(f"环境 {env_name} 处理完毕。\n")
        # 可根据需要添加延时，避免频繁请求
        time.sleep(5)

if __name__ == "__main__":
    main()
