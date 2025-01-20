# /home/star/Yanjun/RL-VLM-F/dataGeneration/direct_openai_manager.py
# -------------------------------------------------------------------------------
"""
direct_openai_manager.py

实现 DirectOpenAIManager 类，管理多个 DirectOpenAIHandler，
以并行方式直接调用 OpenAI API 而不使用 Batch API。
"""

import os
import time
import json
import threading
from queue import Queue
from typing import List, Dict
from direct_openai_handler import DirectOpenAIHandler
from config import OPENAI_API_KEYS

class DirectOpenAIManager:
    """
    管理多个 DirectOpenAIHandler 并行处理任务。
    实现类似 MultiOpenAIBatchManager 的调度，但直接调用 API。
    """

    def __init__(self, max_workers_per_key: int = 2):
        """
        :param max_workers_per_key: 每个 Key 同时工作的最大线程数
        """
        self.max_workers_per_key = max_workers_per_key
        self.handlers: List[DirectOpenAIHandler] = []
        self.task_queue = Queue()
        self.results: Dict[str, dict] = {}  # custom_id -> API response

    def load_handlers(self, api_keys: List[str]):
        """
        加载一批 API Key，创建对应的 DirectOpenAIHandler
        """
        self.handlers = [DirectOpenAIHandler(key, idx) for idx, key in enumerate(api_keys, start=1)]
        print(f"[DirectManager] 已加载 {len(self.handlers)} 个 DirectOpenAIHandler.")

    def add_tasks(self, tasks: List[Dict]):
        """
        添加待处理任务到队列，任务格式包含 custom_id 和 body(其中有 user_content)等信息。
        """
        for task in tasks:
            self.task_queue.put(task)

    def worker(self, handler: DirectOpenAIHandler):
        """
        每个线程使用一个 handler 从队列中取任务并处理。
        处理完成后，将结果记录到 self.results。
        若出现异常，则将任务放回队列等待下一次处理(无限重试)。
        """
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
            except:
                break
            custom_id = task.get("custom_id")
            user_content = task["body"]["messages"][-1]["content"]

            try:
                response = handler.process_pair(user_content, max_tokens=task["body"].get("max_tokens", 2000))
                self.results[custom_id] = response
                print(f"[{handler.name_tag()}] 成功处理任务 {custom_id}")
            except Exception as e:
                print(f"[{handler.name_tag()}] 处理任务 {custom_id} 时发生异常: {e}, 重回队列后稍后重试.")
                self.task_queue.put(task)
                time.sleep(5)
            finally:
                self.task_queue.task_done()

    def process_tasks(self, tasks: List[Dict], resume_file: str = None) -> Dict[str, dict]:
        """
        并行处理所有任务，直到队列为空。支持从 resume_file 中加载已完成的结果、跳过重复任务。

        :param tasks: 原始任务列表(每个元素带有 custom_id )
        :param resume_file: 若不为空，则从该文件中加载已有结果并跳过。
        :return: 最终的结果 custom_id -> response
        """
        # 如果传入了上次运行的结果文件，则先加载已完成的 custom_id
        skip_ids = set()
        if resume_file and os.path.isfile(resume_file):
            with open(resume_file, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        obj = json.loads(line)
                        # 这里假定 obj = {"custom_id": ..., "response": ...}
                        c_id = obj.get("custom_id")
                        if c_id:
                            # 将现有结果存入 self.results，后续跳过
                            self.results[c_id] = obj.get("response", {})
                            skip_ids.add(c_id)
                    except:
                        pass
            print(f"[DirectManager] 已从 {resume_file} 中加载 {len(skip_ids)} 条已有结果.")

        # 仅将尚未处理的任务加入队列
        pending_tasks = [t for t in tasks if t.get("custom_id") not in skip_ids]
        print(f"[DirectManager] 共接收 {len(tasks)} 条任务，其中跳过 {len(skip_ids)} 条, 剩余 {len(pending_tasks)} 条待处理.")
        self.add_tasks(pending_tasks)

        # 启动多线程 worker
        threads = []
        for handler in self.handlers:
            for _ in range(self.max_workers_per_key):
                t = threading.Thread(target=self.worker, args=(handler,))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()
        print("[DirectManager] 所有任务已处理完毕。")

        return self.results
