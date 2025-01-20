# /home/star/Yanjun/RL-VLM-F/dataGeneration/multi_openai_manager.py
# -------------------------------------------------------------------------------
"""
multi_openai_manager.py

在并行、多 Key 调度的基础上，新增“resume_map”机制：
1) 若某些 chunk_file 已在之前的调用中成功得到 output_file，则无需再次提交API。
2) 支持 test_mode 与 official_mode 两种使用场景：
   - test_mode=True 时出现错误只做有限重试，用于检测 chunk_size 是否可行；
   - test_mode=False 时出现错误无限重试，但有 fail_count 上限保护。
3) 注释详尽，帮助后续维护。
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from openai_batch import OpenAIBatchHandler


class MultiOpenAIBatchManager:
    """
    MultiOpenAIBatchManager: 同时使用多个OpenAIBatchHandler(Key)进行并行处理,
    并支持 'resume' 机制(跳过已经完成处理的 chunk)，
    避免因API限制(24小时后才能继续)而重复调用已成功的chunk。

    特点:
      - test_mode=True: 用于EnvironmentProcessor的二分搜索/快速测试，出现错误只做有限重试后放弃。
      - test_mode=False: official模式, 出现错误无限重试+ fail_count上限保护，绝不跳过任何chunk除非resume发现已完成。
      - resume_map: chunk_file -> output_file。如果发现 chunk_file 已有相应 output_file，则跳过提交。
    """

    def __init__(
        self,
        test_mode: bool = False,
        max_tasks_per_key: int = 2,
        minimal_retries_for_test: int = 3,
        max_failures_per_chunk: int = 99,
        resume_map: Optional[Dict[str, str]] = None
    ):
        """
        :param test_mode: True => test_mode (最少重试)；False => official_mode (无限重试)。
        :param max_tasks_per_key: 每个Key可同时处理的 chunk 数量上限。
        :param minimal_retries_for_test: test_mode下，最多重试几次后仍失败则返回False。
        :param max_failures_per_chunk: official_mode下，若单chunk fail_count>=此值 => 抛异常退出。
        :param resume_map: 若不为 None，则表示 chunk_file->output_file 已经完成。可跳过这些 chunk。
        """
        self.test_mode = test_mode
        self.max_tasks_per_key = max_tasks_per_key
        self.minimal_retries_for_test = minimal_retries_for_test
        self.max_failures_per_chunk = max_failures_per_chunk
        self.resume_map: Dict[str, str] = resume_map if resume_map else {}

        self.handlers: List[OpenAIBatchHandler] = []
        self.running_jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> job_info
        self.results_map: Dict[str, str] = {}  # chunk_file -> output_file

    def load_handlers(self, api_keys: List[str]):
        """
        创建 OpenAIBatchHandler (每个Key一个), key_index=1...n
        """
        self.handlers.clear()
        for idx, key in enumerate(api_keys, start=1):
            h = OpenAIBatchHandler(api_key=key, key_index=idx)
            self.handlers.append(h)
        print(f"[MultiManager] (test_mode={self.test_mode}) 已加载 {len(self.handlers)} 个 Key.")

    def process_chunk_files_test(self, chunk_files: List[str], out_dir: str) -> bool:
        """
        test_mode=True:
          - 加载 resume_map, 跳过已完成 chunk
          - 最少重试, 失败后返回False(不会无限重试)
        :param chunk_files: 待处理 chunk 列表
        :param out_dir: 存放输出的目录
        :return: True=全部成功(含已resume)，False=任一chunk最终失败
        """
        print(f"[MultiManager-Test] 开始处理 {len(chunk_files)} 个 chunk. (test_mode=True)")
        if not self.handlers:
            print("[MultiManager-Test] 无可用 Key => 返回False")
            return False

        self.running_jobs.clear()
        self.results_map.clear()

        # 先将 resume_map 中已完成的记录放入 results_map
        for cf in chunk_files:
            if cf in self.resume_map:
                self.results_map[cf] = self.resume_map[cf]

        waiting_queue = [cf for cf in chunk_files if cf not in self.results_map]

        while waiting_queue or self.running_jobs:
            self._dispatch_jobs_test(waiting_queue, out_dir)
            ok = self._poll_jobs_test()
            if not ok:
                print("[MultiManager-Test] 发生错误 => 返回False")
                return False
            time.sleep(1)

        print("[MultiManager-Test] 全部 chunk(含resume) 成功完成 => True")
        return True

    def process_chunk_files_official(self, chunk_files: List[str], out_dir: str) -> List[str]:
        """
        test_mode=False(official mode):
          - 不会放弃任何chunk，但有 fail_count>=max_failures_per_chunk => 抛异常保护
          - 支持 resume_map 跳过已完成的 chunk
        :param chunk_files: 待处理文件列表
        :param out_dir: 存放输出目录
        :return: 与 chunk_files 对应的 output_file 列表
        """
        print(f"[MultiManager-Official] 开始处理 {len(chunk_files)} 个 chunk. (test_mode=False)")
        if not self.handlers:
            raise RuntimeError("[MultiManager-Official] 无可用Handler, 请先 load_handlers().")

        self.running_jobs.clear()
        self.results_map.clear()

        # 填入resume
        for cf in chunk_files:
            if cf in self.resume_map:
                self.results_map[cf] = self.resume_map[cf]

        waiting_queue = [cf for cf in chunk_files if cf not in self.results_map]

        while waiting_queue or self.running_jobs:
            self._dispatch_jobs_official(waiting_queue, out_dir)
            self._poll_jobs_official()
            time.sleep(2)

        # 整理输出(保证与 chunk_files 顺序对应)
        outs = []
        for cf in chunk_files:
            outs.append(self.results_map.get(cf, ""))  # 应该都能拿到
        print("[MultiManager-Official] 全部 chunk(含已resume) 成功完成.")
        return outs

    # ------------------------- Test Mode专用逻辑 ------------------------------
    def _dispatch_jobs_test(self, waiting_queue: List[str], out_dir: str):
        """
        test_mode下，将待处理chunk分配给空闲的Key
        """
        for h in self.handlers:
            active_count = sum(
                1 for job in self.running_jobs.values()
                if job["handler"]==h and job["status"]=="running"
            )
            while active_count<self.max_tasks_per_key and waiting_queue:
                cf = waiting_queue.pop(0)
                job_id = f"test_job_{time.time()}_{os.path.basename(cf)}"
                outp = os.path.join(out_dir, os.path.splitext(os.path.basename(cf))[0]+"_output.jsonl")
                job_info = {
                    "handler": h,
                    "chunk_file": cf,
                    "output_file": outp,
                    "status": "running",
                    "fail_count": 0,
                    "batch_id": ""
                }
                self.running_jobs[job_id] = job_info
                self._submit_single_test(job_id, job_info)
                active_count+=1

    def _submit_single_test(self, job_id: str, info: Dict[str,Any]):
        """
        test_mode: 上传+create_batch(有限重试)；若失败则fail_count++
        """
        h = info["handler"]
        cf = info["chunk_file"]
        try:
            fid = h.upload_batch_file(cf)
        except Exception as e:
            info["fail_count"]+=1
            print(f"[MultiManager-Test] upload_batch_file异常: {type(e).__name__}, {e}")
            if info["fail_count"]>=self.minimal_retries_for_test:
                info["status"] = "error"
            return

        # create_batch
        try:
            binfo = h.create_batch(fid)
            b_id = binfo.get("id","")
            if not b_id:
                info["fail_count"]+=1
                if info["fail_count"]>=self.minimal_retries_for_test:
                    info["status"] = "error"
                return
            info["batch_id"] = b_id
        except Exception as e2:
            info["fail_count"]+=1
            print(f"[MultiManager-Test] create_batch异常: {type(e2).__name__}, {e2}")
            if info["fail_count"]>=self.minimal_retries_for_test:
                info["status"] = "error"

    def _poll_jobs_test(self) -> bool:
        """
        test_mode: 轮询 running job => check_batch_status => completed则下载；failed/异常 => fail_count++ => 超过则标记error => 返回False
        """
        finished=[]
        for jid, job in list(self.running_jobs.items()):
            if job["status"]=="error":
                return False
            if job["status"]!="running":
                continue

            b_id = job["batch_id"]
            if not b_id:
                # create_batch尚未成功
                continue
            h = job["handler"]

            try:
                st = h.check_batch_status(b_id)
            except Exception as cex:
                job["fail_count"]+=1
                print(f"[MultiManager-Test] check_batch_status异常: {type(cex).__name__}, {cex}")
                if job["fail_count"]>=self.minimal_retries_for_test:
                    job["status"]="error"
                continue

            st_status = st.get("status","")
            if st_status=="completed":
                # download
                ofid = st.get("output_file_id","")
                if not ofid:
                    job["status"] = "error"
                    return False
                try:
                    h.download_batch_results(ofid, job["output_file"])
                    self.results_map[job["chunk_file"]] = job["output_file"]
                    finished.append(jid)
                except Exception as dl_ex:
                    job["fail_count"]+=1
                    if job["fail_count"]>=self.minimal_retries_for_test:
                        job["status"]="error"
            elif st_status=="failed":
                job["fail_count"]+=1
                print(f"[MultiManager-Test] job_id={jid}, batch失败 => fail_count={job['fail_count']}")
                if job["fail_count"]>=self.minimal_retries_for_test:
                    job["status"]="error"
            elif st_status in ("queued","running","validating","in_progress"):
                pass
            else:
                # unknown
                job["fail_count"]+=1
                if job["fail_count"]>=self.minimal_retries_for_test:
                    job["status"]="error"

        for fj in finished:
            del self.running_jobs[fj]

        for jinfo in self.running_jobs.values():
            if jinfo["status"]=="error":
                return False
        return True

    # ----------------------- Official Mode专用逻辑 ----------------------------
    def _dispatch_jobs_official(self, waiting_queue: List[str], out_dir: str):
        """
        official mode: 分配等待中的 chunk 给空闲 Key，无限重试。
        """
        for h in self.handlers:
            active_count = sum(
                1 for j in self.running_jobs.values()
                if j["handler"]==h and j["status"]=="running"
            )
            while active_count<self.max_tasks_per_key and waiting_queue:
                cf = waiting_queue.pop(0)
                job_id = f"official_job_{time.time()}_{os.path.basename(cf)}"
                outp = os.path.join(out_dir, os.path.splitext(os.path.basename(cf))[0]+"_output.jsonl")
                job_info = {
                    "handler": h,
                    "chunk_file": cf,
                    "output_file": outp,
                    "status": "running",
                    "fail_count": 0,
                    "batch_id": ""
                }
                self.running_jobs[job_id] = job_info
                self._submit_single_official(job_id, job_info)
                active_count+=1

    def _submit_single_official(self, job_id: str, info: Dict[str,Any]):
        """
        official mode: upload + create_batch,若失败无限重试(带fail_count保护)
        """
        h = info["handler"]
        cf = info["chunk_file"]

        # upload(无限重试)
        while True:
            try:
                fid = h.upload_batch_file(cf)
                break
            except Exception as e:
                info["fail_count"]+=1
                print(f"[MultiManager-Official] upload_batch_file异常: {type(e).__name__}, {e}")
                if info["fail_count"]>=self.max_failures_per_chunk:
                    raise RuntimeError(
                        f"[MultiManager-Official] job_id={job_id},fail_count={info['fail_count']},放弃."
                    )
                time.sleep(30)

        # create_batch(无限重试)
        while True:
            try:
                binfo = h.create_batch(fid)
                b_id = binfo.get("id","")
                if b_id:
                    info["batch_id"] = b_id
                    return
                else:
                    info["fail_count"]+=1
                    if info["fail_count"]>=self.max_failures_per_chunk:
                        raise RuntimeError(
                            f"[MultiManager-Official] job_id={job_id},fail_count={info['fail_count']},放弃."
                        )
                    time.sleep(30)
            except Exception as e2:
                info["fail_count"]+=1
                print(f"[MultiManager-Official] create_batch异常: {type(e2).__name__}, {e2}")
                if info["fail_count"]>=self.max_failures_per_chunk:
                    raise RuntimeError(
                        f"[MultiManager-Official] job_id={job_id},fail_count={info['fail_count']},放弃."
                    )
                time.sleep(30)

    def _poll_jobs_official(self):
        """
        official mode: 检查batch状态, completed则下载 => success
                       failed => retry create_batch
        """
        finished=[]
        for jid, info in list(self.running_jobs.items()):
            if info["status"]!="running":
                continue
            b_id = info["batch_id"]
            if not b_id:
                # create_batch尚未成功
                continue

            h = info["handler"]
            cf = info["chunk_file"]
            outp = info["output_file"]

            # check status(无限重试, fail_count>=max=>抛异常)
            st = None
            while True:
                try:
                    st = h.check_batch_status(b_id)
                    break
                except Exception as cex:
                    info["fail_count"]+=1
                    print(f"[MultiManager-Official] check_batch_status异常:{type(cex).__name__},{cex}")
                    if info["fail_count"]>=self.max_failures_per_chunk:
                        raise RuntimeError(
                            f"[MultiManager-Official] job_id={jid},fail_count={info['fail_count']}=>放弃."
                        )
                    time.sleep(30)

            st_status = st.get("status","")
            if st_status=="completed":
                # 下载(无限重试)
                ofid = st.get("output_file_id","")
                if not ofid:
                    info["fail_count"]+=1
                    if info["fail_count"]>=self.max_failures_per_chunk:
                        raise RuntimeError(
                            f"[MultiManager-Official] job_id={jid},无ofid且fail_count过多,放弃."
                        )
                    continue
                self._download_with_retry_official(h, ofid, outp, jid, info)
                self.results_map[cf] = outp
                finished.append(jid)
            elif st_status=="failed":
                print(f"[MultiManager-Official] job_id={jid} => batch失败 => retry create_batch")
                info["fail_count"]+=1
                if info["fail_count"]>=self.max_failures_per_chunk:
                    raise RuntimeError(
                        f"[MultiManager-Official] job_id={jid},fail_count={info['fail_count']}=>放弃."
                    )
                self._retry_create_batch_official(h, st, jid, info)
            elif st_status in ("queued","running","validating","in_progress"):
                pass
            else:
                # unknown => 当做failed
                info["fail_count"]+=1
                print(f"[MultiManager-Official] job_id={jid},未知状态={st_status}")
                if info["fail_count"]>=self.max_failures_per_chunk:
                    raise RuntimeError(
                        f"[MultiManager-Official] job_id={jid},fail_count={info['fail_count']}=>放弃"
                    )
                self._retry_create_batch_official(h, st, jid, info)

        for fj in finished:
            del self.running_jobs[fj]

    def _download_with_retry_official(
        self,
        h: OpenAIBatchHandler,
        output_file_id: str,
        out_path: str,
        jid: str,
        info: Dict[str,Any]
    ):
        """
        下载(无限重试, fail_count>=max=>抛异常)
        """
        while True:
            try:
                h.download_batch_results(output_file_id, out_path)
                print(f"[MultiManager-Official] job_id={jid}, chunk={info['chunk_file']} => 完成下载 => {out_path}")
                return
            except Exception as e:
                info["fail_count"]+=1
                print(f"[MultiManager-Official] 下载异常: {type(e).__name__}, {e}")
                if info["fail_count"]>=self.max_failures_per_chunk:
                    raise RuntimeError(
                        f"[MultiManager-Official] job_id={jid},下载失败过多,fCount={info['fail_count']}=>放弃."
                    )
                time.sleep(30)

    def _retry_create_batch_official(
        self,
        h:OpenAIBatchHandler,
        st:Dict[str,Any],
        jid:str,
        info:Dict[str,Any]
    ):
        """
        official_mode: batch=failed => re-create batch(无限重试). fail_count>=max =>抛异常
        """
        while True:
            if info["fail_count"]>=self.max_failures_per_chunk:
                raise RuntimeError(
                    f"[MultiManager-Official] job_id={jid},fail_count={info['fail_count']}=>放弃."
                )
            try:
                h.wait_for_queue_space()
                binfo = h.create_batch(st.get("input_file_id",""))
                new_b_id = binfo.get("id","")
                if new_b_id:
                    info["batch_id"] = new_b_id
                    print(f"[MultiManager-Official] job_id={jid} => 重试 create_batch成功 => {new_b_id}")
                    return
                else:
                    info["fail_count"]+=1
                    if info["fail_count"]>=self.max_failures_per_chunk:
                        raise RuntimeError(
                            f"[MultiManager-Official] job_id={jid},fail_count={info['fail_count']}=>放弃."
                        )
                    time.sleep(30)
            except Exception as e2:
                info["fail_count"]+=1
                print(f"[MultiManager-Official] 重试create_batch异常: {type(e2).__name__}, {e2}")
                if info["fail_count"]>=self.max_failures_per_chunk:
                    raise RuntimeError(
                        f"[MultiManager-Official] job_id={jid},fail_count={info['fail_count']}=>放弃."
                    )
                time.sleep(30)
