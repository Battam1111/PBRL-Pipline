#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gcs_helper.py

本模块提供与 Google Cloud Storage 交互的辅助函数，
主要用于将本地生成的 JSONL 任务请求文件上传至指定的 GCS 桶，
以便 Gemini 批量预测作业调用。

要求：
  - 安装 google-cloud-storage 包：pip install google-cloud-storage
  - 环境中已配置好 Google Cloud 的认证（例如 gcloud auth application-default login）
"""

import os
from google.cloud import storage

def upload_file_to_gcs(local_file: str, bucket_name: str, destination_blob_name: str) -> str:
    """
    上传本地文件至指定 GCS 桶，并返回该文件在 GCS 上的 URI。

    参数：
      local_file: 本地文件路径。
      bucket_name: GCS 桶名称。
      destination_blob_name: 文件在桶中的目标路径（包含文件名）。
    返回：
      文件的 GCS URI，格式为 gs://bucket_name/destination_blob_name
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)
    return f"gs://{bucket_name}/{destination_blob_name}"