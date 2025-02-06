# -*- coding: utf-8 -*-
"""
render_point_cloud_v4.py

功能：
  1) 递归或非递归地扫描给定文件夹/文件列表中的所有 .npy 点云文件；
  2) 加载并渲染点云至 .jpg 图像，可指定相机方向/多视图、视场角、画布大小、背景色等；
  3) 可以包含颜色 (若 .npy 中后3列为RGB)；
  4) 使用 Open3D 的离屏渲染器 (OffscreenRenderer) 进行可视化输出；
  5) 对异常做好捕获，并保证后续文件能够继续处理；
  6) 保持输入文件夹的子目录结构，确保输出图像对应于输入的子目录，并去除文件名中的 '_part_X' 部分。

依赖：
  - open3d>=0.15
  - numpy
  - Pillow
  - pathlib
  - logging

示例用法：
  1) 直接运行脚本：
     python render_point_cloud_v4.py
  2) 或在其他脚本中引入本类：
     from render_point_cloud_v4 import PointCloudRenderer
     然后按需实例化并调用 process_all()。

作者：ChatGPT 改进版本
日期：2025/04/27
"""

import os
import sys
import glob
import re
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
import logging
import traceback

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO，可以根据需要调整为DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 将日志输出到标准输出
    ]
)
logger = logging.getLogger(__name__)


class PointCloudRenderer:
    """
    PointCloudRenderer 用于渲染 .npy 点云文件并输出 .jpg 图像。

    主要特性：
    1) 可递归检索子目录中的所有 .npy 文件；
    2) 支持多方向相机视图：camera_directions=[(x,y,z), ...]；
    3) 自动计算相机位置，基于点云包围盒与 FOV，确保拍摄到全部点云；
    4) 可在点云中加入颜色 (RGB)，若 .npy 数据列 >= 6，则视为 [3:6] 为颜色信息；
    5) 可配置背景颜色、图像宽高、是否递归搜索、是否使用颜色等；
    6) 使用 Open3D 的离屏渲染器进行可视化输出，保存为 .jpg；
    7) 保持输入文件夹的子目录结构，确保输出图像对应于输入的子目录，并去除文件名中的 '_part_X' 部分。

    用法：
        renderer = PointCloudRenderer(
            input_paths=["some/folder", "some/other/pointcloud.npy"],
            output_folder="rendered_imgs",
            width=800,
            height=600,
            camera_up=[0, 1, 0],
            camera_directions=[[0,0,1], [1,0,1]],
            fov=60.0,
            background_color=[1,1,1,1],
            use_color=True,
            recursive=True
        )
        renderer.process_all()

    主要方法：
      - process_all()： 依次处理 input_paths 里的每个文件或文件夹
      - process_single_path(path, base_path)： 处理单个文件/文件夹，base_path 用于相对路径计算
      - get_npy_files_in_dir(dir_path)： 递归或非递归获取 .npy 文件列表
      - load_point_cloud(npy_file)： 加载 .npy 为 Open3D 点云
      - compute_camera_for_direction(pcd, direction)： 基于包围盒 & 方向计算相机位置
      - render_single(pcd, output_jpg, camera_location, camera_target)： 离屏渲染并保存图像
      - clean_filename(filename)： 去除文件名中的 '_part_X' 部分

    """

    def __init__(
        self,
        input_paths,
        output_folder,
        width=800,
        height=600,
        camera_up=[0, 1, 0],
        camera_directions=None,
        fov=60.0,
        background_color=[1.0, 1.0, 1.0, 1.0],
        use_color=True,
        recursive=True
    ):
        """
        初始化 PointCloudRenderer 实例。

        参数：
          input_paths: str 或者 list[str], 待处理的文件或文件夹路径
          output_folder: str, 输出图像的根目录
          width, height: int, 渲染图像宽高
          camera_up: list[float], 相机上向量 (如 [0,1,0])
          camera_directions: list[list[float]], 相机朝向列表，如 [[1,0,0], [0,0,1]]，可指定多个方向
          fov: float，视场角(度数)，如 60.0
          background_color: list[float], [r,g,b,a], 背景颜色(0-1)
          use_color: bool, 是否从点云中读取RGB (若 .npy 列>=6)
          recursive: bool, 是否在文件夹中递归搜索 .npy
        """
        # 将 input_paths 统一转换为 Path 对象列表
        if isinstance(input_paths, str):
            self.input_paths = [Path(input_paths)]
        elif isinstance(input_paths, list):
            self.input_paths = [Path(p) for p in input_paths]
        else:
            raise ValueError("input_paths 需为字符串或字符串列表。")

        self.output_folder = Path(output_folder)
        self.width = width
        self.height = height
        self.camera_up = np.array(camera_up, dtype=np.float64)
        if self.camera_up.shape != (3,):
            raise ValueError("camera_up 必须是包含三个元素的列表。")
        self.camera_directions = camera_directions if camera_directions else [[0, 0, 1]]
        self.fov = fov
        if len(background_color) not in [3, 4]:
            raise ValueError("background_color 必须是包含3或4个元素的列表。")
        self.background_color = background_color
        self.use_color = use_color
        self.recursive = recursive

        # 如果输出文件夹不存在，则创建
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # 检查 Open3D 是否支持 OffscreenRenderer
        if not hasattr(o3d.visualization.rendering, "OffscreenRenderer"):
            raise ImportError("Open3D 版本过低，不支持 OffscreenRenderer。请升级 Open3D 至 >=0.15 版本。")

        logger.info("PointCloudRenderer 初始化完成。")

    def get_npy_files_in_dir(self, dir_path):
        """
        返回指定目录下的所有 .npy 文件列表，支持递归搜索。

        参数：
          dir_path: Path, 目录路径

        返回：
          list[Path]: .npy 文件的 Path 对象列表
        """
        if self.recursive:
            npy_files = list(dir_path.rglob("*.npy"))
        else:
            npy_files = list(dir_path.glob("*.npy"))
        logger.debug(f"在目录 {dir_path} 中找到 {len(npy_files)} 个 .npy 文件。")
        return npy_files

    def clean_filename(self, filename):
        """
        清理文件名，去除 '_part_X' 部分。

        例如:
          'pc_001215_bin_1_r_3.38_t_1216.00_emb_801d8a58_part_0' => 'pc_001215_bin_1_r_3.38_t_1216.00_emb_801d8a58'

        参数：
          filename: str, 原始文件名（不含扩展名）

        返回：
          str: 清理后的文件名
        """
        # 使用正则表达式匹配 '_part_' 后的数字，并去除该部分
        cleaned = re.sub(r'_part_\d+$', '', filename)
        if cleaned != filename:
            logger.debug(f"清理文件名: '{filename}' => '{cleaned}'")
        return cleaned

    def load_point_cloud(self, npy_file):
        """
        加载 .npy 格式点云文件并返回 Open3D 点云对象 (o3d.geometry.PointCloud)。

        参数：
          npy_file: Path, .npy 文件路径

        返回：
          o3d.geometry.PointCloud: 加载的点云对象

        异常：
          FileNotFoundError: 文件不存在
          ValueError: 点云数据格式不正确
        """
        if not npy_file.exists():
            raise FileNotFoundError(f"文件 {npy_file} 不存在。")

        try:
            data = np.load(npy_file)
        except Exception as e:
            raise ValueError(f"无法加载文件 {npy_file}：{e}")

        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"点云数据 {npy_file} 至少需要3维坐标信息 (XYZ)。")

        # 构造 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # 前3列 => 坐标

        # 若 use_color=True 且 data列>=6，自动读 [3:6] 作为 RGB
        if self.use_color and data.shape[1] >= 6:
            colors = data[:, 3:6].astype(np.float64)
            # 如果颜色值大于1，则认为是 0-255 => 归一化到[0,1]
            if colors.max() > 1.0:
                colors /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        logger.debug(f"加载点云文件 {npy_file} 成功，点数: {len(pcd.points)}。")
        return pcd

    def compute_camera_for_direction(self, pcd, direction):
        """
        根据给定点云 pcd 和方向 direction，计算相机位置与目标点(相机中心对准处)。

        算法：
          1) 获取 pcd 的 oriented_bounding_box => 得到其 center, extent
          2) 取 extent 中最大的值 max_extent
          3) 基于视场角 fov，估算拍摄距离 distance
          4) 方向向量 direction 归一化后 => camera_location = center + direction * distance

        参数：
          pcd: o3d.geometry.PointCloud, 点云对象
          direction: list[float], 相机方向向量

        返回：
          tuple: (camera_location (np.ndarray), camera_target (np.ndarray))

        异常：
          ValueError: direction 为零向量
        """
        obb = pcd.get_oriented_bounding_box()
        center = obb.center
        extent = obb.extent
        max_extent = max(extent)

        # 若点云尺寸极小，可做微量放大，避免相机太靠近
        if max_extent < 1e-9:
            max_extent = 1e-3

        # distance = (max_extent/2) / tan(fov/2) * 1.5
        distance = (max_extent / 2.0) / np.tan(np.deg2rad(self.fov / 2.0)) * 1.5

        # 归一化 direction
        direction = np.array(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError(f"camera_direction={direction} 不能为零向量。")
        direction /= norm

        # 计算相机位置
        camera_location = center + direction * distance
        camera_target = center

        logger.debug(f"计算相机位置: {camera_location}, 目标点: {camera_target}, 方向: {direction}, 距离: {distance:.4f}")
        return camera_location, camera_target

    def render_single(self, pcd, output_jpg, camera_location, camera_target):
        """
        渲染单个点云到 output_jpg，使用 Open3D OffscreenRenderer。

        参数：
          pcd: o3d.geometry.PointCloud, 点云对象
          output_jpg: Path, 输出的 JPG 文件路径
          camera_location: np.ndarray, 相机位置
          camera_target: np.ndarray, 相机目标点

        异常：
          Exception: 渲染过程中可能出现的任何异常
        """
        try:
            # 创建离屏渲染器
            renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
            scene = renderer.scene

            # 配置材质
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"  # 无光照
            material.point_size = 2.0

            # 添加点云到场景
            scene.add_geometry("pointcloud", pcd, material)
            scene.set_background(self.background_color)

            # 设置相机视角
            renderer.setup_camera(self.fov, camera_target, camera_location, self.camera_up.tolist())

            # 渲染
            img_o3d = renderer.render_to_image()
            if img_o3d is None:
                raise RuntimeError("渲染图像失败，返回图像为空。")

            img_data = np.asarray(img_o3d)
            pil_img = Image.fromarray(img_data)

            # 保存 JPG
            output_jpg.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(output_jpg, format="JPEG", quality=100)

            logger.info(f"成功渲染并保存图像至 {output_jpg}")

        except Exception as e:
            logger.error(f"渲染过程中出现错误: {e}")
            logger.debug(traceback.format_exc())
        finally:
            # 删除渲染器对象，释放资源
            if 'renderer' in locals():
                del renderer

    def process_single_path(self, path, base_path):
        """
        处理单个路径：
          1) 若 path 是单个 .npy 文件，则仅处理该文件；
          2) 若 path 是文件夹，则检索其中的所有 .npy 文件；
          3) 若 path 非法，则跳过。

        参数：
          path: Path, 待处理的文件或文件夹路径
          base_path: Path, 输入路径的基准路径，用于计算相对路径
        """
        if path.is_file():
            if path.suffix.lower() != ".npy":
                logger.warning(f"跳过无效文件: {path} (非 .npy 格式)")
                return
            npy_files = [path]
            # 如果输入路径是文件，则基准路径为文件所在目录
            base_path = path.parent
        elif path.is_dir():
            # 如果输入路径是目录，尝试设定 base_path 为 path / 'data'，如果存在
            data_subdir = path / 'data'
            if data_subdir.exists() and data_subdir.is_dir():
                base_path = data_subdir
                logger.debug(f"设定基准路径为 {base_path}（包含 'data' 子目录）")
            else:
                base_path = path
                logger.debug(f"设定基准路径为 {base_path}（不包含 'data' 子目录）")
            npy_files = self.get_npy_files_in_dir(path)
            if not npy_files:
                logger.warning(f"文件夹 {path} 中未找到任何 .npy 文件。")
                return
        else:
            logger.warning(f"输入路径 {path} 不存在或非文件/文件夹，跳过。")
            return

        for npy_file in npy_files:
            try:
                # 加载点云
                pcd = self.load_point_cloud(npy_file)
            except Exception as e:
                logger.error(f"加载文件 {npy_file} 时发生错误：{e}")
                logger.debug(traceback.format_exc())
                continue

            # 计算相对路径以保持目录结构
            try:
                relative_dir = npy_file.parent.relative_to(base_path)
                logger.debug(f"相对目录为：{relative_dir}")
            except ValueError:
                # 如果无法计算相对路径，则使用输出文件夹根目录
                relative_dir = Path('.')
                logger.debug(f"无法计算相对路径，使用根目录。")

            # 定义任务输出文件夹
            task_output_folder = self.output_folder / path.name / relative_dir
            task_output_folder.mkdir(parents=True, exist_ok=True)

            # 基本文件名，去除 '_part_X' 部分
            base_filename = self.clean_filename(npy_file.stem)

            # 对每个指定方向都渲染一张图
            for idx, direction in enumerate(self.camera_directions):
                try:
                    cam_loc, cam_target = self.compute_camera_for_direction(pcd, direction)
                except Exception as e:
                    logger.error(f"计算相机方向 {direction} 时发生错误：{e}")
                    logger.debug(traceback.format_exc())
                    continue

                # 定义输出图片文件名
                filename = f"{base_filename}_view{idx+1}.jpg"
                output_jpg = task_output_folder / filename

                # 执行渲染
                self.render_single(pcd, output_jpg, cam_loc, cam_target)

    def process_all(self):
        """
        对 input_paths 中的每个路径依次调用 process_single_path。
        """
        for path in self.input_paths:
            if not path.exists():
                logger.warning(f"输入路径 {path} 不存在，跳过。")
                continue
            logger.info(f"开始处理路径: {path}")
            self.process_single_path(path, path.parent if path.is_file() else path)
        logger.info("所有路径处理完成。")


if __name__ == "__main__":
    # ============== 配置参数 ==============
    # 多个输入路径示例: 每个路径都可能是单个文件或一个文件夹
    input_paths = [
        # "tests",
        # "data/pointclouds/metaworld_soccer-v2",
        "data/DensePointClouds/metaworld_soccer-v2",
        # "data/DensePointClouds/metaworld_drawer-open-v2",
        # "data/DensePointClouds/metaworld_door-open-v2",
        # "data/DensePointClouds/metaworld_disassemble-v2",
        # "data/DensePointClouds/metaworld_handle-pull-side-v2",
        # "data/DensePointClouds/metaworld_peg-insert-side-v2"
    ]

    output_folder = "data/renderPointCloud"
    # output_folder = "tests"  # 输出文件夹路径，建议修改为合适的路径
    width = 400  # 渲染图像宽度
    height = 400  # 渲染图像高度
    camera_up = [0, 1, 0]  # 相机上向量
    camera_directions = [
        # metaworld_soccer-v2, metaworld_drawer-open-v2, metaworld_door-open-v2, metaworld_disassemble-v2
        [1, -1, 1],
        [-1, -1, 1],
        # metaworld_handle-pull-side-v2
        # [0, 0, 1],
        # [1, 0, 1],
        # metaworld_peg-insert-side-v2
        # [-1, 0, 0],
        # [1, 0, 0],
    ]
    fov = 60.0  # 视场角（度）
    background_color = [1.0, 1.0, 1.0, 1.0]  # 背景颜色 (RGBA)，范围 [0,1]
    use_color = True  # 是否使用点云颜色
    recursive = True  # 是否递归搜索子目录
    # ======================================

    # 实例化渲染器
    renderer = PointCloudRenderer(
        input_paths=input_paths,
        output_folder=output_folder,
        width=width,
        height=height,
        camera_up=camera_up,
        camera_directions=camera_directions,
        fov=fov,
        background_color=background_color,
        use_color=use_color,
        recursive=recursive
    )

    # 开始处理
    renderer.process_all()
