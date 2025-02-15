# -*- coding: utf-8 -*-
"""
render_point_cloud_v4.py

功能：
  1) 递归或非递归地扫描给定文件夹/文件列表中的所有 .npy 点云文件；
  2) 加载并渲染点云至 .jpg 图像，可指定相机方向/多视图、视场角、画布大小、背景色等；
  3) 支持点云包含颜色信息（若 .npy 数据中后3列为RGB）；
  4) 使用 Open3D 的离屏渲染器 (OffscreenRenderer) 进行可视化输出；
  5) 对异常进行捕获，保证后续文件继续处理；
  6) 保持输入文件夹的子目录结构，输出图像文件名去除 '_part_X' 部分；
  7) **改进相机参数计算**：采用鲁棒统计方法（中位数+指定百分位数）确定点云“核心区域”，计算出最接近且统一的相机距离，既能确保捕捉点云核心，又避免因离群点导致镜头忽远忽近。

依赖：
  - open3d>=0.15
  - numpy
  - Pillow
  - pathlib
  - logging

示例用法：
  1) 直接运行脚本：
     python render_point_cloud_v4.py
  2) 或在其他脚本中引入：
     from render_point_cloud_v4 import PointCloudRenderer
     实例化并调用 process_all()。

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

# 配置日志记录：日志级别可根据需要调整（DEBUG/INFO）
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PointCloudRenderer:
    """
    PointCloudRenderer 用于渲染 .npy 点云文件并输出 .jpg 图像。
    
    主要特性：
      1) 可递归检索子目录中的所有 .npy 文件；
      2) 支持多方向相机视图：camera_directions=[(x,y,z), ...]；
      3) 采用基于点云数据的鲁棒统计方法计算点云中心和分布半径，
         从而在保证能够拍摄到点云核心的前提下，尽可能拉近镜头，
         且所有渲染出来的图像使用相同的相机距离（即镜头位置一致）；
      4) 支持点云内含颜色（若 .npy 数据列数>=6，则 [3:6] 为RGB）；
      5) 配置背景色、图像尺寸、是否递归搜索、是否使用颜色等；
      6) 使用 Open3D 的离屏渲染器进行可视化输出，保存为 .jpg 文件；
      7) 保持输入文件夹的子目录结构，输出文件名去除 '_part_X' 部分；
      8) 可选地限制相机距离上限（max_distance），防止因离群点导致镜头过远。

    用法示例：
        renderer = PointCloudRenderer(
            input_paths=["some/folder", "some/other/pointcloud.npy"],
            output_folder="rendered_imgs",
            width=800,
            height=600,
            camera_up=[0, 1, 0],
            camera_directions=[[0, 0, 1], [1, 0, 1]],
            fov=60.0,
            background_color=[1, 1, 1, 1],
            use_color=True,
            recursive=True,
            robust_percentile=95,   # 采用95百分位作为鲁棒统计（默认值）
            zoom_margin=1.1,        # 缩放边缘余量（默认1.1）
            max_distance=None       # 可选：最大允许相机距离（默认无限制）
        )
        renderer.process_all()

    主要方法：
      - process_all()：遍历 input_paths 中的每个路径，依次处理；
      - process_single_path(path, base_path)：处理单个文件或文件夹；
      - get_npy_files_in_dir(dir_path)：获取指定目录下的所有 .npy 文件（支持递归）；
      - load_point_cloud(npy_file)：加载 .npy 点云数据为 Open3D 点云对象；
      - compute_camera_for_direction(pcd, direction)：基于点云鲁棒统计方法计算相机位置和目标点；
      - render_single(pcd, output_jpg, camera_location, camera_target)：离屏渲染并保存图像；
      - clean_filename(filename)：清理文件名，去除 '_part_X' 部分；
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
        recursive=True,
        robust_percentile=95,
        zoom_margin=1.1,
        max_distance=None
    ):
        """
        初始化 PointCloudRenderer 实例。

        参数：
          input_paths: str 或 list[str]，待处理的文件或文件夹路径；
          output_folder: str，输出图像的根目录；
          width, height: int，渲染图像宽度和高度；
          camera_up: list[float]，相机上向量（例如 [0,1,0]）；
          camera_directions: list[list[float]]，相机观察方向列表，如 [[1,0,0], [0,0,1]]；
                          若未指定，则默认使用 [[0,0,1]]；
          fov: float，视场角（单位：度），例如 60.0；
          background_color: list[float]，[r,g,b,a]，背景颜色，取值范围 [0,1]；
          use_color: bool，是否使用点云中的RGB颜色（若 .npy 数据列>=6）；
          recursive: bool，是否在文件夹中递归搜索 .npy 文件；
          robust_percentile: int，计算点云分布时采用的百分位数（默认95），用于忽略离群点；
          zoom_margin: float，镜头边缘余量因子（默认1.1），在保证捕捉核心区域的同时尽可能拉近镜头；
          max_distance: float 或 None，若不为 None，则为相机允许的最大距离；
        """
        # 将 input_paths 转换为 pathlib.Path 对象列表
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

        # 鲁棒相机参数设置：百分位数、缩放余量、最大距离（若有）
        self.robust_percentile = robust_percentile
        self.zoom_margin = zoom_margin
        self.max_distance = max_distance

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
          dir_path: Path，目录路径

        返回：
          list[Path]：该目录下的所有 .npy 文件路径列表
        """
        if self.recursive:
            npy_files = list(dir_path.rglob("*.npy"))
        else:
            npy_files = list(dir_path.glob("*.npy"))
        logger.debug(f"在目录 {dir_path} 中找到 {len(npy_files)} 个 .npy 文件。")
        return npy_files

    def clean_filename(self, filename):
        """
        清理文件名，去除文件名中尾部形如 '_part_数字' 的部分。

        例如：
          'pc_001215_bin_1_r_3.38_t_1216.00_emb_801d8a58_part_0'
          清理后变为 'pc_001215_bin_1_r_3.38_t_1216.00_emb_801d8a58'

        参数：
          filename: str，原始文件名（不含扩展名）

        返回：
          str：清理后的文件名
        """
        cleaned = re.sub(r'_part_\d+$', '', filename)
        if cleaned != filename:
            logger.debug(f"清理文件名: '{filename}' => '{cleaned}'")
        return cleaned

    def load_point_cloud(self, npy_file):
        """
        加载 .npy 格式点云文件，返回 Open3D 点云对象。

        参数：
          npy_file: Path，.npy 文件路径

        返回：
          o3d.geometry.PointCloud：加载后的点云对象

        异常：
          FileNotFoundError：文件不存在；
          ValueError：点云数据格式不正确（必须至少包含3列XYZ）。
        """
        if not npy_file.exists():
            raise FileNotFoundError(f"文件 {npy_file} 不存在。")

        try:
            data = np.load(npy_file)
        except Exception as e:
            raise ValueError(f"无法加载文件 {npy_file}：{e}")

        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"点云数据 {npy_file} 至少需要3维坐标信息 (XYZ)。")

        # 构造 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # 前3列为XYZ坐标

        # 若 use_color 为 True 且数据列数>=6，则自动读取 RGB 信息（列3-5）
        if self.use_color and data.shape[1] >= 6:
            colors = data[:, 3:6].astype(np.float64)
            # 若颜色数值大于1，则认为范围为0-255，需归一化到[0,1]
            if colors.max() > 1.0:
                colors /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        logger.debug(f"加载点云文件 {npy_file} 成功，点数: {len(pcd.points)}。")
        return pcd

    def compute_camera_for_direction(self, pcd, direction):
        """
        根据给定点云 pcd 及期望的相机观察方向 direction，
        利用鲁棒统计方法计算点云核心区域，从而统一相机距离，
        保证在能拍摄到点云核心区域的前提下尽可能拉近镜头，
        且各个渲染视角的镜头位置保持一致。

        算法步骤：
          1. 从点云中提取所有点坐标，采用各坐标的中位数计算鲁棒中心（center）；
          2. 计算每个点到中心的欧氏距离，并取指定百分位数（默认95%）作为“鲁棒半径”；
          3. 根据视场角（fov），利用公式：
                 d = (鲁棒半径 / tan(fov/2)) * zoom_margin
             计算相机距离；若计算结果超过 max_distance（若设置），则采用最大距离；
          4. 对给定的观察方向归一化后，计算相机位置：
                 camera_location = center + direction_normalized * d；
          5. 相机目标点设为 center。

        参数：
          pcd: o3d.geometry.PointCloud，点云对象；
          direction: list[float]，期望的相机观察方向向量。

        返回：
          tuple: (camera_location (np.ndarray), camera_target (np.ndarray))
        """
        # 提取点云中所有点的坐标（shape=(N,3)）
        points = np.asarray(pcd.points)
        if points.size == 0:
            raise ValueError("点云中没有有效点。")
        # 计算鲁棒中心：采用各坐标的中位数，以减少离群点的干扰
        center = np.median(points, axis=0)

        # 计算每个点到中心的欧氏距离
        distances = np.linalg.norm(points - center, axis=1)
        # 采用指定百分位数（默认为95%）计算“鲁棒半径”，确保大部分点处于该半径内
        robust_radius = np.percentile(distances, self.robust_percentile)
        if robust_radius < 1e-9:
            robust_radius = 1e-3  # 避免过小

        # 根据视场角 fov 计算相机距离 d，使得点云核心区域刚好充满视野
        # 公式：d = (robust_radius / tan(fov/2)) * zoom_margin
        d = (robust_radius / np.tan(np.deg2rad(self.fov / 2.0))) * self.zoom_margin

        # 若设置了最大距离限制，则取较小值，避免镜头过远
        if self.max_distance is not None and d > self.max_distance:
            logger.debug(f"计算的相机距离 {d:.4f} 超过最大限制 {self.max_distance}，采用最大距离。")
            d = self.max_distance

        # 对输入方向向量归一化
        direction = np.array(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError(f"camera_direction={direction} 不能为零向量。")
        direction /= norm

        # 计算相机位置：从中心沿归一化方向移动距离 d
        camera_location = center + direction * d
        camera_target = center

        # 输出详细调试信息
        logger.debug(f"点云鲁棒中心: {center}")
        logger.debug(f"点云各点到中心距离（{self.robust_percentile}百分位）: {robust_radius}")
        logger.debug(f"计算得到的相机距离: {d:.4f}")
        logger.debug(f"归一化后的相机方向: {direction}")
        logger.debug(f"最终计算的相机位置: {camera_location}, 目标点: {camera_target}")

        return camera_location, camera_target

    def render_single(self, pcd, output_jpg, camera_location, camera_target):
        """
        使用 Open3D 的离屏渲染器将单个点云渲染为图像，并保存为 JPG 文件。

        参数：
          pcd: o3d.geometry.PointCloud，点云对象；
          output_jpg: Path，输出 JPG 文件路径；
          camera_location: np.ndarray，相机位置；
          camera_target: np.ndarray，相机目标点（通常为点云中心）。

        异常：
          若渲染过程中出现异常，将捕获错误并输出日志。
        """
        try:
            # 创建离屏渲染器，并设置图像尺寸
            renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
            scene = renderer.scene

            # 配置材质：采用无光照着色器，并设置点大小
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 2.0

            # 将点云添加至场景
            scene.add_geometry("pointcloud", pcd, material)
            scene.set_background(self.background_color)

            # 设置相机视角：根据视场角、目标点、相机位置及上向量配置
            renderer.setup_camera(self.fov, camera_target, camera_location, self.camera_up.tolist())

            # 执行离屏渲染，获取图像
            img_o3d = renderer.render_to_image()
            if img_o3d is None:
                raise RuntimeError("渲染图像失败，返回图像为空。")

            # 将 Open3D 图像转换为 NumPy 数组，再转换为 PIL 图像对象
            img_data = np.asarray(img_o3d)
            pil_img = Image.fromarray(img_data)

            # 确保输出目录存在，然后保存为 JPG（质量100）
            output_jpg.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(output_jpg, format="JPEG", quality=100)

            logger.info(f"成功渲染并保存图像至 {output_jpg}")

        except Exception as e:
            logger.error(f"渲染过程中出现错误: {e}")
            logger.debug(traceback.format_exc())
        finally:
            # 释放渲染器资源
            if 'renderer' in locals():
                del renderer

    def process_single_path(self, path, base_path):
        """
        处理单个输入路径：
          - 若 path 为单个 .npy 文件，则仅处理该文件；
          - 若 path 为文件夹，则检索其中的所有 .npy 文件；
          - 若路径无效，则跳过处理。

        参数：
          path: Path，待处理的文件或文件夹路径；
          base_path: Path，用于计算相对路径的基准目录。
        """
        if path.is_file():
            if path.suffix.lower() != ".npy":
                logger.warning(f"跳过无效文件: {path}（非 .npy 格式）")
                return
            npy_files = [path]
            # 若输入路径为文件，则基准路径为文件所在目录
            base_path = path.parent
        elif path.is_dir():
            # 若输入路径为目录，尝试查找是否存在 'data' 子目录
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
                # 加载点云数据
                pcd = self.load_point_cloud(npy_file)
            except Exception as e:
                logger.error(f"加载文件 {npy_file} 时发生错误：{e}")
                logger.debug(traceback.format_exc())
                continue

            # 计算相对路径，以保持输出目录结构与输入一致
            try:
                relative_dir = npy_file.parent.relative_to(base_path)
                logger.debug(f"相对目录为：{relative_dir}")
            except ValueError:
                relative_dir = Path('.')
                logger.debug("无法计算相对目录，使用根目录。")

            # 定义任务输出文件夹（保留输入文件夹名称及子目录结构）
            task_output_folder = self.output_folder / path.name / relative_dir
            task_output_folder.mkdir(parents=True, exist_ok=True)

            # 清理文件名，去除尾部 '_part_X'
            base_filename = self.clean_filename(npy_file.stem)

            # 针对每个指定的相机方向均进行一次渲染
            for idx, direction in enumerate(self.camera_directions):
                try:
                    # 计算相机位置和目标点（注意：此处使用统一的相机距离，保证所有视角一致）
                    cam_loc, cam_target = self.compute_camera_for_direction(pcd, direction)
                except Exception as e:
                    logger.error(f"计算相机方向 {direction} 时发生错误：{e}")
                    logger.debug(traceback.format_exc())
                    continue

                # 定义输出图片文件名，依次编号
                filename = f"{base_filename}_view{idx+1}.jpg"
                output_jpg = task_output_folder / filename

                # 执行渲染操作
                self.render_single(pcd, output_jpg, cam_loc, cam_target)

    def process_all(self):
        """
        遍历 input_paths 中的每个路径，依次调用 process_single_path 进行处理，
        最终完成所有点云文件的渲染。
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
    # 输入路径示例：每个路径既可为单个文件，也可为文件夹
    input_paths = [
        # "data/DensePointClouds/metaworld_soccer-v2",
        # "data/DensePointClouds/metaworld_drawer-open-v2",
        # "data/DensePointClouds/metaworld_door-open-v2",
        "data/DensePointClouds/metaworld_disassemble-v2",
        # "data/DensePointClouds/metaworld_handle-pull-side-v2",
        # "data/DensePointClouds/metaworld_peg-insert-side-v2"
    ]

    output_folder = "data/renderPointCloud"  # 输出图像保存根目录
    width = 400     # 渲染图像宽度
    height = 400    # 渲染图像高度
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
    fov = 60.0              # 视场角（度）
    background_color = [1.0, 1.0, 1.0, 1.0]  # 背景颜色 (RGBA)，取值范围 [0,1]
    use_color = True        # 是否使用点云颜色（RGB）
    recursive = True        # 是否递归搜索子目录
    robust_percentile = 95  # 用于计算鲁棒半径的百分位数
    zoom_margin = 1.5       # 镜头边缘余量因子，越小镜头越接近核心区域
    max_distance = 0.7     # 可选：若设定最大相机距离，可防止因离群点导致镜头过远

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
        recursive=recursive,
        robust_percentile=robust_percentile,
        zoom_margin=zoom_margin,
        max_distance=max_distance
    )

    # 开始处理所有输入路径
    renderer.process_all()