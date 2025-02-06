import os
import glob
import numpy as np
import open3d as o3d
from PIL import Image

class PointCloudRenderer:
    """
    PointCloudRenderer 用于加载 .npy 点云文件并对其进行渲染，最终输出
    为 JPEG 格式的图像。它能够灵活设置相机、背景及是否使用颜色等渲染配置。
    同时支持处理多个 input_path，并能递归检索子文件夹中的 .npy 文件。
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
        # 将 input_paths 统一转换为列表
        if isinstance(input_paths, str):
            self.input_paths = [input_paths]
        elif isinstance(input_paths, list):
            self.input_paths = input_paths
        else:
            raise ValueError("input_paths 需为字符串或字符串列表。")

        self.output_folder = output_folder
        self.width = width
        self.height = height
        self.camera_up = camera_up
        self.camera_directions = camera_directions
        self.fov = fov
        self.background_color = background_color
        self.use_color = use_color
        self.recursive = recursive

        # 渲染前准备：若输出文件夹不存在，先创建
        os.makedirs(self.output_folder, exist_ok=True)

    def compute_camera_for_direction(self, pcd, direction):
        """
        给定点云对象 pcd 和指定方向向量 direction，计算摄像机位置与目标点。
        """
        # 通过点云的有向包围盒来获取中心、尺度信息
        obb = pcd.get_oriented_bounding_box()
        center = obb.center
        extent = obb.extent
        max_extent = max(extent)

        # 基于视角和最大包围盒尺寸，计算合适的拍摄距离
        distance = (max_extent / 2) / np.tan(np.deg2rad(self.fov / 2)) * 1.5

        # 规范化方向向量
        direction = np.array(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("camera_direction 不能为零向量。")
        direction /= norm

        # 将摄像机位置固定在 center + direction * distance 处
        camera_location = center + direction * distance
        return camera_location, center

    def load_point_cloud(self, npy_file):
        """
        加载 .npy 格式点云文件并返回 Open3D 点云对象。
        """
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"文件 {npy_file} 不存在。")

        data = np.load(npy_file)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"点云数据 {npy_file} 至少需要包含三维坐标信息 (XYZ)。")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])

        if self.use_color and data.shape[1] >= 6:
            colors = data[:, 3:6].astype(np.float64)
            if colors.max() > 1.0:
                colors /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def render_single(self, pcd, output_jpg, camera_location, camera_target):
        """
        渲染单个点云对象并保存为 JPEG 图像。
        """
        try:
            # 创建离屏渲染器
            renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
            scene = renderer.scene

            # 配置点云的材质
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 2.0

            # 将点云添加到场景
            scene.add_geometry("pointcloud", pcd, material)

            # 设置背景色
            scene.set_background(self.background_color)

            # 设置摄像机参数
            renderer.setup_camera(self.fov, camera_target, camera_location, self.camera_up)

            # 渲染场景
            img = renderer.render_to_image()
            img_data = np.asarray(img)
            pil_img = Image.fromarray(img_data)

            os.makedirs(os.path.dirname(output_jpg), exist_ok=True)
            pil_img.save(output_jpg, format="JPEG", quality=100)

            # 清理几何体，释放渲染器资源
            scene.clear_geometry()
        except Exception as e:
            print(f"渲染过程中出现错误: {e}")
        finally:
            # 删除渲染器对象以释放资源
            if 'renderer' in locals():
                del renderer

        print(f"成功渲染并保存图像至 {output_jpg}")

    def get_npy_files_in_dir(self, dir_path):
        """
        返回指定目录下的所有 .npy 文件列表，支持递归搜索。
        """
        if self.recursive:
            return [os.path.join(dp, f) 
                    for dp, dn, filenames in os.walk(dir_path) 
                    for f in filenames if f.endswith(".npy")]
        else:
            return glob.glob(os.path.join(dir_path, "*.npy"))

    def process_single_path(self, path):
        """
        处理单一路径，可为文件夹或单个 .npy 文件。
        """
        if os.path.isfile(path):
            if not path.endswith(".npy"):
                print(f"跳过无效文件: {path} (非 .npy 格式)")
                return
            task_name = os.path.basename(os.path.dirname(path))
            file_list = [path]
        elif os.path.isdir(path):
            task_name = os.path.basename(os.path.normpath(path))
            file_list = self.get_npy_files_in_dir(path)
            if not file_list:
                print(f"文件夹 {path} 中未找到任何 .npy 文件。")
                return
        else:
            print(f"输入路径 {path} 无效，跳过。")
            return

        task_output_folder = os.path.join(self.output_folder, task_name)
        os.makedirs(task_output_folder, exist_ok=True)

        # 处理相机方向列表
        if self.camera_directions and isinstance(self.camera_directions, list) and len(self.camera_directions) > 0:
            directions = self.camera_directions
        else:
            print("警告: 未指定相机方向，使用默认 [0, 0, 1]")
            directions = [[0, 0, 1]]

        for npy_file in file_list:
            base_filename = os.path.splitext(os.path.basename(npy_file))[0]

            try:
                pcd = self.load_point_cloud(npy_file)
            except Exception as e:
                print(f"加载文件 {npy_file} 时发生错误：{e}")
                continue

            for idx, direction in enumerate(directions):
                try:
                    cam_loc, cam_target = self.compute_camera_for_direction(pcd, direction)
                except Exception as e:
                    print(f"计算方向 {direction} 时发生错误：{e}")
                    continue

                filename = f"{base_filename}_view{idx+1}.jpg"
                output_jpg = os.path.join(task_output_folder, filename)

                try:
                    self.render_single(pcd, output_jpg, cam_loc, cam_target)
                except Exception as e:
                    print(f"渲染文件 {npy_file} 在方向 {direction} 时发生错误：{e}")

    def process_all(self):
        """
        对所有输入路径执行渲染流程。
        """
        for path in self.input_paths:
            self.process_single_path(path)

if __name__ == "__main__":
    # ============== 配置参数 ==============
    # 多个输入路径示例: 每个路径都可能是单个文件或一个文件夹
    input_paths = [
        # "tests",
        # "data/DensePointClouds/metaworld_soccer-v2",
        # "data/DensePointClouds/metaworld_drawer-open-v2",
        # "data/DensePointClouds/metaworld_door-open-v2",
        # "data/DensePointClouds/metaworld_disassemble-v2",
        # "data/DensePointClouds/metaworld_handle-pull-side-v2",
        "data/DensePointClouds/metaworld_peg-insert-side-v2"
    ]

    output_folder = "data/renderPointCloud"
    # output_folder = "tests"
    width = 400
    height = 400
    camera_up = [0, 1, 0]
    camera_directions = [
    # metaworld_soccer-v2, metaworld_drawer-open-v2, metaworld_door-open-v2, metaworld_disassemble-v2
        # [1, -1, 1],
        # [-1, -1, 1],
    # metaworld_handle-pull-side-v2
        # [0, 0, 1],
        # [1, 0, 1],
    # metaworld_peg-insert-side-v2
        [-1, 0, 0],
        [1, 0, 0],
    ]
    fov = 60.0
    background_color = [1.0, 1.0, 1.0, 1.0]
    use_color = True
    recursive = True
    # ======================================

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

    renderer.process_all()