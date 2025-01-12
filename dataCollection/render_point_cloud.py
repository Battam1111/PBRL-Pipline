import os
import numpy as np
import open3d as o3d
from PIL import Image

class PointCloudRenderer:
    def __init__(self, 
                 input_path, 
                 output_folder, 
                 width=800, 
                 height=600,
                 camera_location=None, 
                 camera_target=None, 
                 camera_up=[0, 1, 0],
                 camera_direction=None,
                 camera_directions=None,
                 fov=60.0, 
                 background_color=[1.0, 1.0, 1.0, 1.0],
                 use_color=True):
        """
        初始化渲染器配置参数。
        """
        self.input_path = input_path
        self.output_folder = output_folder
        self.width = width
        self.height = height
        self.camera_location = camera_location
        self.camera_target = camera_target
        self.camera_up = camera_up
        self.camera_direction = camera_direction
        self.camera_directions = camera_directions
        self.fov = fov
        self.background_color = background_color
        self.use_color = use_color

    def compute_camera_for_direction(self, pcd, direction):
        """
        给定点云和指定方向，计算固定的摄像机位置和目标点，使得该相机位置在该方向上观看点云并保持固定。
        """
        obb = pcd.get_oriented_bounding_box()
        center = obb.center
        extent = obb.extent
        max_extent = max(extent)
        # 计算合适的距离，以确保点云在视野内
        distance = (max_extent / 2) / np.tan(np.deg2rad(self.fov / 2)) * 1.5

        # 使用提供的固定方向
        direction = np.array(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("camera_direction 不能为零向量。")
        direction /= norm

        camera_location = center + direction * distance
        return camera_location, center

    def load_point_cloud(self, npy_file):
        """
        加载 .npy 点云文件并返回 Open3D 点云对象。
        """
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"文件 {npy_file} 不存在。")

        data = np.load(npy_file)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"点云数据 {npy_file} 至少需要包含三维坐标信息。")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])

        if self.use_color and data.shape[1] >= 6:
            colors = data[:, 3:6].astype(np.float64)
            if colors.max() > 1.0:
                colors /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def render_single(self, pcd, output_jpg):
        """
        渲染单个点云对象并保存为 JPEG 图像以提高压缩效率。
        """
        renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        scene = renderer.scene

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 2.0
        scene.add_geometry("pointcloud", pcd, material)

        scene.set_background(self.background_color)
        renderer.setup_camera(self.fov, self.camera_target, self.camera_location, self.camera_up)

        # 获取渲染的图像
        img = renderer.render_to_image()
        # 将 Open3D 图像转换为 numpy 数组
        img_data = np.asarray(img)
        # 转换为 PIL 图像对象
        pil_img = Image.fromarray(img_data)
        # 保存为 JPEG 格式
        os.makedirs(os.path.dirname(output_jpg), exist_ok=True)
        pil_img.save(output_jpg, format="JPEG", quality=90)

        scene.clear_geometry()
        print(f"成功渲染并保存图像至 {output_jpg}")

    def process(self):
        """
        根据输入路径处理点云渲染。对于每个指定的相机方向，固定计算摄像机位置和目标，然后针对每个文件渲染图像。
        """
        # 根据输入路径确定任务名
        if os.path.isfile(self.input_path):
            task_name = os.path.basename(os.path.dirname(self.input_path))
        elif os.path.isdir(self.input_path):
            task_name = os.path.basename(os.path.normpath(self.input_path))
        else:
            print(f"输入路径 {self.input_path} 无效。")
            return

        # 在输出文件夹下创建任务名子文件夹
        task_output_folder = os.path.join(self.output_folder, task_name)
        os.makedirs(task_output_folder, exist_ok=True)

        # 获取待处理的文件列表
        file_list = []
        if os.path.isfile(self.input_path) and self.input_path.endswith(".npy"):
            file_list.append(self.input_path)
        elif os.path.isdir(self.input_path):
            file_list = glob.glob(os.path.join(self.input_path, "*.npy"))
            if not file_list:
                print(f"文件夹 {self.input_path} 中未找到 .npy 文件。")
                return

        # 确定相机方向列表
        if self.camera_directions is not None and isinstance(self.camera_directions, list):
            directions = self.camera_directions
        else:
            directions = [self.camera_direction]  # 单一方向或 None

        # 对于每个相机方向，先固定计算摄像机位置和目标
        direction_to_camera = {}
        sample_pcd = self.load_point_cloud(file_list[0])

        for idx, direction in enumerate(directions):
            try:
                cam_loc, cam_target = self.compute_camera_for_direction(sample_pcd, direction)
                direction_to_camera[tuple(direction)] = (cam_loc, cam_target)
                print(f"为方向 {direction} 计算得到固定的相机位置: {cam_loc}, 目标: {cam_target}")
            except Exception as e:
                print(f"计算方向 {direction} 时发生错误：{e}")

        # 针对每个文件和每个方向进行渲染
        for npy_file in file_list:
            base_filename = os.path.splitext(os.path.basename(npy_file))[0]
            try:
                pcd = self.load_point_cloud(npy_file)
            except Exception as e:
                print(f"加载文件 {npy_file} 时发生错误：{e}")
                continue

            for idx, direction in enumerate(directions):
                key = tuple(direction)
                if key not in direction_to_camera:
                    print(f"未找到方向 {direction} 对应的摄像机设置，跳过。")
                    continue

                self.camera_direction = direction
                self.camera_location, self.camera_target = direction_to_camera[key]

                filename = f"{base_filename}_view{idx+1}.jpg"
                output_jpg = os.path.join(task_output_folder, filename)
                try:
                    self.render_single(pcd, output_jpg)
                except Exception as e:
                    print(f"渲染文件 {npy_file} 在方向 {direction} 时发生错误：{e}")

if __name__ == "__main__":
    import glob  # 确保glob被导入
    # ============== 配置参数 ==============
    input_path = "test/pointclouds/metaworld_soccer-v2"  # 或者指向文件夹或单个 .npy 文件
    output_folder = "test/renderPointCloud"
    width = 400
    height = 400
    camera_location = None    
    camera_target = None      
    camera_up = [0, 1, 0]
    
    camera_directions = [
        [1, -1, 1],
        [-1, -1, 1],
    ]
    
    fov = 60.0
    background_color = [1.0, 1.0, 1.0, 1.0]
    use_color = False
    # ======================================

    renderer = PointCloudRenderer(
        input_path=input_path,
        output_folder=output_folder,
        width=width,
        height=height,
        camera_location=camera_location,
        camera_target=camera_target,
        camera_up=camera_up,
        camera_directions=camera_directions,
        fov=fov,
        background_color=background_color,
        use_color=use_color
    )

    renderer.process()
