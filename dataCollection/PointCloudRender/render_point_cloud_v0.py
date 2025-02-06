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
        camera_location=None,
        camera_target=None,
        camera_up=[0, 1, 0],
        camera_direction=None,
        camera_directions=None,
        fov=60.0,
        background_color=[1.0, 1.0, 1.0, 1.0],
        use_color=True,
        recursive=True
    ):
        """
        初始化渲染器配置参数。

        参数:
        ----
        input_paths: str 或 list
            输入路径(字符串)或路径列表(可处理多个输入)。每个输入可以是指向包含
            .npy 文件的文件夹，也可以是单独的 .npy 文件。
        output_folder: str
            渲染输出文件夹路径。
        width: int
            渲染图像的宽度。
        height: int
            渲染图像的高度。
        camera_location: list or tuple
            直接指定摄像机的位置 (x, y, z)；仅当不使用 camera_directions
            来自动计算位置时有效。
        camera_target: list or tuple
            摄像机所对准的目标点 (x, y, z)。
        camera_up: list
            摄像机在世界坐标系中的“上”方向向量。
        camera_direction: list or tuple
            单一相机拍摄朝向向量 (x, y, z)。若 camera_directions 未指定，则
            使用此参数进行渲染。
        camera_directions: list of list
            多个相机拍摄朝向向量的集合。每个元素都是长度为 3 的列表或元组。
        fov: float
            摄像机视场角(角度制)。
        background_color: list
            背景颜色(R, G, B, A)，数值范围在 [0, 1] 之间。
        use_color: bool
            是否从点云中读取 RGB 颜色并进行渲染。
        recursive: bool
            若为 True，则对输入路径的子文件夹进行递归检索；否则只检索当前文件夹。
        """
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
        self.camera_location = camera_location
        self.camera_target = camera_target
        self.camera_up = camera_up
        self.camera_direction = camera_direction
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

        其核心思路是:
        1. 基于点云的有向包围盒(Oriented Bounding Box)，获取点云的中心和最大
           尺寸信息；
        2. 按照设定的视场角 fov 计算合适的拍摄距离 distance，使得点云在视野内；
        3. 将摄像机放置在 `center + direction * distance` 处，并使目标点对准点云中心。

        参数:
        ----
        pcd: open3d.geometry.PointCloud
            Open3D 点云对象。
        direction: list or np.array
            摄像机拍摄方向向量。

        返回:
        ----
        (camera_location, center): tuple[np.array, np.array]
            - camera_location: 摄像机位置（世界坐标系）
            - center: 摄像机对准的目标点（世界坐标系）
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

        如果使用颜色且数据满足 (N, 6) 或更多，
        则将第四、五、六列视为 RGB 三通道信息。

        参数:
        ----
        npy_file: str
            指定的 .npy 文件路径。

        返回:
        ----
        pcd: open3d.geometry.PointCloud
            加载后的点云对象。
        """
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"文件 {npy_file} 不存在。")

        data = np.load(npy_file)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"点云数据 {npy_file} 至少需要包含三维坐标信息 (XYZ)。")

        # 建立点云对象，并设定其坐标
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])

        # 如果使用颜色，且有至少 6 列，则将第4-6列作为RGB
        if self.use_color and data.shape[1] >= 6:
            colors = data[:, 3:6].astype(np.float64)
            if colors.max() > 1.0:  # 若颜色范围是 0~255，则归一化到 0~1
                colors /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd


    def render_single(self, pcd, output_jpg):
        """
        渲染单个点云对象并保存为 JPEG 图像，以提高压缩和存储效率。

        参数:
        ----
        pcd: open3d.geometry.PointCloud
            要渲染的点云对象。
        output_jpg: str
            输出图像文件名，包括后缀 .jpg。
        """
        # 创建离屏渲染器
        renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        scene = renderer.scene

        # 配置点云的材质
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"  # 更简单的着色器
        material.point_size = 2.0

        # 将点云添加到场景中
        scene.add_geometry("pointcloud", pcd, material)

        # 设置渲染背景
        scene.set_background(self.background_color)

        # 设置摄像机参数
        renderer.setup_camera(self.fov, self.camera_target, self.camera_location, self.camera_up)

        # 渲染场景到图像
        img = renderer.render_to_image()

        # 转为 numpy 数组
        img_data = np.asarray(img)

        # 转为 PIL 图像对象
        pil_img = Image.fromarray(img_data)

        # 保存为 JPEG 格式
        os.makedirs(os.path.dirname(output_jpg), exist_ok=True)
        pil_img.save(output_jpg, format="JPEG", quality=90)

        # 清理几何体（可选步骤，防止内存过多占用）
        scene.clear_geometry()

        print(f"成功渲染并保存图像至 {output_jpg}")


    def get_npy_files_in_dir(self, dir_path):
        """
        根据是否递归检索子文件夹，返回该文件夹内所有 .npy 文件列表。

        参数:
        ----
        dir_path: str
            目标文件夹的路径。

        返回:
        ----
        file_list: list
            该文件夹及子文件夹(如指定)下的 .npy 文件列表(绝对路径)。
        """
        if self.recursive:
            # 递归检索所有子文件夹
            return [os.path.join(dp, f) 
                    for dp, dn, filenames in os.walk(dir_path) 
                    for f in filenames if f.endswith(".npy")]
        else:
            # 仅检索当前文件夹
            return glob.glob(os.path.join(dir_path, "*.npy"))


    def process_single_path(self, path):
        """
        处理单个 input_path：可以是文件夹，也可以是单个 .npy 文件。

        包括以下步骤:
        1. 若为文件夹，则获取其所有 .npy 文件列表；
        2. 若为文件，则直接渲染该文件；
        3. 对最终获取到的所有 .npy 文件进行渲染处理；
        4. 输出结果保存到 self.output_folder 下，以任务名区分。
        """
        # 若输入是单个 .npy 文件，则其“任务名”取父文件夹名
        if os.path.isfile(path):
            if not path.endswith(".npy"):
                print(f"跳过无效文件: {path} (非 .npy 格式)")
                return
            task_name = os.path.basename(os.path.dirname(path))
            file_list = [path]
        elif os.path.isdir(path):
            # 若是文件夹，则其“任务名”即该文件夹名称
            task_name = os.path.basename(os.path.normpath(path))
            file_list = self.get_npy_files_in_dir(path)
            if not file_list:
                print(f"文件夹 {path} 中未找到任何 .npy 文件。")
                return
        else:
            print(f"输入路径 {path} 无效，跳过。")
            return

        # 为当前任务创建输出文件夹
        task_output_folder = os.path.join(self.output_folder, task_name)
        os.makedirs(task_output_folder, exist_ok=True)

        # 如果用户指定了多个相机方向，则需要遍历每个方向
        if self.camera_directions is not None and isinstance(self.camera_directions, list):
            directions = self.camera_directions
        else:
            # 若没有指定列表，就只用单一方向
            if self.camera_direction is None:
                # 如果依旧没有指定单一方向，提示一下
                print("警告: 未指定相机方向，使用默认 [0, 0, 1]")
                directions = [[0, 0, 1]]
            else:
                directions = [self.camera_direction]

        # 如果代码中要依赖sample_pcd进行摄像机位置计算，则取第一个文件作为“基准”
        try:
            sample_pcd = self.load_point_cloud(file_list[0])
        except Exception as e:
            print(f"无法加载文件 {file_list[0]} 进行相机计算，跳过此路径: {e}")
            return

        # 给每个方向计算固定摄像机位置与目标点
        direction_to_camera = {}
        for idx, direction in enumerate(directions):
            try:
                cam_loc, cam_target = self.compute_camera_for_direction(sample_pcd, direction)
                direction_to_camera[tuple(direction)] = (cam_loc, cam_target)
                print(f"为方向 {direction} 计算得到固定的相机位置: {cam_loc}, 目标: {cam_target}")
            except Exception as e:
                print(f"计算方向 {direction} 时发生错误：{e}")
                continue

        # 遍历待渲染的点云文件
        for npy_file in file_list:
            base_filename = os.path.splitext(os.path.basename(npy_file))[0]

            # 加载点云
            try:
                pcd = self.load_point_cloud(npy_file)
            except Exception as e:
                print(f"加载文件 {npy_file} 时发生错误：{e}")
                continue

            # 针对每个方向进行渲染
            for idx, direction in enumerate(directions):
                key = tuple(direction)
                if key not in direction_to_camera:
                    print(f"未找到方向 {direction} 对应的摄像机设置，跳过。")
                    continue

                # 设置摄像机参数
                self.camera_direction = direction
                self.camera_location, self.camera_target = direction_to_camera[key]

                # 生成输出图像路径
                filename = f"{base_filename}_view{idx+1}.jpg"
                output_jpg = os.path.join(task_output_folder, filename)

                # 开始渲染
                try:
                    self.render_single(pcd, output_jpg)
                except Exception as e:
                    print(f"渲染文件 {npy_file} 在方向 {direction} 时发生错误：{e}")


    def process_all(self):
        """
        对 self.input_paths 中的每个路径执行渲染流程。支持同时处理多个路径。
        注意，每个路径都会各自创建对应的任务子文件夹。
        """
        for path in self.input_paths:
            self.process_single_path(path)


if __name__ == "__main__":
    """
    示例用法:
    假设我们要同时处理多个文件夹或 .npy 文件，则可以把它们整合到一个列表里。
    你可以根据需要只放单个字符串或列表。
    """

    # ============== 配置参数 ==============
    # 多个输入路径示例: 每个路径都可能是单个文件或一个文件夹
    input_paths = [
        "tests",
        # "data/pointclouds/metaworld_soccer-v2",
        # "data/pointclouds/metaworld_drawer-open-v2",
        # "data/pointclouds/metaworld_door-open-v2",
        # "data/pointclouds/metaworld_disassemble-v2",
        # "data/pointclouds/metaworld_handle-pull-side-v2",
        # "data/pointclouds/metaworld_peg-insert-side-v2"
    ]

    # output_folder = "data/renderPointCloud"
    output_folder = "tests"
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
    use_color = True
    recursive = True  # 若想只处理当前文件夹，不遍历子文件夹，可设为 False
    # ======================================

    # 创建渲染器对象
    renderer = PointCloudRenderer(
        input_paths=input_paths,
        output_folder=output_folder,
        width=width,
        height=height,
        camera_location=camera_location,
        camera_target=camera_target,
        camera_up=camera_up,
        camera_directions=camera_directions,
        fov=fov,
        background_color=background_color,
        use_color=use_color,
        recursive=recursive
    )

    # 执行渲染流程
    renderer.process_all()