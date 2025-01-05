import threading
import open3d as o3d
import mujoco
import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)

class ScenePointCloudExtractor:
    """
    从 MuJoCo 模型和数据中提取场景的完整点云，包括 xyz 坐标和 rgb 颜色信息。
    """

    def __init__(self, model, data, task_related_body_names=None, num_points=8192, random_seed=None):
        """
        初始化提取器。

        参数：
            model: MuJoCo 模型对象。
            data: MuJoCo 数据对象。
            task_related_body_names (list): 任务相关的身体名称列表，默认为空列表。
            num_points (int): 每个几何体表面采样的点数。
            random_seed (int): 随机数种子，用于结果复现。
        """
        self.model = model
        self.data = data
        self.task_related_body_names = task_related_body_names if task_related_body_names else []
        self.num_points = num_points
        if random_seed is not None:
            np.random.seed(random_seed)

        # 构建身体的父子关系映射
        self._build_body_tree()

    def _build_body_tree(self):
        """
        构建身体的父子关系映射，用于遍历身体树。
        """
        self.body_parentid = self.model.body_parentid
        self.body_children = {i: [] for i in range(self.model.nbody)}
        for i in range(1, self.model.nbody):  # 从 1 开始，跳过世界身体
            parent_id = self.body_parentid[i]
            self.body_children[parent_id].append(i)

    def extract_point_cloud(self):
        """
        提取场景的完整点云数据。

        返回：
            point_cloud (np.ndarray): 点云数据，形状为 (N, 6)，包含 xyz 和 rgb 信息。
        """
        all_points = []
        all_colors = []

        # 获取任务相关的几何体索引
        task_related_geom_ids = self._get_task_related_geom_ids()

        if not task_related_geom_ids:
            print("No task-related geoms found.")
            return np.empty((0, 6))

        for geom_id in task_related_geom_ids:
            # 提取几何体信息
            geom_type = self.model.geom_type[geom_id]
            geom_pos = self.data.geom_xpos[geom_id]
            geom_xmat = self.data.geom_xmat[geom_id].reshape(3, 3)
            geom_size = self.model.geom_size[geom_id]

            # 获取颜色信息
            geom_rgb = self._get_geom_rgb(geom_id)

            # 获取旋转矩阵（MuJoCo 的 geom_xmat 是列主序，需要转置）
            rotation_matrix = geom_xmat

            try:
                if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    vertices = self._sample_points_on_box(geom_size)
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    vertices = self._sample_points_on_sphere(geom_size[0])
                elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    vertices = self._sample_points_on_cylinder(geom_size[0], geom_size[1])
                elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    vertices = self._sample_points_on_capsule(geom_size[0], geom_size[1])
                elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                    vertices = self._sample_points_on_ellipsoid(geom_size)
                elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    vertices = self._get_mesh_vertices(geom_id)
                else:
                    print(f"Unsupported geom type: {geom_type}")
                    continue  # 跳过不支持的几何体类型
            except ValueError as e:
                print(f"Error processing geom_id {geom_id}: {e}")
                continue

            # 转换到全局坐标系
            vertices = vertices @ rotation_matrix.T + geom_pos
            colors = np.tile(geom_rgb, (vertices.shape[0], 1))

            all_points.append(vertices)
            all_colors.append(colors)

        if all_points:
            all_points = np.vstack(all_points)
            all_colors = np.vstack(all_colors)
            point_cloud = np.hstack((all_points, all_colors))
        else:
            point_cloud = np.empty((0, 6))

        return point_cloud

    def _get_task_related_geom_ids(self):
        """
        获取任务相关的几何体索引。

        返回：
            task_related_geom_ids (list): 任务相关的几何体 ID 列表。
        """
        task_related_geom_ids = []

        for body_name in self.task_related_body_names:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            except ValueError:
                print(f"Body name '{body_name}' not found in model.")
                continue

            # 递归地收集该身体及其子孙身体的几何体 ID
            geom_ids = self._collect_geom_ids(body_id)
            task_related_geom_ids.extend(geom_ids)

        return task_related_geom_ids

    def _collect_geom_ids(self, body_id):
        """
        递归地收集指定身体及其子孙身体的几何体 ID。

        参数：
            body_id (int): 身体 ID。

        返回：
            geom_ids (list): 收集到的几何体 ID 列表。
        """
        geom_ids = []

        # 获取该身体的几何体
        geom_start = self.model.body_geomadr[body_id]
        geom_num = self.model.body_geomnum[body_id]

        # 获取身体名称
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        # print(f"Processing Body ID: {body_id}, Name: '{body_name}', geom_start: {geom_start}, geom_num: {geom_num}")

        if geom_num > 0:
            geom_ids.extend(range(geom_start, geom_start + geom_num))
            # print(f"Body '{body_name}' has geoms: {geom_ids[-geom_num:]}")
        else:
            pass
            # print(f"Body '{body_name}' has no direct geoms.")

        # 递归地处理子身体
        child_body_ids = self.body_children.get(body_id, [])
        # print(f"Body '{body_name}' has children: {child_body_ids}")
        for child_body_id in child_body_ids:
            geom_ids.extend(self._collect_geom_ids(child_body_id))

        return geom_ids


    def _get_geom_rgb(self, geom_id):
        """
        获取几何体的 RGB 颜色信息，考虑材质和透明度。

        参数：
            geom_id (int): 几何体 ID。

        返回：
            geom_rgb (np.ndarray): 形状为 (3,) 的 RGB 数组。
        """
        geom_rgba = self.model.geom_rgba[geom_id]
        geom_rgba = np.clip(geom_rgba, 0, 1)

        material_id = self.model.geom_matid[geom_id]
        if material_id != -1:
            material_rgba = self.model.mat_rgba[material_id]
            material_rgba = np.clip(material_rgba, 0, 1)
            # 结合材质和几何体的颜色，考虑透明度
            alpha = geom_rgba[3] * material_rgba[3]
            if alpha > 0:
                geom_rgb = (geom_rgba[:3] * geom_rgba[3] + material_rgba[:3] * material_rgba[3] * (1 - geom_rgba[3])) / alpha
            else:
                geom_rgb = np.zeros(3)
        else:
            geom_rgb = geom_rgba[:3]
        return geom_rgb

    def _sample_points_on_box(self, size):
        """
        在盒子表面均匀采样点。

        参数：
            size (np.ndarray): 盒子的尺寸，形状为 (3,)。

        返回：
            points (np.ndarray): 采样点，形状为 (N, 3)。
        """
        # 计算每个面的面积
        areas = np.array([
            size[1] * size[2],  # x 面
            size[0] * size[2],  # y 面
            size[0] * size[1],  # z 面
        ])
        areas = np.hstack([areas, areas])  # 对于正反两面

        # 根据面积分配采样点数量
        total_area = np.sum(areas)
        face_points = np.ceil(self.num_points * areas / total_area).astype(int)

        points = []

        # 六个面分别采样
        # x 方向的面
        for sign in [-1, 1]:
            n = face_points[0]
            ys = np.random.uniform(-size[1], size[1], n)
            zs = np.random.uniform(-size[2], size[2], n)
            xs = np.full(n, sign * size[0])
            points.append(np.vstack((xs, ys, zs)).T)

        # y 方向的面
        for sign in [-1, 1]:
            n = face_points[1]
            xs = np.random.uniform(-size[0], size[0], n)
            zs = np.random.uniform(-size[2], size[2], n)
            ys = np.full(n, sign * size[1])
            points.append(np.vstack((xs, ys, zs)).T)

        # z 方向的面
        for sign in [-1, 1]:
            n = face_points[2]
            xs = np.random.uniform(-size[0], size[0], n)
            ys = np.random.uniform(-size[1], size[1], n)
            zs = np.full(n, sign * size[2])
            points.append(np.vstack((xs, ys, zs)).T)

        return np.vstack(points)

    def _sample_points_on_sphere(self, radius):
        """
        在球体表面均匀采样点。

        参数：
            radius (float): 球体半径。

        返回：
            points (np.ndarray): 采样点，形状为 (N, 3)。
        """
        phi = np.arccos(1 - 2 * np.random.uniform(0, 1, self.num_points))
        theta = np.random.uniform(0, 2 * np.pi, self.num_points)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return np.vstack((x, y, z)).T

    def _sample_points_on_cylinder(self, radius, height):
        """
        在圆柱体表面均匀采样点。

        参数：
            radius (float): 圆柱体半径。
            height (float): 圆柱体高度。

        返回：
            points (np.ndarray): 采样点，形状为 (N, 3)。
        """
        # 计算侧面和顶底面积
        side_area = 2 * np.pi * radius * height
        cap_area = np.pi * radius ** 2
        areas = np.array([side_area, cap_area * 2])

        # 根据面积分配采样点数量
        total_area = np.sum(areas)
        nums = np.ceil(self.num_points * areas / total_area).astype(int)
        side_points_num = nums[0]
        cap_points_num = nums[1] // 2

        # 侧面采样
        theta = np.random.uniform(0, 2 * np.pi, side_points_num)
        z = np.random.uniform(-height / 2, height / 2, side_points_num)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        side_points = np.vstack((x, y, z)).T

        # 顶底面采样
        r = np.sqrt(np.random.uniform(0, radius ** 2, cap_points_num))
        theta = np.random.uniform(0, 2 * np.pi, cap_points_num)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        top_z = np.full(cap_points_num, height / 2)
        bottom_z = np.full(cap_points_num, -height / 2)
        top_points = np.vstack((x, y, top_z)).T
        bottom_points = np.vstack((x, y, bottom_z)).T

        return np.vstack((side_points, top_points, bottom_points))

    def _sample_points_on_capsule(self, radius, height):
        """
        在胶囊体表面均匀采样点。

        参数：
            radius (float): 胶囊体半径。
            height (float): 圆柱部分的高度。

        返回：
            points (np.ndarray): 采样点，形状为 (N, 3)。
        """
        # 分配采样点数量
        cylinder_area = 2 * np.pi * radius * height
        sphere_area = 4 * np.pi * radius ** 2
        areas = np.array([cylinder_area, sphere_area])
        total_area = np.sum(areas)
        nums = np.ceil(self.num_points * areas / total_area).astype(int)
        cylinder_points_num = nums[0]
        sphere_points_num = nums[1]

        # 圆柱部分采样
        cylinder_points = self._sample_points_on_cylinder(radius, height)
        if len(cylinder_points) > cylinder_points_num:
            cylinder_points = cylinder_points[:cylinder_points_num]

        # 球面部分采样
        sphere_points = self._sample_points_on_sphere(radius)
        sphere_points = sphere_points[:sphere_points_num]

        # 将球面点分为上半球和下半球
        half = sphere_points_num // 2
        top_sphere_points = sphere_points[:half] + np.array([0, 0, height / 2])
        bottom_sphere_points = sphere_points[half:] + np.array([0, 0, -height / 2])

        return np.vstack((cylinder_points, top_sphere_points, bottom_sphere_points))

    def _sample_points_on_ellipsoid(self, size):
        """
        在椭球体表面均匀采样点。

        参数：
            size (np.ndarray): 椭球体的半轴长度，形状为 (3,)。

        返回：
            points (np.ndarray): 采样点，形状为 (N, 3)。
        """
        phi = np.arccos(1 - 2 * np.random.uniform(0, 1, self.num_points))
        theta = np.random.uniform(0, 2 * np.pi, self.num_points)
        x = size[0] * np.sin(phi) * np.cos(theta)
        y = size[1] * np.sin(phi) * np.sin(theta)
        z = size[2] * np.cos(phi)
        return np.vstack((x, y, z)).T

    def _get_mesh_vertices(self, geom_id):
        """
        获取网格几何体的顶点，并转换到几何体的局部坐标系。

        参数：
            geom_id (int): 几何体 ID。

        返回：
            vertices (np.ndarray): 顶点坐标，形状为 (N, 3)。
        """
        mesh_id = self.model.geom_dataid[geom_id]
        vert_adr = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        vertices = self.model.mesh_vert[vert_adr:vert_adr + vert_num]
        vertices = vertices.reshape(-1, 3)
        return vertices




# 吸取教训，别自个整蛊
class PointCloudSaver:
    """
    用于保存点云的工具类。
    """

    def __init__(self, filename='/home/star/Yanjun/RL-VLM-F/html/point_cloud-ori.ply'):
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


class SawyerSoccerEnvV2(SawyerXYZEnv):
    OBJ_RADIUS = 0.013
    TARGET_RADIUS = 0.07

    def __init__(self, tasks=None, render_mode=None, width=500, height=500):
        # goal_low = (-0.1, 0.8, 0.0)
        # goal_high = (0.1, 0.9, 0.0)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        # obj_low = (-0.1, 0.6, 0.03)
        # obj_high = (0.1, 0.7, 0.03)

        # 设置渲染宽度和高度
        self.width = width
        self.height = height

        self.N = 8192  # 点云采样点数

        # 设置渲染模式和摄像机参数
        self.render_mode = render_mode
        self.camera_name = None
        self.camera_id = None

        self.visualizer = PointCloudSaver()  # 初始化保存器

        obj_low = (0, 0.65, 0.03)
        obj_high = (0, 0.65, 0.03)
        goal_low = (0, 0.85, 0.0)
        goal_high = (0, 0.85, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_pos": np.array([0, 0.6, 0.03]),
            "obj_init_angle": 0.3,
            # "hand_init_pos": np.array([0.0, 0.53, 0.05]),
            "hand_init_pos": np.array([0.0, 0.56, 0.05]),
        }
        self.goal = np.array([0.0, 0.9, 0.03])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_soccer.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        success = float(target_to_obj <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(
            self.touching_object
            and (tcp_opened > 0)
            and (obj[2] - 0.02 > self.obj_init_pos[2])
        )
        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com("soccer_ball")

    def _get_quat_objects(self):
        geom_xmat = self.data.body("soccer_ball").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    # 标记：pointcloud
    def render(self, mode=''):
        """
        渲染环境。

        参数：
            mode (str): 渲染模式，可以是 'human'、'rgb_array'、'depth_array' 或 'pointcloud'。
            width (int, 可选): 渲染宽度。如果为 None，则使用环境的默认宽度。
            height (int, 可选): 渲染高度。如果为 None，则使用环境的默认高度。
            camera_name (str): 摄像机名称。

        返回：
            如果 mode 为 'pointcloud'，返回完整的场景点云数据；否则，返回渲染结果。
        """
        if mode == 'pointcloud':
            extractor = ScenePointCloudExtractor(self.model, self.data, task_related_body_names=["soccer_ball", "goal_whole"])
            point_cloud = extractor.extract_point_cloud()

            # self.visualizer.save_point_cloud(point_cloud)  # 保存点云文件

            # 防止点云不符合要求
            if point_cloud.shape[0] > self.N:
                indices = np.random.choice(point_cloud.shape[0], self.N, replace=False)
                point_cloud = point_cloud[indices]
            elif point_cloud.shape[0] < self.N:
                padding = np.zeros((self.N - point_cloud.shape[0], 6), dtype=np.float32)
                point_cloud = np.vstack((point_cloud, padding))

            return point_cloud
        else:
            return super().render()


    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self.target_pos = self._target_pos
        self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_whole")
        ] = self._target_pos
        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )

        return self._get_obs()

    def _gripper_caging_reward(self, action, obj_position, obj_radius):
        pad_success_margin = 0.05
        grip_success_margin = obj_radius + 0.01
        x_z_success_margin = 0.005

        tcp = self.tcp_center
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")
        delta_object_y_left_pad = left_pad[1] - obj_position[1]
        delta_object_y_right_pad = obj_position[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_position[1] - self.init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_position[1] - self.init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        right_gripping = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_gripping = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        assert right_caging >= 0 and right_caging <= 1
        assert left_caging >= 0 and left_caging <= 1

        y_caging = reward_utils.hamacher_product(right_caging, left_caging)
        y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

        assert y_caging >= 0 and y_caging <= 1

        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_position) + np.array(
            [0.0, -obj_position[1], 0.0]
        )
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
        init_obj_x_z = self.obj_init_pos + np.array([0.0, -self.obj_init_pos[1], 0.0])
        init_tcp_x_z = self.init_tcp + np.array([0.0, -self.init_tcp[1], 0.0])

        tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        )
        x_z_caging = reward_utils.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        assert right_caging >= 0 and right_caging <= 1
        gripper_closed = min(max(0, action[-1]), 1)
        assert gripper_closed >= 0 and gripper_closed <= 1
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)
        assert caging >= 0 and caging <= 1

        if caging > 0.95:
            gripping = y_gripping
        else:
            gripping = 0.0
        assert gripping >= 0 and gripping <= 1

        caging_and_gripping = (caging + gripping) / 2
        assert caging_and_gripping >= 0 and caging_and_gripping <= 1

        return caging_and_gripping

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        tcp_opened = obs[3]
        x_scaling = np.array([3.0, 1.0, 1.0])
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)
        target_to_obj = np.linalg.norm((obj - self._target_pos) * x_scaling)
        target_to_obj_init = np.linalg.norm((obj - self.obj_init_pos) * x_scaling)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )

        goal_line = self._target_pos[1] - 0.1
        if obj[1] > goal_line and abs(obj[0] - self._target_pos[0]) > 0.10:
            in_place = np.clip(
                in_place - 2 * ((obj[1] - goal_line) / (1 - goal_line)), 0.0, 1.0
            )

        object_grasped = self._gripper_caging_reward(action, obj, self.OBJ_RADIUS)

        reward = (3 * object_grasped) + (6.5 * in_place)

        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.0
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            np.linalg.norm(obj - self._target_pos),
            object_grasped,
            in_place,
        )


class TrainSoccerv2(SawyerSoccerEnvV2):
    tasks = None

    def __init__(self):
        SawyerSoccerEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestSoccerv2(SawyerSoccerEnvV2):
    tasks = None

    def __init__(self):
        SawyerSoccerEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
