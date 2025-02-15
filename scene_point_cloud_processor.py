import threading
import mujoco
import numpy as np
import open3d as o3d

class ScenePointCloudExtractor:
    """
    从 MuJoCo 模型和数据中提取场景的完整点云，包括 xyz 坐标和 rgba(可保留 alpha 也可截断)。
    同时支持对 mesh 进行表面采样，以获取稠密点云（如球门网等）。
    
    新增功能：
    - 分配点数时不再随意截断，确保所有几何体均衡分配、总点数完全等于设定值。
    - 扩展 extract_point_cloud 方法，返回一个列表，其中第一个元素为指定点数的点云数组，
      第二个元素为不限制点数的完整点云数组（采样尽可能多的点）。
    - 针对 mesh 的面积计算和采样均采用向量化方法，并添加缓存机制，大幅提高运行效率。
    """

    def __init__(self, model, data,
                 task_related_body_names=None,
                 num_points=8192,
                 random_seed=None):
        """
        初始化 ScenePointCloudExtractor 实例。

        参数：
            model: MuJoCo 模型对象 (mjModel)。
            data: MuJoCo 数据对象 (mjData)。
            task_related_body_names (list): 任务相关的身体名称列表，默认为空列表。
            num_points (int): 期望最终返回的点云总数（严格保证返回的数量就是这个值）。
            random_seed (int): 随机数种子，用于结果复现。
        """
        self.model = model
        self.data = data
        self.task_related_body_names = task_related_body_names if task_related_body_names else []
        self.num_points = num_points
        # 用于无限制点云的密集采样因子，根据需要调整
        self.dense_factor = 30.0  
        if random_seed is not None:
            np.random.seed(random_seed)

        # 构建身体的父子关系映射（便于后续递归遍历）
        self._build_body_tree()
        # 用于缓存 mesh 数据（如顶点、面片、面积等），避免重复计算，提高效率
        self._mesh_cache = {}

    def _build_body_tree(self):
        """
        构建身体的父子关系映射，用于遍历身体树。
        """
        self.body_parentid = self.model.body_parentid
        self.body_children = {i: [] for i in range(self.model.nbody)}
        for i in range(1, self.model.nbody):  # 从 1 开始，跳过世界身体 (body0)
            parent_id = self.body_parentid[i]
            self.body_children[parent_id].append(i)

    def extract_point_cloud(self):
        """
        提取场景的完整点云数据，返回一个长度为2的列表：
            第一个元素为 (N, 6) 的点云数组，其中 N == self.num_points，
            第二个元素为不限制点数的完整点云数组，尽可能多地采样点。
        每个数组的前3列为 xyz 坐标，后3列为 rgb 颜色。
        若想保留 alpha，可自行修改代码以包含第7列。

        返回：
            [limited_point_cloud, dense_point_cloud] 两个点云数组。
        """
        # 获取任务相关的几何体 ID
        task_related_geom_ids = self._get_task_related_geom_ids()
        if not task_related_geom_ids:
            print("[WARN] 未找到任务相关的几何体，返回空数组。")
            empty_array = np.empty((0, 6))
            return [empty_array, empty_array]

        # 计算每个 geom 的表面积（或代表面积），以便按面积比例分配采样点数
        geom_areas = []
        for geom_id in task_related_geom_ids:
            area = self._compute_geom_area(geom_id)
            geom_areas.append((geom_id, area))

        # 针对指定点数的点云分配采样点数
        allocated_pts_list = self._allocate_points(geom_areas, self.num_points)

        # 针对密集点云，使用采样因子扩大采样点数
        dense_total_points = int(self.num_points * self.dense_factor)
        allocated_pts_list_dense = self._allocate_points(geom_areas, dense_total_points)

        # 分别采样得到两个点云数组
        limited_points, limited_colors = self._sample_point_clouds(geom_areas, allocated_pts_list)
        dense_points, dense_colors = self._sample_point_clouds(geom_areas, allocated_pts_list_dense)

        # 合并采样点和颜色，生成最终的 (N,6) 点云数组（前3列 xyz，后3列 rgb）
        if len(limited_points) == 0:
            print("[WARN] 限制点云采样不到点，返回空数组。")
            limited_point_cloud = np.empty((0, 6))
        else:
            limited_all_points = np.vstack(limited_points)    # (M1, 3)
            limited_all_colors = np.vstack(limited_colors)      # (M1, 4)
            limited_point_cloud = np.hstack((limited_all_points, limited_all_colors))[:, :6]

        if len(dense_points) == 0:
            print("[WARN] 密集点云采样不到点，返回空数组。")
            dense_point_cloud = np.empty((0, 6))
        else:
            dense_all_points = np.vstack(dense_points)    # (M2, 3)
            dense_all_colors = np.vstack(dense_colors)      # (M2, 4)
            dense_point_cloud = np.hstack((dense_all_points, dense_all_colors))[:, :6]

        return [limited_point_cloud, dense_point_cloud]

    def _sample_point_clouds(self, geom_areas, allocated_pts_list):
        """
        根据分配的点数列表，对每个 geom 执行采样，并分别收集采样点和颜色数据。

        参数：
            geom_areas: List[(geom_id, area)] 每个几何体及其表面积的列表
            allocated_pts_list: 每个几何体分配到的采样点数列表

        返回：
            all_points, all_colors 两个列表，分别包含采样的点坐标和颜色数据。
        """
        all_points = []
        all_colors = []
        # 遍历每个几何体及其对应的采样点数
        for (geom_id, _), this_geom_num_pts in zip(geom_areas, allocated_pts_list):
            if this_geom_num_pts <= 0:
                # 若该几何体分配的点数为 0，则跳过采样
                continue

            # 获取当前几何体的类型、位置、旋转矩阵、尺寸和颜色
            geom_type = self.model.geom_type[geom_id]
            geom_pos  = self.data.geom_xpos[geom_id]
            geom_xmat = self.data.geom_xmat[geom_id].reshape(3, 3)  # 旋转矩阵 3x3
            geom_size = self.model.geom_size[geom_id]
            geom_rgba = self._get_geom_rgba(geom_id)

            # 根据不同几何体类型调用相应的采样函数
            try:
                if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    vertices_local = self._sample_points_on_box(geom_size, this_geom_num_pts)
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    vertices_local = self._sample_points_on_sphere(geom_size[0], this_geom_num_pts)
                elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    vertices_local = self._sample_points_on_cylinder(geom_size[0], geom_size[1], this_geom_num_pts)
                elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    vertices_local = self._sample_points_on_capsule(geom_size[0], geom_size[1], this_geom_num_pts)
                elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                    vertices_local = self._sample_points_on_ellipsoid(geom_size, this_geom_num_pts)
                elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    vertices_local = self._sample_points_on_mesh(geom_id, this_geom_num_pts)
                else:
                    print(f"[WARN] 不支持的几何体类型: {geom_type}，跳过采样。")
                    continue
            except ValueError as e:
                print(f"[WARN] 处理 geom_id {geom_id} 时出错: {e}")
                continue

            # 将局部坐标转换为全局坐标：X_global = R * X_local + t
            vertices_global = vertices_local @ geom_xmat.T + geom_pos

            # 生成与采样点数匹配的颜色数组（RGBA），这里后续只取前三列作为 RGB
            colors = np.tile(geom_rgba, (vertices_global.shape[0], 1))  # shape=(N,4)

            all_points.append(vertices_global)
            all_colors.append(colors)

        return all_points, all_colors

    def _allocate_points(self, geom_areas, total_points):
        """
        根据每个几何体的面积及类型，精确分配采样点数，确保总数严格等于 total_points。
        同时保证每个几何体至少获得一定的最低采样点数。

        参数：
            geom_areas: List[(geom_id, area)] 每个几何体及其表面积列表
            total_points: int 总采样点数

        返回：
            与 geom_areas 长度相同的 numpy 数组 allocated_pts_list，每个元素为对应几何体的采样点数。
        """
        num_geoms = len(geom_areas)

        # 设定每个几何体的最小采样点数，至少为300个或总点数的1%
        min_points_per_geom = np.array([
            max(300, int(0.01 * total_points)) for _ in range(num_geoms)
        ])

        # 计算剩余可分配点数
        leftover_points = total_points - np.sum(min_points_per_geom)

        # 按照面积比例分配剩余的采样点数
        total_area = sum(area for _, area in geom_areas)
        proportional_points = [
            int(leftover_points * (area / total_area)) for _, area in geom_areas
        ]

        # 得到初步分配结果，并调整使总和严格等于 total_points
        allocated_points = min_points_per_geom + np.array(proportional_points)
        allocated_points = self._adjust_allocation_to_match_total(allocated_points, total_points)

        return allocated_points

    def _adjust_allocation_to_match_total(self, allocated_points, total_points):
        """
        调整采样点分配，使得所有几何体采样点总和严格等于 total_points。
        若总数不足或超出，则随机选择一些几何体进行增减调整。

        参数：
            allocated_points: numpy 数组，初步分配的采样点数列表
            total_points: int 目标采样点数

        返回：
            调整后的 numpy 数组，元素总和等于 total_points。
        """
        current_total = np.sum(allocated_points)
        difference = total_points - current_total

        if difference == 0:
            return allocated_points

        # 随机选择需要调整的几何体索引，逐个加减1直至满足总数要求
        adjustment_indices = np.random.choice(
            range(len(allocated_points)),
            size=abs(difference),
            replace=True
        )
        for idx in adjustment_indices:
            allocated_points[idx] += np.sign(difference)

        return allocated_points

    def _get_geom_rgba(self, geom_id):
        """
        结合 geom_rgba 与材质属性（emission, reflectance, specular, shininess）对颜色进行融合处理，
        返回 (4,) 数组 [r, g, b, a]（颜色值均在 [0,1] 范围内）。

        参数：
            geom_id: int 几何体索引

        返回：
            final_rgba: np.array，融合后的颜色值
        """
        # 将几何体颜色裁剪至 [0,1] 范围
        geom_rgba = np.clip(self.model.geom_rgba[geom_id], 0, 1)

        material_id = self.model.geom_matid[geom_id]
        if material_id < 0:
            # 若无材质信息，则直接返回几何体颜色
            return geom_rgba

        # 读取材质属性并裁剪颜色值
        material_rgba = np.clip(self.model.mat_rgba[material_id], 0, 1)
        shininess    = self.model.mat_shininess[material_id]
        reflectance  = self.model.mat_reflectance[material_id]
        specular     = self.model.mat_specular[material_id]
        emission     = self.model.mat_emission[material_id]

        alpha_geom = geom_rgba[3]
        alpha_mat  = material_rgba[3]
        final_alpha = alpha_geom * alpha_mat

        # 基础颜色：几何体颜色与材质颜色的乘积
        base_rgb = geom_rgba[:3] * material_rgba[:3]

        # 考虑自发光，使颜色加深（增加白色成分）
        if emission > 1e-9:
            base_rgb += emission * np.array([1, 1, 1])
            base_rgb = np.clip(base_rgb, 0, 1)

        # 考虑反射率，使颜色向白色偏移
        if reflectance > 1e-9:
            base_rgb = (1 - reflectance) * base_rgb + reflectance * np.array([1, 1, 1])
            base_rgb = np.clip(base_rgb, 0, 1)

        # 考虑高光效应
        if specular > 1e-9:
            base_rgb += specular * np.array([1, 1, 1])
            base_rgb = np.clip(base_rgb, 0, 1)

        # 简化处理：考虑光滑度（shininess）对颜色的微调
        if shininess > 1e-9:
            base_rgb = base_rgb * (1 - 0.5 * shininess) + 0.5 * shininess * np.array([1, 1, 1])
            base_rgb = np.clip(base_rgb, 0, 1)

        final_rgba = np.array([base_rgb[0], base_rgb[1], base_rgb[2], final_alpha], dtype=np.float32)
        return final_rgba

    def _get_task_related_geom_ids(self):
        """
        获取任务相关的几何体索引，支持指定 body 及其所有子孙 body 关联的几何体。
        遍历每个指定 body 名称，递归收集其及所有子孙的 geom 索引。

        返回：
            去重后的任务相关 geom 索引列表。
        """
        task_related_geom_ids = []
        for body_name in self.task_related_body_names:
            try:
                # 根据名称获取 body 索引
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            except ValueError:
                print(f"[WARN] 模型中未找到 body 名称 '{body_name}'。")
                continue

            geom_ids = self._collect_geom_ids(body_id)
            task_related_geom_ids.extend(geom_ids)

        # 去重，避免重复采样同一几何体
        return list(set(task_related_geom_ids))

    def _collect_geom_ids(self, body_id):
        """
        递归收集指定 body 及其所有子孙 body 的 geom 索引。

        参数：
            body_id: int 指定 body 的索引

        返回：
            geom_ids: list 收集到的几何体索引列表。
        """
        geom_ids = []
        geom_start = self.model.body_geomadr[body_id]
        geom_num = self.model.body_geomnum[body_id]
        if geom_num > 0:
            geom_ids.extend(range(geom_start, geom_start + geom_num))
        # 递归处理子孙 body
        child_body_ids = self.body_children.get(body_id, [])
        for child_body_id in child_body_ids:
            geom_ids.extend(self._collect_geom_ids(child_body_id))
        return geom_ids

    def _compute_geom_area(self, geom_id):
        """
        根据几何体类型近似计算其表面积。
        对于 mesh 类型几何体，使用实际面片面积（向量化计算）进行精确计算，并利用缓存机制加速。

        参数：
            geom_id: int 几何体索引

        返回：
            面积值（float）
        """
        geom_type = self.model.geom_type[geom_id]
        size = self.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            x, y, z = size
            return 2.0 * (x * y + y * z + x * z)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = size[0]
            return 4.0 * np.pi * r**2
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = size[0], size[1]
            return 2.0 * np.pi * r * h + 2.0 * np.pi * r * r
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r, h = size[0], size[1]
            return 2.0 * np.pi * r * h + 4.0 * np.pi * r * r
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            a, b, c = size
            p = 1.6  # 近似公式指数参数
            t = (a**p * b**p + b**p * c**p + a**p * c**p) / 3.0
            return 4.0 * np.pi * (t**(1.0/p))
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            return self._compute_mesh_area(geom_id)
        else:
            return 0.0

    def _compute_mesh_area(self, geom_id):
        """
        针对 mesh 类型几何体，通过遍历所有三角面片（向量化操作）计算真实表面积，
        并使用缓存机制保存计算结果，避免重复计算同一 mesh。

        参数：
            geom_id: int 几何体索引

        返回：
            total_area: float 该 mesh 的总表面积
        """
        mesh_id = self.model.geom_dataid[geom_id]
        # 检查缓存中是否已有面积数据
        if mesh_id in self._mesh_cache and 'area' in self._mesh_cache[mesh_id]:
            return self._mesh_cache[mesh_id]['area']

        # 获取 mesh 的顶点和面片数据
        vert_adr = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        face_adr = self.model.mesh_faceadr[mesh_id]
        face_num = self.model.mesh_facenum[mesh_id]
        vertices = self.model.mesh_vert[vert_adr : vert_adr + vert_num].reshape(-1, 3)
        faces = self.model.mesh_face[face_adr : face_adr + face_num].reshape(-1, 3)

        # 向量化计算每个三角形面积
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
        total_area = np.sum(areas)

        # 缓存面积以及顶点、面片数据
        if mesh_id not in self._mesh_cache:
            self._mesh_cache[mesh_id] = {}
        self._mesh_cache[mesh_id]['area'] = total_area
        self._mesh_cache[mesh_id]['vertices'] = vertices
        self._mesh_cache[mesh_id]['faces'] = faces

        return total_area

    def _sample_points_on_box(self, size, n):
        """
        在盒子表面随机均匀采样 n 个点。

        参数：
            size: tuple, 盒子的半尺寸 (x, y, z)
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于盒子表面。
        """
        x, y, z = size
        # 计算盒子六个面的面积（前提：相对对称的两个面面积相同）
        areas = np.array([y * z, x * z, x * y])
        areas = np.concatenate([areas, areas])  # 六个面
        total_area = np.sum(areas)
        # 按比例计算每个面应分配的采样点数（向下取整）
        face_points = (n * areas / total_area).astype(int)

        # 由于取整可能使总数小于 n，根据余数进行补充分配
        allocated_so_far = face_points.sum()
        leftover = n - allocated_so_far
        if leftover > 0:
            remainders = (n * areas / total_area) - face_points
            sorted_idx = np.argsort(-remainders)  # 余数从大到小排序
            face_points[sorted_idx[:leftover]] += 1

        points = []
        # 针对 x 方向的两个面采样
        for sign in [-1, 1]:
            nn = face_points[0]
            ys = np.random.uniform(-y, y, nn)
            zs = np.random.uniform(-z, z, nn)
            xs = np.full(nn, sign * x)
            points.append(np.vstack((xs, ys, zs)).T)
            face_points = np.roll(face_points, -1)

        # 针对 y 方向的两个面采样
        for sign in [-1, 1]:
            nn = face_points[0]
            xs = np.random.uniform(-x, x, nn)
            zs = np.random.uniform(-z, z, nn)
            ys = np.full(nn, sign * y)
            points.append(np.vstack((xs, ys, zs)).T)
            face_points = np.roll(face_points, -1)

        # 针对 z 方向的两个面采样
        for sign in [-1, 1]:
            nn = face_points[0]
            xs = np.random.uniform(-x, x, nn)
            ys = np.random.uniform(-y, y, nn)
            zs = np.full(nn, sign * z)
            points.append(np.vstack((xs, ys, zs)).T)
            face_points = np.roll(face_points, -1)

        return np.vstack(points)

    def _sample_points_on_sphere(self, radius, n):
        """
        在球面上随机均匀采样 n 个点。

        参数：
            radius: float, 球的半径
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于球面上。
        """
        phi = np.arccos(1 - 2 * np.random.rand(n))
        theta = 2 * np.pi * np.random.rand(n)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return np.column_stack((x, y, z))

    def _sample_points_on_cylinder(self, radius, height, n):
        """
        在圆柱表面随机均匀采样 n 个点，包括侧面以及上下两个端面。

        参数：
            radius: float, 圆柱半径
            height: float, 圆柱高度
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于圆柱表面。
        """
        # 计算侧面和端面的面积
        side_area = 2 * np.pi * radius * height
        cap_area  = np.pi * radius**2
        areas = np.array([side_area, 2 * cap_area])
        total = np.sum(areas)

        # 根据面积比例分配侧面与端面采样点数
        side_count = int(n * side_area / total)
        leftover = n - side_count
        top_count = leftover // 2
        bot_count = leftover - top_count

        # 侧面采样（均匀分布角度与高度）
        theta = 2 * np.pi * np.random.rand(side_count)
        zvals = np.random.uniform(-height/2, height/2, side_count)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        side_points = np.column_stack((x, y, zvals))

        # 顶面采样（极坐标均匀分布）
        r_top = np.sqrt(np.random.rand(top_count)) * radius
        theta_top = 2 * np.pi * np.random.rand(top_count)
        x_top = r_top * np.cos(theta_top)
        y_top = r_top * np.sin(theta_top)
        z_top = np.full(top_count, height/2)
        top_points = np.column_stack((x_top, y_top, z_top))

        # 底面采样
        r_bot = np.sqrt(np.random.rand(bot_count)) * radius
        theta_bot = 2 * np.pi * np.random.rand(bot_count)
        x_bot = r_bot * np.cos(theta_bot)
        y_bot = r_bot * np.sin(theta_bot)
        z_bot = np.full(bot_count, -height/2)
        bot_points = np.column_stack((x_bot, y_bot, z_bot))

        return np.vstack((side_points, top_points, bot_points))

    def _sample_points_on_capsule(self, radius, height, n):
        """
        在胶囊体表面随机均匀采样 n 个点，包括中间圆柱部分及两端半球部分。

        参数：
            radius: float, 胶囊体半径
            height: float, 圆柱部分高度
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于胶囊体表面。
        """
        # 计算圆柱部分与球部分的面积
        cylinder_area = 2 * np.pi * radius * height
        sphere_area   = 4 * np.pi * radius**2
        total_area = cylinder_area + sphere_area

        # 按面积比例分配采样点数
        cyl_count = int(n * (cylinder_area / total_area))
        sph_count = n - cyl_count

        # 圆柱部分采样
        cylinder_pts = self._sample_points_on_cylinder(radius, height, cyl_count)

        # 球面部分采样，并分为上半球与下半球后平移到对应位置
        sphere_pts = self._sample_points_on_sphere(radius, sph_count)
        half = sph_count // 2
        top_sphere = sphere_pts[:half] + np.array([0, 0, height/2])
        bottom_sphere = sphere_pts[half:] + np.array([0, 0, -height/2])

        return np.vstack((cylinder_pts, top_sphere, bottom_sphere))

    def _sample_points_on_ellipsoid(self, size, n):
        """
        在椭球表面随机均匀采样 n 个点。

        参数：
            size: tuple, 椭球半轴长度 (a, b, c)
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于椭球表面。
        """
        a, b, c = size
        phi = np.arccos(1 - 2 * np.random.rand(n))
        theta = 2 * np.pi * np.random.rand(n)
        x = a * np.sin(phi) * np.cos(theta)
        y = b * np.sin(phi) * np.sin(theta)
        z = c * np.cos(phi)
        return np.column_stack((x, y, z))

    def _sample_points_on_mesh(self, geom_id, n):
        """
        在 mesh 表面随机采样 n 个点。
        采用向量化方法：先根据各面片面积计算多项分布的采样点数，再对所有采样点统一生成随机 barycentric 坐标。

        参数：
            geom_id: int, mesh 几何体的索引
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于 mesh 表面。
        """
        mesh_id = self.model.geom_dataid[geom_id]
        # 尝试从缓存中获取顶点和面片数据
        if mesh_id in self._mesh_cache and 'vertices' in self._mesh_cache[mesh_id] and 'faces' in self._mesh_cache[mesh_id]:
            vertices = self._mesh_cache[mesh_id]['vertices']
            faces = self._mesh_cache[mesh_id]['faces']
        else:
            vert_adr = self.model.mesh_vertadr[mesh_id]
            vert_num = self.model.mesh_vertnum[mesh_id]
            face_adr = self.model.mesh_faceadr[mesh_id]
            face_num = self.model.mesh_facenum[mesh_id]
            vertices = self.model.mesh_vert[vert_adr : vert_adr + vert_num].reshape(-1, 3)
            faces = self.model.mesh_face[face_adr : face_adr + face_num].reshape(-1, 3)
            if mesh_id not in self._mesh_cache:
                self._mesh_cache[mesh_id] = {}
            self._mesh_cache[mesh_id]['vertices'] = vertices
            self._mesh_cache[mesh_id]['faces'] = faces

        # 向量化计算每个面片的面积
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        face_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
        total_area = face_areas.sum()

        if total_area < 1e-12:
            # 若面片面积总和极小，直接返回部分顶点，避免除零错误
            return vertices[:min(len(vertices), n)]

        # 计算每个面片的采样概率
        face_probs = face_areas / total_area
        # 根据多项分布分配各面片采样点数，确保总数为 n
        face_counts = np.random.multinomial(n, face_probs)

        # 向量化采样：对每个面片重复其索引 face_counts 次
        face_indices = np.repeat(np.arange(len(faces)), face_counts)
        total_sampled = face_indices.shape[0]
        # 对所有采样点生成随机 barycentric 坐标
        u = np.random.rand(total_sampled)
        v = np.random.rand(total_sampled)
        mask = (u + v) > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        # 根据重复后的面片索引获得各三角形顶点，并计算采样点坐标：P = v0 + u*(v1-v0) + v*(v2-v0)
        f_selected = faces[face_indices]
        v0_sel = vertices[f_selected[:, 0]]
        v1_sel = vertices[f_selected[:, 1]]
        v2_sel = vertices[f_selected[:, 2]]
        sampled_points = v0_sel + np.expand_dims(u, axis=1) * (v1_sel - v0_sel) + np.expand_dims(v, axis=1) * (v2_sel - v0_sel)
        return sampled_points

class PointCloudSaver:
    """
    用于保存点云的工具类。
    提供将点云数据保存为 PLY 文件的功能，便于后续可视化或分析。
    """

    def __init__(self, filename='/home/star/Yanjun/RL-VLM-F/html/point_cloud-ori.ply'):
        """
        初始化 PointCloudSaver 实例。

        参数：
            filename (str): 点云文件的保存路径。
        """
        self.filename = filename

    def save_point_cloud(self, point_cloud):
        """
        异步保存点云数据到指定文件。

        参数：
            point_cloud: numpy 数组，形状为 (N,6)，前3列为 xyz 坐标，后3列为 rgb 颜色。
        """
        threading.Thread(target=self._save_point_cloud_file, args=(point_cloud,)).start()

    def _save_point_cloud_file(self, point_cloud):
        """
        实际执行点云文件保存操作的内部方法。

        参数：
            point_cloud: numpy 数组，形状为 (N,6)。
        """
        try:
            pcd = o3d.geometry.PointCloud()
            # 设置点云坐标信息
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            # 设置点云颜色信息
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
            # 写入点云到指定文件
            o3d.io.write_point_cloud(self.filename, pcd)
        except Exception as e:
            print(f"保存点云时发生错误: {e}")
