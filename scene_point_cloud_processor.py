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
    """

    def __init__(self, model, data,
                 task_related_body_names=None,
                 num_points=8192,
                 random_seed=None):
        """
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
        # 用于无限制点云的密集采样因子，可以根据需要调整
        self.dense_factor = 100  
        if random_seed is not None:
            np.random.seed(random_seed)

        # 构建身体的父子关系映射
        self._build_body_tree()

    def _build_body_tree(self):
        """构建身体的父子关系映射，用于遍历身体树。"""
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
        若想保留 alpha，可自行修改代码以包括第7列。
        """
        # 获取任务相关的几何体 ID
        task_related_geom_ids = self._get_task_related_geom_ids()
        if not task_related_geom_ids:
            print("[WARN] No task-related geoms found. Return empty arrays.")
            empty_array = np.empty((0, 6))
            return [empty_array, empty_array]

        # 计算每个 geom 的表面积（或代表面积），并做分配
        geom_areas = []
        for geom_id in task_related_geom_ids:
            area = self._compute_geom_area(geom_id)
            geom_areas.append((geom_id, area))

        # 针对指定点数的点云分配点数
        allocated_pts_list = self._allocate_points(geom_areas, self.num_points)

        # 针对不限制点数的点云，使用密集采样因子扩大采样点数
        dense_total_points = int(self.num_points * self.dense_factor)
        allocated_pts_list_dense = self._allocate_points(geom_areas, dense_total_points)

        # 分别采样两个点云数组
        limited_points, limited_colors = self._sample_point_clouds(geom_areas, allocated_pts_list)
        dense_points,   dense_colors   = self._sample_point_clouds(geom_areas, allocated_pts_list_dense)

        # 合并并处理第一个点云数组
        if len(limited_points) == 0:
            print("[WARN] No points sampled for limited point cloud. Return empty array.")
            limited_point_cloud = np.empty((0, 6))
        else:
            limited_all_points = np.vstack(limited_points)    # (M1, 3)
            limited_all_colors = np.vstack(limited_colors)    # (M1, 4)
            limited_point_cloud = np.hstack((limited_all_points, limited_all_colors))[:, :6]  # (M1,6)

        # 合并并处理密集点云数组
        if len(dense_points) == 0:
            print("[WARN] No points sampled for dense point cloud. Return empty array.")
            dense_point_cloud = np.empty((0, 6))
        else:
            dense_all_points = np.vstack(dense_points)    # (M2, 3)
            dense_all_colors = np.vstack(dense_colors)    # (M2, 4)
            dense_point_cloud = np.hstack((dense_all_points, dense_all_colors))[:, :6]  # (M2,6)

        # 返回包含两个点云数组的列表
        return [limited_point_cloud, dense_point_cloud]

    def _sample_point_clouds(self, geom_areas, allocated_pts_list):
        """
        根据分配的点数列表，对每个 geom 执行采样，并分别收集点和颜色。
        返回两个列表：all_points 和 all_colors，其中每个元素对应一个 geom 的采样点和颜色数组。
        """
        all_points = []
        all_colors = []
        # 使用 zip 来遍历 geom_areas 和对应的分配点数
        for (geom_id, _), this_geom_num_pts in zip(geom_areas, allocated_pts_list):
            if this_geom_num_pts <= 0:
                # 如果该几何体分配的点数为0，则跳过采样
                continue

            geom_type = self.model.geom_type[geom_id]
            geom_pos  = self.data.geom_xpos[geom_id]
            geom_xmat = self.data.geom_xmat[geom_id].reshape(3, 3)  # 旋转矩阵
            geom_size = self.model.geom_size[geom_id]
            geom_rgba = self._get_geom_rgba(geom_id)

            # 根据不同的几何体类型进行相应的采样
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
                    print(f"[WARN] Unsupported geom type: {geom_type}, skipping.")
                    continue
            except ValueError as e:
                print(f"[WARN] Error processing geom_id {geom_id}: {e}")
                continue

            # 将局部坐标转换为全局坐标
            vertices_global = vertices_local @ geom_xmat.T + geom_pos

            # 生成颜色数组，与采样点数量匹配
            colors = np.tile(geom_rgba, (vertices_global.shape[0], 1))  # shape=(N,4)

            all_points.append(vertices_global)
            all_colors.append(colors)

        return all_points, all_colors

    def _allocate_points(self, geom_areas, total_points):
        """
        根据每个 geom 的面积和类型，精准地给它们分配点数，使得分配后总数 == total_points。
        同时，确保每个 geom 至少分配到一个合理的最低点数。

        参数：
            geom_areas: List[(geom_id, area)] 每个几何体及其表面积的列表
            total_points: int 总共需要分配的点数

        返回：
            一个与 geom_areas 长度相同的 numpy 数组 allocated_pts_list，
            其中每个元素表示对应几何体分配到的点数。
        """
        num_geoms = len(geom_areas)

        # 1) 设置每个物体的最小点数（基于几何体类型调整，此处统一设定为至少1%或300个点）
        min_points_per_geom = np.array([
            max(300, int(0.01 * total_points))  for _ in range(num_geoms)
        ])

        # 2) 计算剩余的可分配点数
        leftover_points = total_points - np.sum(min_points_per_geom)

        # 3) 按面积比例分配剩余点数
        total_area = sum(area for _, area in geom_areas)
        proportional_points = [
            int(leftover_points * (area / total_area))  for _, area in geom_areas
        ]

        # 4) 确保最终点数总和与 total_points 一致
        allocated_points = min_points_per_geom + proportional_points
        allocated_points = self._adjust_allocation_to_match_total(allocated_points, total_points)

        return allocated_points

    def _adjust_allocation_to_match_total(self, allocated_points, total_points):
        """
        调整分配的点数，使总和恰好等于 total_points。
        如果超出或不足，则随机选择一些几何体增减点数。

        参数：
            allocated_points: numpy 数组，初步分配的点数列表
            total_points: int 目标总点数

        返回：
            调整后的 numpy 数组，使其元素总和等于 total_points
        """
        current_total = np.sum(allocated_points)
        difference = total_points - current_total

        if difference == 0:
            return allocated_points

        # 随机选择若干几何体进行调整
        adjustment_indices = np.random.choice(
            range(len(allocated_points)),
            size=abs(difference),
            replace=True
        )

        for idx in adjustment_indices:
            allocated_points[idx] += np.sign(difference)

        return allocated_points

    # ============= 获取最终 RGBA =============
    def _get_geom_rgba(self, geom_id):
        """
        考虑 geom_rgba 与 material (emission, reflectance, specular, shininess) 做一次近似融合。
        返回 (4,) => [r,g,b,a]，颜色值范围在 [0,1] 之间。
        """
        geom_rgba = np.clip(self.model.geom_rgba[geom_id], 0, 1)

        material_id = self.model.geom_matid[geom_id]
        if material_id < 0:
            # 没有材质则直接返回几何体颜色
            return geom_rgba

        # 读取材质属性
        material_rgba = np.clip(self.model.mat_rgba[material_id], 0, 1)
        shininess    = self.model.mat_shininess[material_id]
        reflectance  = self.model.mat_reflectance[material_id]
        specular     = self.model.mat_specular[material_id]
        emission     = self.model.mat_emission[material_id]

        alpha_geom = geom_rgba[3]
        alpha_mat  = material_rgba[3]
        final_alpha = alpha_geom * alpha_mat

        # 基础色彩计算
        base_rgb = geom_rgba[:3] * material_rgba[:3]

        # 考虑自发光的影响
        if emission > 1e-9:
            base_rgb += emission * np.array([1,1,1])
            base_rgb = np.clip(base_rgb, 0, 1)

        # 考虑反射率影响，使颜色向白色靠拢
        if reflectance > 1e-9:
            base_rgb = (1 - reflectance)*base_rgb + reflectance*np.array([1,1,1])
            base_rgb = np.clip(base_rgb, 0, 1)

        # 考虑高光影响，进一步增加白色成分
        if specular > 1e-9:
            base_rgb += specular*np.array([1,1,1])
            base_rgb = np.clip(base_rgb, 0, 1)

        # 考虑光滑度对颜色的微调（非常简化处理）
        if shininess > 1e-9:
            base_rgb = base_rgb*(1 - 0.5*shininess) + 0.5*shininess*np.array([1,1,1])
            base_rgb = np.clip(base_rgb, 0, 1)

        final_rgba = np.array([base_rgb[0], base_rgb[1], base_rgb[2], final_alpha], dtype=np.float32)
        return final_rgba

    def _get_task_related_geom_ids(self):
        """
        获取任务相关的几何体索引，支持 body 及其子孙 body 的所有 geom。
        遍历每个指定的 body 名称，收集其及其所有子孙 body 关联的几何体 ID。
        """
        task_related_geom_ids = []
        for body_name in self.task_related_body_names:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            except ValueError:
                print(f"[WARN] Body name '{body_name}' not found in model.")
                continue

            geom_ids = self._collect_geom_ids(body_id)
            task_related_geom_ids.extend(geom_ids)

        # 去重，避免重复采样同一个几何体
        return list(set(task_related_geom_ids))

    def _collect_geom_ids(self, body_id):
        """
        递归收集指定 body 及其子孙 body 的 geom ID。

        参数：
            body_id: int 指定 body 的 ID

        返回：
            geom_ids: list 收集到的几何体 ID 列表
        """
        geom_ids = []
        geom_start = self.model.body_geomadr[body_id]
        geom_num = self.model.body_geomnum[body_id]
        if geom_num > 0:
            geom_ids.extend(range(geom_start, geom_start + geom_num))

        child_body_ids = self.body_children.get(body_id, [])
        for child_body_id in child_body_ids:
            geom_ids.extend(self._collect_geom_ids(child_body_id))
        return geom_ids

    # ------------------ 面积计算 ------------------
    def _compute_geom_area(self, geom_id):
        """
        根据 geom 类型近似计算表面积。
        对于 mesh 类型的几何体，使用实际面片面积进行精确计算。
        """
        geom_type = self.model.geom_type[geom_id]
        size = self.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            x, y, z = size
            return 2.0 * (x*y + y*z + x*z)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = size[0]
            return 4.0 * np.pi * r**2
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = size[0], size[1]
            return 2.0 * np.pi * r * h + 2.0 * np.pi * r*r
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r, h = size[0], size[1]
            return 2.0 * np.pi * r * h + 4.0 * np.pi * r*r
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            a, b, c = size
            p = 1.6
            t = (a**p * b**p + b**p * c**p + a**p * c**p) / 3.0
            return 4.0 * np.pi * (t**(1.0/p))
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            return self._compute_mesh_area(geom_id)
        else:
            return 0.0

    def _compute_mesh_area(self, geom_id):
        """
        对于 mesh 类型的几何体，通过遍历其三角面片计算真实表面积。

        参数：
            geom_id: int 几何体的 ID

        返回：
            total_area: float 该 mesh 的总表面积
        """
        mesh_id = self.model.geom_dataid[geom_id]
        vert_adr = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        face_adr = self.model.mesh_faceadr[mesh_id]
        face_num = self.model.mesh_facenum[mesh_id]

        vertices = self.model.mesh_vert[vert_adr : vert_adr + vert_num].reshape(-1, 3)
        faces = self.model.mesh_face[face_adr : face_adr + face_num].reshape(-1, 3)

        total_area = 0.0
        for f in faces:
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            total_area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        return total_area

    # ------------------ 各种几何采样 ------------------
    def _sample_points_on_box(self, size, n):
        """
        在盒子表面随机均匀采样 n 个点。
        
        参数：
            size: tuple, 盒子的半尺寸 (x, y, z)
            n: int, 需要采样的点数

        返回：
            points: (n,3) 的采样点数组，位于盒子表面
        """
        x, y, z = size
        areas = np.array([y*z, x*z, x*y])  # 三个面对称面
        areas = np.concatenate([areas, areas])  # 六个面
        total_area = np.sum(areas)
        # 计算每个面的采样点数
        face_points = (n * areas / total_area).astype(int)

        # 因为取整可能导致总数小于 n，所以根据余数补足
        allocated_so_far = face_points.sum()
        leftover = n - allocated_so_far
        if leftover > 0:
            remainders = (n * areas / total_area) - face_points
            sorted_idx = np.argsort(-remainders)  # 按降序排序余数
            face_points[sorted_idx[:leftover]] += 1

        points = []
        # x-face
        for sign in [-1, 1]:
            nn = face_points[0]
            ys = np.random.uniform(-y, y, nn)
            zs = np.random.uniform(-z, z, nn)
            xs = np.full(nn, sign * x)
            points.append(np.vstack((xs, ys, zs)).T)
            face_points = np.roll(face_points, -1)

        # y-face
        for sign in [-1, 1]:
            nn = face_points[0]
            xs = np.random.uniform(-x, x, nn)
            zs = np.random.uniform(-z, z, nn)
            ys = np.full(nn, sign * y)
            points.append(np.vstack((xs, ys, zs)).T)
            face_points = np.roll(face_points, -1)

        # z-face
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
            (n,3) 的采样点数组，位于球面上
        """
        phi = np.arccos(1 - 2 * np.random.rand(n))
        theta = 2 * np.pi * np.random.rand(n)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return np.column_stack((x, y, z))

    def _sample_points_on_cylinder(self, radius, height, n):
        """
        在圆柱表面随机均匀采样 n 个点，包含侧面和上下两个端面。

        参数：
            radius: float, 圆柱半径
            height: float, 圆柱高度
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于圆柱表面
        """
        side_area = 2 * np.pi * radius * height
        cap_area  = np.pi * radius**2
        areas = np.array([side_area, 2*cap_area])
        total = np.sum(areas)

        # 计算侧面应采样的点数
        side_count = int(n * side_area / total)
        leftover = n - side_count
        # 将剩余的点数平均分配给上下两个端面
        top_count = leftover // 2
        bot_count = leftover - top_count

        # 侧面采样
        theta = 2 * np.pi * np.random.rand(side_count)
        zvals = np.random.uniform(-height/2, height/2, side_count)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        side_points = np.column_stack((x, y, zvals))

        # 顶面采样
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
        在胶囊体表面随机均匀采样 n 个点，包含圆柱部分和两端半球部分。

        参数：
            radius: float, 胶囊体半径
            height: float, 圆柱部分的高度
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于胶囊表面
        """
        cylinder_area = 2 * np.pi * radius * height
        sphere_area   = 4 * np.pi * radius**2
        total_area = cylinder_area + sphere_area

        # 按比例分配点数给圆柱部分和球面部分
        cyl_count = int(n * (cylinder_area / total_area))
        sph_count = n - cyl_count

        # 圆柱部分采样
        cylinder_pts = self._sample_points_on_cylinder(radius, height, cyl_count)

        # 对球面部分进行采样并分割成上下半球
        sphere_pts = self._sample_points_on_sphere(radius, sph_count)
        half = sph_count // 2
        top_sphere    = sphere_pts[:half] + np.array([0, 0,  height/2])
        bottom_sphere = sphere_pts[half:] + np.array([0, 0, -height/2])

        return np.vstack((cylinder_pts, top_sphere, bottom_sphere))

    def _sample_points_on_ellipsoid(self, size, n):
        """
        在椭球表面随机均匀采样 n 个点。

        参数：
            size: tuple, 椭球的半轴长度 (a, b, c)
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于椭球表面
        """
        a, b, c = size
        phi = np.arccos(1 - 2*np.random.rand(n))
        theta = 2*np.pi*np.random.rand(n)
        x = a * np.sin(phi)*np.cos(theta)
        y = b * np.sin(phi)*np.sin(theta)
        z = c * np.cos(phi)
        return np.column_stack((x, y, z))

    def _sample_points_on_mesh(self, geom_id, n):
        """
        在 mesh 表面随机采样 n 个点。

        参数：
            geom_id: int, mesh 几何体的 ID
            n: int, 需要采样的点数

        返回：
            (n,3) 的采样点数组，位于 mesh 表面
        """
        mesh_id = self.model.geom_dataid[geom_id]
        vert_adr = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        face_adr = self.model.mesh_faceadr[mesh_id]
        face_num = self.model.mesh_facenum[mesh_id]

        vertices = self.model.mesh_vert[vert_adr : vert_adr + vert_num].reshape(-1, 3)
        faces = self.model.mesh_face[face_adr : face_adr + face_num].reshape(-1, 3)

        # 计算每个面片的面积
        face_areas = np.zeros(face_num, dtype=np.float64)
        for i, f in enumerate(faces):
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            face_areas[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        total_area = face_areas.sum()
        if total_area < 1e-12:
            # 如果 mesh 面积退化，则直接返回部分顶点以避免除零错误
            return vertices[:min(len(vertices), n)]

        # 根据面片面积计算采样概率
        face_probs = face_areas / total_area
        face_counts = np.random.multinomial(n, face_probs)

        sampled_points = []
        for face_idx, count in enumerate(face_counts):
            if count <= 0:
                continue
            f = faces[face_idx]
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]

            # 在三角形面内均匀采样
            u = np.random.rand(count)
            v = np.random.rand(count)
            is_over = (u + v) > 1.0
            u[is_over] = 1 - u[is_over]
            v[is_over] = 1 - v[is_over]
            tri_pts = v0[None, :] + u[:, None]*(v1 - v0) + v[:, None]*(v2 - v0)
            sampled_points.append(tri_pts)

        if len(sampled_points) == 0:
            return np.empty((0, 3))

        return np.vstack(sampled_points)

class PointCloudSaver:
    """
    用于保存点云的工具类。
    提供将点云数据保存为 PLY 文件的功能，以便后续可视化或分析。
    """

    def __init__(self, filename='/home/star/Yanjun/RL-VLM-F/html/point_cloud-ori.ply'):
        """
        初始化 PointCloudSaver 实例。

        参数:
            filename (str): 保存点云文件的路径。
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
            point_cloud: numpy 数组，形状为 (N,6)
        """
        try:
            pcd = o3d.geometry.PointCloud()
            # 提取点的坐标信息
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            # 提取点的颜色信息
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:] )
            # 写入点云到文件
            o3d.io.write_point_cloud(self.filename, pcd)
        except Exception as e:
            print(f"Error saving point cloud: {e}")
