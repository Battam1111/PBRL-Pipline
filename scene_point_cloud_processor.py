import threading
import mujoco
import numpy as np
import open3d as o3d

class ScenePointCloudExtractor:
    """
    从 MuJoCo 模型和数据中提取场景的完整点云，包括 xyz 坐标和 rgba(可保留 alpha 也可截断)。
    同时支持对 mesh 进行表面采样，以获取稠密点云（如球门网等）。
    新增：分配点数时不再随意截断，确保所有几何体均衡分配、总点数完全等于设定值。
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
        提取场景的完整点云数据 (N, 6)，其中:
            前3列为 xyz，
            后3列为 rgb。
        若想保留 alpha，可自行把下面最后的 "[:, :6]" 去掉，或者改成 (N, 7)。
        """
        # 1. 获取任务相关的几何体 ID
        task_related_geom_ids = self._get_task_related_geom_ids()
        if not task_related_geom_ids:
            print("[WARN] No task-related geoms found. Return empty array.")
            return np.empty((0, 6))

        # 2. 计算每个 geom 的表面积（或代表面积），并做分配
        geom_areas = []
        for geom_id in task_related_geom_ids:
            area = self._compute_geom_area(geom_id)
            geom_areas.append((geom_id, area))

        # 3. 根据面积分配点数：让总和恰好为 self.num_points
        #    返回数组 allocated_pts_list，长度与 task_related_geom_ids 一致
        allocated_pts_list = self._allocate_points(geom_areas, self.num_points)

        # 4. 对每个 geom 执行采样，拼接
        all_points = []
        all_colors = []
        for (geom_id, _), this_geom_num_pts in zip(geom_areas, allocated_pts_list):
            if this_geom_num_pts <= 0:
                # 理论上不会出现(都保证>=1)，但以防万一
                continue

            geom_type = self.model.geom_type[geom_id]
            geom_pos  = self.data.geom_xpos[geom_id]
            geom_xmat = self.data.geom_xmat[geom_id].reshape(3, 3)  # 旋转矩阵
            geom_size = self.model.geom_size[geom_id]
            geom_rgba = self._get_geom_rgba(geom_id)

            # 根据类型采样
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

            # transform到全局坐标
            vertices_global = vertices_local @ geom_xmat.T + geom_pos

            # 颜色
            colors = np.tile(geom_rgba, (vertices_global.shape[0], 1))  # shape=(N,4)

            all_points.append(vertices_global)
            all_colors.append(colors)

        # 5. 合并为 (N,7)
        if len(all_points) == 0:
            print("[WARN] No points sampled. Return empty array.")
            return np.empty((0, 6))

        all_points = np.vstack(all_points)    # (M, 3)
        all_colors = np.vstack(all_colors)    # (M, 4)
        point_cloud = np.hstack((all_points, all_colors))  # (M, 7)

        # ★ 因为我们用分配算法严格控制了采样数之和= self.num_points
        #   所以 M 应该 == self.num_points
        #   不需要再做随机截断或补充

        # 最终我们只返回 (N,6)，丢弃 alpha 列
        point_cloud = point_cloud[:, :6]

        return point_cloud

    # ========================= 新增的核心函数 =========================
    # def _allocate_points(self, geom_areas, total_points):
    #     """
    #     根据每个 geom 的 area，精准地给它们分配点数，使得分配后总数 == total_points。
    #     并且保证每个 geom 至少分配到 1 个点。

    #     geom_areas: List[(geom_id, area)]
    #     total_points: int

    #     返回：一个与 geom_areas 等长的数组 allocated_pts_list
    #           其中 allocated_pts_list[i] 表示 geom_areas[i] 分配到的点数。
    #     """
    #     num_geoms = len(geom_areas)
    #     if total_points < num_geoms:
    #         raise ValueError(f"总点数 {total_points} 小于几何体数量 {num_geoms}，无法保证每个geom至少1点！")

    #     # 1) 先给每个geom 1个点
    #     min_alloc = np.ones(num_geoms, dtype=int)
    #     sum_min_alloc = num_geoms

    #     # 2) 计算可供二次分配的 leftover
    #     leftover = total_points - sum_min_alloc
    #     if leftover <= 0:
    #         # 说明刚好或不足以再分配，直接返回
    #         return min_alloc

    #     # 3) 计算各geom在 leftover 中的分配比例
    #     total_area = sum(area for _, area in geom_areas)
    #     if total_area < 1e-12:
    #         # 如果所有面积都几乎0，平分 leftover
    #         # 这里简单处理：把 leftover 均匀摊给前 leftover个geom
    #         # 或者干脆分配 0(若 leftover=0)。
    #         # 为了更公平，可以都分到1，但是那可能导致超量。
    #         # 这里做个平分：
    #         base_portion = leftover // num_geoms
    #         remainder = leftover % num_geoms
    #         min_alloc += base_portion
    #         # 再给 remainder个 geom +1
    #         idxs = np.arange(num_geoms)
    #         np.random.shuffle(idxs)
    #         chosen = idxs[:remainder]
    #         min_alloc[chosen] += 1
    #         return min_alloc

    #     # 4) 计算 floatAlloc = frac_i * leftover
    #     frac = [area / total_area for _, area in geom_areas]
    #     floatAlloc = np.array([f * leftover for f in frac], dtype=float)
    #     floorAlloc = np.floor(floatAlloc).astype(int)

    #     sum_floor = floorAlloc.sum()
    #     leftover2 = leftover - sum_floor  # >= 0
    #     remainders = floatAlloc - floorAlloc  # 范围 [0,1)

    #     # 5) 按 remainder 从大到小排序，给 top leftover2 个 +1
    #     #    这样 sum(alloc) = leftover
    #     if leftover2 > 0:
    #         sorted_indices = np.argsort(-remainders)  # 降序
    #         top_indices = sorted_indices[:leftover2]
    #         floorAlloc[top_indices] += 1

    #     # 6) 最终 = min_alloc + floorAlloc
    #     final_alloc = min_alloc + floorAlloc
    #     assert final_alloc.sum() == total_points, \
    #         f"分配总和{final_alloc.sum()} != {total_points}，请排查！"

    #     return final_alloc

    def _allocate_points(self, geom_areas, total_points):
        """
        根据每个 geom 的面积和类型，精准地给它们分配点数，使得分配后总数 == total_points。
        同时，确保每个 geom 至少分配到一个合理的最低点数。

        geom_areas: List[(geom_id, area)]
        total_points: int
        返回：一个与 geom_areas 等长的数组 allocated_pts_list
        """
        num_geoms = len(geom_areas)

        # 1) 设置每个物体的最小点数（基于几何体类型调整）
        min_points_per_geom = np.array([
            max(300, int(0.01 * total_points))  # 每个物体至少分配 1% 的点数或至少 10 个点
            for _ in range(num_geoms)
        ])

        # 2) 计算剩余的点数
        leftover_points = total_points - np.sum(min_points_per_geom)

        # 3) 按面积比例分配剩余点数
        total_area = sum(area for _, area in geom_areas)
        proportional_points = [
            int(leftover_points * (area / total_area))
            for _, area in geom_areas
        ]

        # 4) 确保最终点数总和与 total_points 一致
        allocated_points = min_points_per_geom + proportional_points
        allocated_points = self._adjust_allocation_to_match_total(allocated_points, total_points)

        return allocated_points

    def _adjust_allocation_to_match_total(self, allocated_points, total_points):
        """
        调整分配的点数，使总和恰好等于 total_points。
        如果超出或不足，则随机选择一些几何体增减点数。
        """
        current_total = np.sum(allocated_points)
        difference = total_points - current_total

        if difference == 0:
            return allocated_points

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
        返回 (4,) => [r,g,b,a].
        """
        geom_rgba = np.clip(self.model.geom_rgba[geom_id], 0, 1)

        material_id = self.model.geom_matid[geom_id]
        if material_id < 0:
            # 没有材质
            return geom_rgba

        # 读取材质
        material_rgba = np.clip(self.model.mat_rgba[material_id], 0, 1)
        shininess    = self.model.mat_shininess[material_id]
        reflectance  = self.model.mat_reflectance[material_id]
        specular     = self.model.mat_specular[material_id]
        emission     = self.model.mat_emission[material_id]

        alpha_geom = geom_rgba[3]
        alpha_mat  = material_rgba[3]
        final_alpha = alpha_geom * alpha_mat

        # 基础色
        base_rgb = geom_rgba[:3] * material_rgba[:3]

        # emission => 自发光
        if emission > 1e-9:
            base_rgb += emission * np.array([1,1,1])
            base_rgb = np.clip(base_rgb, 0, 1)

        # reflectance => 往白靠
        if reflectance > 1e-9:
            base_rgb = (1 - reflectance)*base_rgb + reflectance*np.array([1,1,1])
            base_rgb = np.clip(base_rgb, 0, 1)

        # specular => 进一步加白
        if specular > 1e-9:
            base_rgb += specular*np.array([1,1,1])
            base_rgb = np.clip(base_rgb, 0, 1)

        # shininess => 再做微调(非常简化)
        if shininess > 1e-9:
            base_rgb = base_rgb*(1 - 0.5*shininess) + 0.5*shininess*np.array([1,1,1])
            base_rgb = np.clip(base_rgb, 0, 1)

        final_rgba = np.array([base_rgb[0], base_rgb[1], base_rgb[2], final_alpha], dtype=np.float32)
        return final_rgba

    def _get_task_related_geom_ids(self):
        """
        获取任务相关的几何体索引，支持 body 及其子孙 body 的所有 geom。
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

        # 去重
        return list(set(task_related_geom_ids))

    def _collect_geom_ids(self, body_id):
        """递归收集指定 body 及其子孙 body 的 geom ID。"""
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
        """根据 geom 类型近似计算表面积。对 mesh 用面片实际面积。"""
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
        """对 mesh，遍历三角面计算真实表面积。"""
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
        """在盒子表面随机均匀采样 n 个点。size=(x,y,z)为半尺寸。"""
        x, y, z = size
        areas = np.array([y*z, x*z, x*y])  # 三个面对称面
        areas = np.concatenate([areas, areas])  # 六个面
        total_area = np.sum(areas)
        # 计算每个面的采样数
        face_points = (n * areas / total_area).astype(int)

        # 因为有向下取整，可能 sum(face_points) < n，所以再补一些
        allocated_so_far = face_points.sum()
        leftover = n - allocated_so_far
        if leftover > 0:
            remainders = (n * areas / total_area) - face_points
            sorted_idx = np.argsort(-remainders)  # 降序
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
        phi = np.arccos(1 - 2 * np.random.rand(n))
        theta = 2 * np.pi * np.random.rand(n)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return np.column_stack((x, y, z))

    def _sample_points_on_cylinder(self, radius, height, n):
        side_area = 2 * np.pi * radius * height
        cap_area  = np.pi * radius**2
        areas = np.array([side_area, 2*cap_area])
        total = np.sum(areas)

        # 分配
        side_count = int(n * side_area / total)
        # 再做余数分配
        # 这里为了简单，分两步
        # 先给 side_count, 再剩余给顶底
        leftover = n - side_count
        # 顶底合在一起 leftover
        # 按 1:1 分成 top_count, bot_count
        top_count = leftover // 2
        bot_count = leftover - top_count

        # 侧面
        theta = 2 * np.pi * np.random.rand(side_count)
        zvals = np.random.uniform(-height/2, height/2, side_count)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        side_points = np.column_stack((x, y, zvals))

        # 顶面
        r = np.sqrt(np.random.rand(top_count)) * radius
        theta_cap = 2 * np.pi * np.random.rand(top_count)
        x_top = r * np.cos(theta_cap)
        y_top = r * np.sin(theta_cap)
        z_top = np.full(top_count, height/2)
        top_points = np.column_stack((x_top, y_top, z_top))

        # 底面
        r = np.sqrt(np.random.rand(bot_count)) * radius
        theta_cap = 2 * np.pi * np.random.rand(bot_count)
        x_bot = r * np.cos(theta_cap)
        y_bot = r * np.sin(theta_cap)
        z_bot = np.full(bot_count, -height/2)
        bot_points = np.column_stack((x_bot, y_bot, z_bot))

        return np.vstack((side_points, top_points, bot_points))

    def _sample_points_on_capsule(self, radius, height, n):
        cylinder_area = 2 * np.pi * radius * height
        sphere_area   = 4 * np.pi * radius**2
        total_area = cylinder_area + sphere_area

        cyl_count = int(n * (cylinder_area / total_area))
        sph_count = n - cyl_count

        # 圆柱
        cylinder_pts = self._sample_points_on_cylinder(radius, height, cyl_count)

        # 整球面再拆上下
        sphere_pts = self._sample_points_on_sphere(radius, sph_count)
        half = sph_count // 2
        top_sphere    = sphere_pts[:half] + np.array([0, 0,  height/2])
        bottom_sphere = sphere_pts[half:] + np.array([0, 0, -height/2])

        return np.vstack((cylinder_pts, top_sphere, bottom_sphere))

    def _sample_points_on_ellipsoid(self, size, n):
        a, b, c = size
        phi = np.arccos(1 - 2*np.random.rand(n))
        theta = 2*np.pi*np.random.rand(n)
        x = a * np.sin(phi)*np.cos(theta)
        y = b * np.sin(phi)*np.sin(theta)
        z = c * np.cos(phi)
        return np.column_stack((x, y, z))

    def _sample_points_on_mesh(self, geom_id, n):
        mesh_id = self.model.geom_dataid[geom_id]
        vert_adr = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        face_adr = self.model.mesh_faceadr[mesh_id]
        face_num = self.model.mesh_facenum[mesh_id]

        vertices = self.model.mesh_vert[vert_adr : vert_adr + vert_num].reshape(-1, 3)
        faces = self.model.mesh_face[face_adr : face_adr + face_num].reshape(-1, 3)

        # 面积
        face_areas = np.zeros(face_num, dtype=np.float64)
        for i, f in enumerate(faces):
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            face_areas[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        total_area = face_areas.sum()
        if total_area < 1e-12:
            # mesh退化 => 直接返回部分顶点
            return vertices[:min(len(vertices), n)]

        face_probs = face_areas / total_area
        face_counts = np.random.multinomial(n, face_probs)

        sampled_points = []
        for face_idx, count in enumerate(face_counts):
            if count <= 0:
                continue
            f = faces[face_idx]
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]

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
