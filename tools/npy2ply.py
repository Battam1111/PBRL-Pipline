import os
import glob
import numpy as np
import open3d as o3d

def npy_to_ply_with_color(npy_file, ply_file):
    # 读取 .npy 文件
    data = np.load(npy_file)
    
    # 检查数据格式是否为 6 列（XYZ + RGB）
    if data.shape[1] != 6:
        raise ValueError(f"文件 {npy_file} 中的点云数据应包含 6 列：X, Y, Z, R, G, B")

    # 分离 XYZ 和 RGB 数据
    xyz = data[:, :3]  # 前 3 列是 XYZ 坐标
    rgb = data[:, 3:]  # 后 3 列是 RGB 颜色

    # # 将 RGB 值归一化到 [0, 1] 范围
    # rgb = rgb / 255.0

    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    
    # 设置点云的坐标和颜色
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb)

    # 保存为 .ply 文件
    o3d.io.write_point_cloud(ply_file, point_cloud)
    print(f"成功将 {npy_file} 转换为 {ply_file}")

def batch_convert_npy_to_ply(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取输入文件夹中所有 .npy 文件的列表
    npy_files = glob.glob(os.path.join(input_folder, "*.npy"))
    
    if not npy_files:
        print(f"在 {input_folder} 中未找到任何 .npy 文件。")
        return
    
    # 遍历每个 .npy 文件并转换为 .ply
    for npy_file in npy_files:
        # 构造对应的输出 .ply 文件路径
        file_name = os.path.splitext(os.path.basename(npy_file))[0] + ".ply"
        ply_file = os.path.join(output_folder, file_name)
        
        try:
            npy_to_ply_with_color(npy_file, ply_file)
        except Exception as e:
            print(f"转换 {npy_file} 时发生错误：{e}")

# 使用示例：
# 指定输入文件夹路径和输出文件夹路径
input_folder = "visualization"
output_folder = "v-ply"

batch_convert_npy_to_ply(input_folder, output_folder)
