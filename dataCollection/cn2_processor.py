import os
import re
import csv
from itertools import combinations
from typing import Dict, List, Tuple

# -----------------------------
# 配置部分
# -----------------------------
# 根目录：存放多个环境，每个环境有大量 pointcloud_000xxx_viewX.jpg
RENDER_ROOT_DIR = "data/renderPointCloud"

# 文件名正则：示例：pointcloud_000001_view1.jpg
# 解析出 group(1) = "000001" (编号), group(2) = "1" (view id)
FILENAME_REGEX = r"^pointcloud_(\d+)_view(\w+)\.jpg$"

# 输出时，pairs 存放子目录名
PAIRS_SUBDIR = "pairs"

# -----------------------------------------------------
# 核心逻辑函数
# -----------------------------------------------------

def parse_filename(filename: str) -> Tuple[int, str]:
    """
    根据既定规则解析文件名，提取 (index, view) 信息。
    若匹配失败则抛出 ValueError。
    示例：
       pointcloud_000001_view1.jpg -> (1, '1')
       pointcloud_000210_viewfront.jpg -> (210, 'front')
    """
    match = re.match(FILENAME_REGEX, filename)
    if not match:
        raise ValueError(f"文件名 {filename} 不符合正则规则 {FILENAME_REGEX}")
    index_str = match.group(1)
    view_str = match.group(2)
    # index_str 可能是 "000001" 这种带前导零的形式，转 int
    index_val = int(index_str)
    return index_val, view_str


def gather_images_by_view(env_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    扫描指定环境目录下的所有文件，解析出 (index, view)，
    并按 view 将它们分组。
    
    返回格式: {
       "view1": [(1, "pointcloud_000001_view1.jpg"), (2, "..."), ...],
       "view2": [...],
       ...
    }
    注意：我们需要记录完整文件名或文件路径，以便后续输出时引用。
    这里用 (index, filename) 的形式存储。
    """
    view_dict: Dict[str, List[Tuple[int, str]]] = {}

    if not os.path.isdir(env_dir):
        return view_dict

    all_files = os.listdir(env_dir)
    for f in all_files:
        # 只处理 .jpg
        if not f.lower().endswith(".jpg"):
            continue
        full_path = os.path.join(env_dir, f)
        if not os.path.isfile(full_path):
            continue

        try:
            index_val, view_str = parse_filename(f)
        except ValueError:
            # 文件名不合规则跳过或打印警告
            # print(f"警告：{f} 不符合命名规则，已跳过。")
            continue

        if view_str not in view_dict:
            view_dict[view_str] = []
        # 将 (index, 文件名) 加入对应 view 分组
        view_dict[view_str].append((index_val, f))

    # 按 index 排序，保证输出有序
    for v in view_dict:
        view_dict[v].sort(key=lambda x: x[0])

    return view_dict


def generate_pairs_for_each_view(view_dict: Dict[str, List[Tuple[int, str]]]) -> Dict[str, List[Tuple[int, int]]]:
    """
    针对每个 view，基于 (index, filename) 列表做 nC2 组合。
    返回结构: {
       "view1": [ (idx1, idx2), (idx3, idx4), ...],
       "view2": [...],
       ...
    }
    注意：只是返回 index 索引对，如果需要实际文件名对，也可改成 (file1, file2)。
    """
    pairs_dict: Dict[str, List[Tuple[int, int]]] = {}

    for v, items in view_dict.items():
        # items: [(index_val, filename), ...] 
        indices = [it[0] for it in items]  # 只拿 index
        combis = list(combinations(indices, 2))  # nC2
        pairs_dict[v] = combis

    return pairs_dict


def save_pairs_to_csv(env_name: str, env_dir: str, view_dict: Dict[str, List[Tuple[int, str]]], pairs_dict: Dict[str, List[Tuple[int,int]]]) -> None:
    """
    将生成的 pairs 信息以 CSV（或其他）形式存储。
    这里示范：每个 view 一个 CSV，或合并为单个 CSV 都可以。
    
    CSV字段示例：view, idx1, idx2, filename1, filename2
    """
    # 创建 /pairs 子目录
    pairs_dir = os.path.join(env_dir, PAIRS_SUBDIR)
    os.makedirs(pairs_dir, exist_ok=True)

    # 方式1：每个 view 一个 CSV
    for v, pairs_list in pairs_dict.items():
        # CSV 文件名
        csv_path = os.path.join(pairs_dir, f"pairs_view_{v}.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["view", "idx1", "filename1", "idx2", "filename2"])
            
            # 为了获取 filename1, filename2，需要在 view_dict[v] 里按 index 找到对应文件名
            # 先做个辅助 dict: index -> filename
            index_to_filename = { it[0]: it[1] for it in view_dict[v] }

            for (i1, i2) in pairs_list:
                f1 = index_to_filename[i1]
                f2 = index_to_filename[i2]
                writer.writerow([v, i1, f1, i2, f2])
        
        print(f"[{env_name}] 已写出 {csv_path}，共 {len(pairs_list)} 组对。")


def process_one_environment(env_name: str, root_dir: str):
    """
    针对单一环境执行：从目录中解析 -> 分组 -> 生成 pairs -> 存储输出
    """
    env_dir = os.path.join(root_dir, env_name)
    if not os.path.isdir(env_dir):
        print(f"[警告] 环境 {env_name} 不存在目录 {env_dir}, 跳过。")
        return
    
    # 1. 收集并按 view 分组
    view_dict = gather_images_by_view(env_dir)
    if not view_dict:
        print(f"[{env_name}] 未找到有效的 pointcloud_xxx_viewY.jpg 文件，跳过。")
        return

    # 2. 针对每个 view 做两两组合
    pairs_dict = generate_pairs_for_each_view(view_dict)

    # 3. 将 pairs 信息输出到 CSV
    save_pairs_to_csv(env_name, env_dir, view_dict, pairs_dict)


def main():
    """
    主函数：遍历 data/renderPointCloud 下的各个环境文件夹。
    每个环境文件夹里做 parse -> group by view -> 两两组合 -> 存储 pairs。
    """
    if not os.path.isdir(RENDER_ROOT_DIR):
        print(f"[错误] 指定的根目录 {RENDER_ROOT_DIR} 不存在。请检查。")
        return
    
    # 列举所有环境子目录
    env_names = [
        d for d in os.listdir(RENDER_ROOT_DIR)
        if os.path.isdir(os.path.join(RENDER_ROOT_DIR, d))
    ]

    if not env_names:
        print(f"[提示] 根目录 {RENDER_ROOT_DIR} 下没有任何子文件夹，无事可做。")
        return
    
    for env_name in env_names:
        print(f"\n=== 开始处理环境: {env_name} ===")
        process_one_environment(env_name, RENDER_ROOT_DIR)
        print(f"=== 环境 {env_name} 处理完成 ===")


if __name__ == "__main__":
    main()
