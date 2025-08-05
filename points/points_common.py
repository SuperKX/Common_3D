'''
    点云的公共功能
    ndarray_2_pypcd_points  ndarray转pypcd
'''
import numpy as np
from pypcd import pypcd
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.spatial import cKDTree


# ndarray转pypcd
def ndarray_2_pypcd_points(points, **kwargs):
    '''
    将多个ndarray，转换成pypcd格式点云[接受多个变量读入]
    :param points: 输入点坐标, ndarray(N, 3)
    :param kwargs: 额外的特征，以关键字参数的形式传入。
                   例如： "colors=colors, normals=normals"
                   每个特征都应该是一个 ndarray，且shape为(N, M)，M是特征的维度。(接受(N,)并改为(N,1))
    :return: pcd 返回 pcd 格式标签点云
    注意：
        1）暂时未验证颜色、法向等多维度数据！！
        2）输入变量名称会 写出为标签值！
        3）后续新增一个字典的方法，可以解析变量名

    xuek 2025.04.22
    '''
    num_points = points.shape[0]  # 点数量
    # 构建字段列表
    cloud_data = [points]  # 数据列表
    fields = ['x', 'y', 'z']
    sizes = [4, 4, 4]
    types = ['F', 'F', 'F']
    counts = [1, 1, 1]

    # 处理额外的特征
    for name, feature in kwargs.items():
        if not isinstance(feature, np.ndarray):
            raise ValueError(f"Feature '{name}' 不是ndarray.")
        if feature.shape[0] != num_points:
            raise ValueError(f"Feature '{name}' 数量不对.")

        # 维度修改
        if feature.ndim == 1:
            feature = feature.reshape(-1, 1)  # 将 (n,) 转换为 (n, 1)

        # 将特征添加到数据列表中
        cloud_data.append(feature)

        # 更新元数据信息
        num_components = feature.shape[1] if feature.ndim > 1 else 1 # 特征维数 #第二维度
        fields.append(name)
        sizes.append(4 * num_components)  # 假设是 float32
        types.append('F'*num_components)
        counts.append(num_components)

    # 将所有数据水平堆叠起来
    cloud = np.hstack(cloud_data)
    cloud = cloud.astype(np.float32)  # 需要一致！

    # 2. 构建 PCD 元数据
    metadata = {
        'version': '0.7',
        'fields': fields,
        'size': [item if isinstance(item, int) else item.tolist() for item in sizes], #  数据类型大小 (float32)
        'type': [item if isinstance(item, str) else ''.join(item) for item in types],  # 数据类型 (float)
        'count': [item if isinstance(item, int) else item.tolist() for item in counts],  # 每个字段的计数
        'width': num_points,
        'height': 1,
        'viewpoint': [0, 0, 0, 1, 0, 0, 0],
        'points': num_points,
        'data': 'binary'  # 'ascii'  # 数据存储格式 (ascii 或 binary_compressed)
    }

    # 3. 创建 PCD 对象
    pcd = pypcd.PointCloud(metadata, cloud)
    return pcd

def gridsample_np_points(data, voxel_size=0.1):
    '''
        点云gridsample
        data            ndarray(n,properties)  # 点云，前三维为坐标
        sampled_data    ndarray(n_sub,properties)  # 点云，前三维为坐标
    '''

    coords = data[:,:3]
    # 计算voxel索引并进行采样
    voxel_indices = (coords // voxel_size).astype(int)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    sampled_data = data[unique_indices]  # 采样后点云
    return sampled_data


# 计算三维点的二维凸包
def convex_hull_from_3dcloud(coords_np):
    '''
    输入3d点云坐标，写出在投影面上的凸包（1个）
    coords_np   ndarray(n,3)
    '''
    if coords_np.shape[0] < 3:
        raise ValueError(f'凸包计算错误：点云数量过少')
    hull = ConvexHull(coords_np[:, :2])
    hull_path = Path(coords_np[hull.vertices][:, :2])  # 转轮廓线
    # 写出轮廓
    if False:
        file_path = r'/home/xuek/桌面/PTV3_Versions/ptv3_raser_cpu/ptv3_deploy/scripts/output/test_temp/guangzhou_20240605140342__xx.poly'
        with open(file_path, "w", encoding="utf-8") as file:
            for vertex in hull_path.vertices:
                file.write(f"{vertex[0]:.6f} {vertex[1]:.6f} 0\n")
            file.write(f"\n")
    # 显示轮廓
    if False:
        import matplotlib.pyplot as plt
        # 创建 PathPatch 对象
        from matplotlib.patches import PathPatch
        points = grid_cloud[:, :2]
        patch = PathPatch(hull_path, facecolor='none', edgecolor='red', lw=2)
        # 创建图形和轴
        fig, ax = plt.subplots()
        # 绘制点云
        ax.scatter(points[:, 0], points[:, 1], label='Points')
        # 添加 PathPatch 到轴
        ax.add_patch(patch)
        # 设置图例
        ax.legend()
        # 显示图形
        plt.show()
    return hull_path

def get_plane_inliers(point_cloud, threshold=0.1, max_iterations=1000, min_inliers=3):
    """
    使用RANSAC算法拟合点云中的平面，并返回符合该平面的点（内点）
    参数:
        point_cloud: 输入点云，ndarray(n, 3)
        threshold: 距离阈值，小于此值的点被视为内点
        max_iterations: 最大迭代次数
        min_inliers: 最小内点数量，用于判断是否为有效平面

    返回:
        inlier_points: 符合拟合平面的点，形状为(m, 3)的ndarray
        plane_params: 拟合的平面参数 [a, b, c, d]，对应平面方程ax + by + cz + d = 0
    """
    # 检查输入点云是否有效
    if not isinstance(point_cloud, np.ndarray) or point_cloud.shape[1] != 3:
        raise ValueError("输入点云必须是形状为(n, 3)的ndarray")

    n_points = point_cloud.shape[0]
    if n_points < 3:
        raise ValueError("点云至少需要3个点才能拟合平面")

    best_inliers = []
    best_plane = None

    # RANSAC迭代
    for _ in range(max_iterations):
        # 随机选择3个点
        indices = random.sample(range(n_points), 3)
        p1, p2, p3 = point_cloud[indices]

        # 计算平面方程
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        # 处理共线点的情况（法向量为零向量）
        if np.allclose(normal, [0, 0, 0]):
            continue

        a, b, c = normal
        d = - (a * p1[0] + b * p1[1] + c * p1[2])

        # 计算所有点到平面的距离，确定内点
        inliers = []
        for i in range(n_points):
            x, y, z = point_cloud[i]
            distance = abs(a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
            if distance < threshold:
                inliers.append(i)

        # 更新最佳平面
        if len(inliers) > len(best_inliers) and len(inliers) >= min_inliers:
            best_inliers = inliers
            best_plane = [a, b, c, d]

    # 如果找到有效平面，使用所有内点优化平面参数
    if best_plane is not None and len(best_inliers) > 0:
        # 使用最小二乘法优化
        inlier_points = point_cloud[best_inliers]
        X = np.hstack((inlier_points, np.ones((inlier_points.shape[0], 1))))
        _, _, V = np.linalg.svd(X)
        best_plane = V[-1, :]
        norm = np.linalg.norm(best_plane[:3])
        if norm > 0:
            best_plane /= norm
        return inlier_points, best_plane
    else:
        raise RuntimeError("无法从点云中拟合出有效的平面")
