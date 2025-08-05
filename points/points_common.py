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
