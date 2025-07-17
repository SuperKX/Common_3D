import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
def 基础函数说明():
    # 1 构建树结构
    input_cloud = np.random.rand(1000,3)
    pcd_back_grid = o3d.geometry.PointCloud()
    pcd_back_grid.points = o3d.utility.Vector3dVector(input_cloud)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_back_grid)  # 只用来找索引

    # 2 树查询
    query_points = np.random.rand(5,3)
    search_radius = 1
    search_num = 5
    for point in query_points:
        # 1）查询距离
        [k, idx, distances] = pcd_tree.search_radius_vector_3d(point, search_radius)  # 注意返回的是平方距离！！
        # 2）查询数量
        [k, idx, distances] = pcd_tree.search_knn_vector_3d(point, search_num)
    return


# 一个点云中查找另一点云的最近点索引（2D）
def find_nearest_ref_points(known_points, interpolate_points):
    """
    在已知点中，查找插入点云中的最近点（2D平面）索引
    :param known_points: 已知的二维点列表，ndarray(n, 2)
    :param interpolate_points: 需要插值的点列表，ndarray(m, 2)
    :return: 每个需要插值的点对应的最近参考点的【索引】，ndarray(m,)
    """
    num_interpolate = interpolate_points.shape[0]
    nearest_indices = []
    for i in range(num_interpolate):
        # 计算需要插值的点与所有已知点的距离
        distances = np.linalg.norm(known_points - interpolate_points[i], axis=1)
        # 找出距离最近的已知点的索引
        nearest_index = np.argmin(distances)
        nearest_indices.append(nearest_index)
    return np.array(nearest_indices)


# 一个点云，在另一个点云中查找搜索半径内所有点索引（2D）
def refresh_neighbor(points_neighbors, points_tosearch, search_radius=1):
    '''
    在点云1中查找距离点云2半径search_radius的所有点（二维平面查找）
    points_neighbors 搜索的点云
    points_tosearch 被查询的点云
    search_radius 查找半径
    return 筛选后的点的索引
    '''
    back_new_2d = np.array(points_neighbors.copy())  # 筛选车辆周边点，剔除多余背景点（降低对高程干扰）
    back_new_2d[:, 2] = 0
    back_new_pcd = o3d.geometry.PointCloud()
    back_new_pcd.points = o3d.utility.Vector3dVector(back_new_2d)
    back_new_kd = o3d.geometry.KDTreeFlann(back_new_pcd)
    # search_radius = 1  # 最近的1m范围内的点
    around_point_idx = []  # 记录需要的点索引
    for point in points_tosearch:
        point_2d = point.copy()  # 注意是深拷贝！！！
        point_2d[2] = 0
        [k, idx, distances] = back_new_kd.search_radius_vector_3d(point_2d, search_radius)
        around_point_idx.extend(idx)
    idx_find = sorted(list(set(around_point_idx)))
    # points_out = [points_neighbors[i] for i in idx_find]
    return idx_find

# 点云平滑
def moving_average_smoothing(point_cloud, radius):
    '''
    平滑处理搜索半径内所有点
    point_cloud 代处理点云
    radius  搜索半径
    '''
    tree = cKDTree(point_cloud[:, :2])  # 构建二维 KD 树
    smoothed_heights = []
    for point in point_cloud:
        idx = tree.query_ball_point(point[:2], radius)  # 查询半径范围内的点
        neighborhood = point_cloud[idx]
        if len(neighborhood) > 0:
            smoothed_height = np.mean(neighborhood[:, 2])  # 计算平均高程
            smoothed_heights.append(smoothed_height)
        else:
            smoothed_heights.append(point[2])
    point_cloud[:, 2] = np.array(smoothed_heights)
    return point_cloud

