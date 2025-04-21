import numpy as np
import open3d as o3d  # xuek2025.03.25

from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.spatial import cKDTree

# 创建栅格
def creat_grid(point_cloud, pix_size=0.2):
    '''
    创建每个点的栅格索引.
    point_cloud 输入的点云坐标
    pix_size    grid尺寸（1、2维）   [0.2,0.3] 或 0.2
    return:
        grid_coords 输入点云每个点对应的栅格索引，list[tuple(2),]
        min_coord   真实坐标最小值 【0.13，0.32】
        cell_size   栅格尺寸 tuple(2)  [234,128]
    '''
    # 1 计算包围盒
    min_coord, max_coord = np.min(point_cloud, axis=0)[:2], np.max(point_cloud, axis=0)[:2]
    # 2 计算每个维度的范围
    ranges = max_coord - min_coord
    # 3 计算每个栅格的大小
    cell_size = ranges / pix_size
    import math
    cell_size = tuple(math.ceil(num) for num in cell_size)
    # 4 计算每个点的栅格索引
    grid_coords = np.floor((point_cloud[:, :2] - min_coord) / pix_size).astype(int)

    return grid_coords, min_coord, cell_size

# 一个点云中找另一个点云最近点
def find_nearest_ref_points(known_points, interpolate_points):
    """
    【2d输入】在一个点云中，找另一个点云的二维最近点，返回最近点的索引
    :param known_points: 已知的二维点列表，形状为 (n, 2) 的数组
    :param interpolate_points: 需要插值的点列表，形状为 (m, 2) 的数组
    :return: 每个需要插值的点对应的最近参考点的索引，形状为 (m,) 的数组
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

def convex_hull_from_3dcloud(coords_np):
    '''
    输入3d点云坐标，写出在投影面上的凸包（1个）
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

#点云平滑处理
def moving_average_smoothing(point_cloud, radius):
    '''
    点云平滑处理：输入点云，输出高度平滑处理后的点云，解决点云分层问题。
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


# 计算剔车辆后的地面补洞点
def add_terrain(points, car_bool, pix_size=0.2, bool_return_color=False):
    '''
    输入点云，及车辆点索引，计算剔除后补充点
    points 分割后坐标点云 ndarray（n,3）
    car_bool 车辆标签的布尔 ndarray（n,）
    pix_size 补充点的栅格尺寸
    bool_return_color 是否返回颜色（每个车辆一个补点颜色）
    返回 add_clouds 补充点坐标
    '''
    add_clouds = np.empty((0, 3))
    if bool_return_color:
        add_labels = np.empty((0, 1))
    # 0 建立映射
    car_id = np.where(car_bool)[0]  # 索引
    back_id = np.where(~car_bool)[0]  # 索引
    # 0.1 创建车点云
    car_coord = points[car_id]  # 车辆点
    pcd_car =o3d.geometry.PointCloud()
    pcd_car.points = o3d.utility.Vector3dVector(car_coord)
    # 0.2 创建背景点
    back_coord = points[back_id]  # 背景点
    pcd_back =o3d.geometry.PointCloud()
    pcd_back.points = o3d.utility.Vector3dVector(back_coord)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_back)

    # 1 聚类
    eps = 1  # 同一聚类中最大点间距
    min_points = 20  # 有效聚类的最小点数
    labels = np.array(pcd_car.cluster_dbscan(eps, min_points, print_progress=True))
    max_label = labels.max()  # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
    for i in range(0, max_label + 1):  # 逐个聚类处理
        # 当前车
        car_i_idx = np.where(labels == i)[0]  # cari索引-新索引
        car_i_coord = car_coord[car_i_idx]  # cari 坐标

        # 3 裁剪
        thresh_height = 0.5  # 车量聚类,裁减0.5m以下的点
        car_i_minz = np.min(car_i_coord, axis=0)[2]  # 最低点
        idx_i = np.where(car_i_coord[:,2] <= car_i_minz+thresh_height)[0]  # 裁减后的车点云索引
        car_i_coord_cut =car_i_coord[idx_i]  # 车裁剪后底部坐标
        if len(car_i_coord_cut) <5:  # 太少点，不补
            continue

        # 4 近邻点搜索
        search_num = 3
        around_point_idx = []
        for query_point in car_i_coord_cut:
            [k, idx, distances] = pcd_tree.search_knn_vector_3d(query_point, search_num)
            # 4.1 优化：太远点去除
            idx = np.array(idx)[np.array(distances) < 5]  # 太远的点去掉
            if idx.size==0:
                continue
            # 4.2 优化：如果车的点远高于邻近点，剔除
            if query_point[2]> np.max(back_coord[idx])+1:
                continue
            around_point_idx.extend(idx)
        # 获取裁剪的背景点
        around_point_idx = list(set(around_point_idx))  # 去除重复索引
        if len(around_point_idx) ==0:
            continue
        back_cloud_seg = back_coord[around_point_idx]  # 裁剪的背景点坐标

        # 5 背景点处理:裁剪\构造轮廓
        # 5.1 裁剪
        # back_i_minz = np.min(back_cloud_seg, axis=0)[2]  # 最低点
        back_seg_i = np.where(back_cloud_seg[:, 2] <= car_i_minz + thresh_height)[0]  # 以车的最低点为参考
        back_cloud_seg_cut = back_cloud_seg[back_seg_i]  # 裁剪后背景点
        if len(back_cloud_seg_cut) < 3:  # 点数量过少也不要
            continue

        # 5.2 凸包轮廓
        grid_cloud = np.vstack((car_i_coord_cut, back_cloud_seg_cut))
        hull_path = convex_hull_from_3dcloud(grid_cloud)

        # 6 构造grid 补充点
        # 6.1 grid构造
        grid_id_np, min_coord, cell_size = creat_grid(grid_cloud, pix_size)  # 拿车辆点和外部点构造网格
        # 背景点索引
        grid_id_np_back = np.floor((back_cloud_seg_cut[:, :2] - min_coord) / pix_size).astype(int)
        grid_set_exist = set(tuple(row) for row in grid_id_np_back)  # 所有grid中已经存在的-背景点
        # 6.2 计算需要补充的grid点(x,y)
        grid_list = [(x,y) for x in range(cell_size[0]) for y in range(cell_size[1])]
        grid_set_empty = set(grid_list)-grid_set_exist  # 需要补充的
        if len(grid_set_empty) ==0:
            continue
        grid_nd_empty = np.array(list(grid_set_empty))  # 二维
        coord_2d_match = grid_nd_empty*pix_size+min_coord  # 真实值
        coord2d_inside = hull_path.contains_points(coord_2d_match)  # 真实的二维点
        coord_2d_match = coord_2d_match[coord2d_inside]

        # 7 最近临求解
        if coord_2d_match.size < 5:  # 少于5个点不补充
            continue
        idxss = find_nearest_ref_points(back_cloud_seg_cut[:,:2], coord_2d_match)
        height = back_cloud_seg_cut[idxss][:,2]  # 取出高程值
        add_cloud = np.hstack((coord_2d_match,height[:,np.newaxis]))
        add_clouds = np.vstack((add_clouds,add_cloud))
        if bool_return_color:
            add_label = np.full((add_cloud.shape[0],1),i)
            add_labels = np.vstack((add_labels,add_label))

    # 8 去除重复点：预设点如果在背景中有接近点，则剔除
    if True:
        nocopy = []
        for j,query_point in enumerate(add_clouds):
            [k, idx, distances] = pcd_tree.search_knn_vector_3d(query_point, 1)
            if distances[0]>0.05:  # 优化：背景点距离在0.3m内，判定不需要补充点
                nocopy.append(j)
            idx = np.array(idx)[np.array(distances) < 0.5]  # 太远的点去掉
            around_point_idx.extend(idx)
        add_clouds = add_clouds[nocopy]
        if bool_return_color:
            add_labels = add_labels[nocopy]

    # 9 开启平滑优化
    if False:
        radius = 2 if pix_size<0.2 else pix_size * 10
        add_clouds = moving_average_smoothing(add_cloud, radius)

    if bool_return_color:
        return add_clouds, add_labels
    else:
        return add_clouds
