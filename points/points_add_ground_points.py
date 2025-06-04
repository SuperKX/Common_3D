
import numpy as np
import open3d as o3d
from pypcd import pypcd
from scipy.spatial import cKDTree
import points_search
import points_common


# 点云拟合地面点
def min_neighbor_height(input_cloud, search_radius = 2,thresh=0.2):
    '''
    修改输入点云高程，将其高度接搜索半径范围（2D平面）内的最低点。业务上用来点云模拟地面点。
    说明：
        0）2d平面内搜索，查找xoy平面内最近点，而非空间上最近点。避免高程差过大导致的检索点错误。
        1）thresh控制阈值，当搜索范围内的最低点高度，与查询点高度小于阈值，则不再更新，以保留查询点附近特征。
        2）梯度下降进行逐步逼近，一方面防止一刀切拉平导致的过于死板，另一方面防止一次性处理后仍然存在遗漏点。
        3）使用两个阈值来控制梯度下降：
            1））距离阈值rate_dis，查询点与最低点的水平距离越近，更多的使用最低点高程。
            2））剃度下降阈值rate_2，加速剃度下降，不使用存在下降太慢的问题。
    Args:
        input_cloud: 输入点云 list(3，2)
        search_radius: 搜索半径
        thresh: 阈值，附近搜索点与当前点小于改值，则不更新
    Returns: 修改输入值input_cloud！
    '''
    # 计算二维kdtree
    input_cloud_2d = np.array(input_cloud.copy())
    input_cloud_2d[:, 2] = 0

    pcd_back_grid = o3d.geometry.PointCloud()
    pcd_back_grid.points = o3d.utility.Vector3dVector(input_cloud_2d)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_back_grid)  # 只用来找索引
    # 增加嵌套循环，更新所有数据。
    is_refreshed = True  # 是否需要更新
    while is_refreshed:
        is_refreshed = False
        input_cloud_np = np.array(input_cloud)
        for i, query_point in enumerate(input_cloud):
            # 直接使用半径搜索替代KNN搜索
            query_point_2d = query_point.copy()
            query_point_2d[2] = 0
            [k, idx, distances] = pcd_tree.search_radius_vector_3d(query_point_2d, search_radius)  # 注意返回的是平方距离！！
            if k > 0:  # 确保找到了邻居点
                neighbor_points = input_cloud_np[idx]
                min_height = np.min(neighbor_points[:, 2])
                if abs(min_height-input_cloud_np[i, 2]) > thresh:
                    is_refreshed = True
                    # 算法1：直接更新
                    # query_point[2] = min_height
                    # 算法2：渐变更新
                    min_index = np.argmin(neighbor_points[:, 2])  # 获取第一个最小值的索引
                    p_min = input_cloud[idx[min_index]]  # 最小值的点
                    rate_dis = distances[min_index]**0.5 / search_radius  # 到最小点水平距离与设定距离的比值，越小说明约近，越大约远
                    if rate_dis < 0 or rate_dis>1:
                        print("查寻点超出范围！")
                    rate_2 = 0.5  # 约束2：防止下降太慢，使用权重约束最小点。
                    query_point[2] = min_height+(query_point[2] - min_height)*rate_dis * rate_2
    return



def creat_grid(point_cloud, pix_size):
    '''
    创建每个点的栅格索引.
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

def add_terrain(points, car_bool, pix_size):
    '''
    补充地面点，根据输入的bool剔除车辆点，并补充缺失的地面点
    points 分割后坐标点云 N*3
    car_id 车辆标签
    pix_size 栅格尺寸
    '''
    add_clouds = np.empty((0, 3))
    # add_labels = np.empty((0, 1))
    # 0 建立映射
    car_id = np.where(car_bool)[0]  # 索引
    back_id = np.where(~car_bool)[0]  # 索引
    # car_idx_map = {oldidx:newidx for newidx,oldidx in enumerate(car_id)}  # 新旧车辆点
    # 0.1 创建车点云
    car_coord = points[car_id]  # 车辆点
    pcd_car = o3d.geometry.PointCloud()
    pcd_car.points = o3d.utility.Vector3dVector(car_coord)
    # 0.2 创建背景点
    back_coord = points[back_id]  # 背景点
    pcd_back = o3d.geometry.PointCloud()
    # 2d点建立kdtree(防止近邻点不符合期待)
    back_coord_2d = back_coord.copy()
    back_coord_2d[:, 2] = 0
    pcd_back.points = o3d.utility.Vector3dVector(back_coord_2d)
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

       # 2 近邻点搜索（2D）
        search_num = 20
        around_point_idx = []
        for query_point in car_i_coord:
            [k, idx, distances] = pcd_tree.search_knn_vector_3d(query_point, search_num)
            around_point_idx.extend(idx)
        # 获取裁剪的背景点
        around_point_idx = list(set(around_point_idx))  # 去除重复索引
        if len(around_point_idx) ==0:
            continue
        back_cloud_seg = back_coord[around_point_idx]  # 裁剪的背景点坐标
        if False:  # 写出最近背景点
            output_path = r'/home/xuek/桌面/TestData/input/车辆补点20250529/test/背景' + str(i) + '.pcd'
            labels_temp = np.array([0]*len(around_point_idx))
            pcd_output = np2pcd(back_cloud_seg, labels_temp)
            pcd_output.save_pcd(output_path, compression='binary')  # 保存为 ASCII 格式
            continue

        # 2 栅格化
        # 2.1 合并车辆、周围背景点
        grid_cloud = np.vstack((car_i_coord, back_cloud_seg))
        # 2.2 grid构造
        grid_id_np, min_coord, cell_size = creat_grid(grid_cloud, pix_size)  # 拿车辆点和外部点构造网格
        # 2.3 计算补点范围xoy：所有车辆点栅格，去除存在更低背景点的栅格
        # 1）计算背景点占用栅格
        grid_id_np_back = np.floor((back_cloud_seg[:, :2] - min_coord) / pix_size).astype(int)  # 背景点在grid中2d坐标 [[2 13], [4 5]]
        grid_set_exist_back = set(tuple(row) for row in grid_id_np_back)  # 所有背景点grid 去重复
        grid_back_points = []  # 背景栅格点（最低高程）
        back_dict = dict()  # 记录grid坐标对应的高程
        for pair in grid_set_exist_back:
            a = np.where(np.all(grid_id_np_back == np.array(pair), axis=1))[0]  # 所有在pair的grid中的点序号
            b = back_cloud_seg[a]  # 格子中点坐标
            h_min = min(b[:,2])  # grid中最小点高程
            back_dict[pair] = h_min
            grid_3d = np.array([(pair[0]+0.5) * pix_size[0] + min_coord[0], (pair[1]+0.5)* pix_size[1] + min_coord[1], h_min])  # 构造grid点坐标
            grid_back_points.append(grid_3d)
        if False:  # 写出最近背景点
            output_path = r'/home/xuek/桌面/TestData/input/车辆补点20250529/test/1-栅格-背景/栅格—back' + str(i) + '.pcd'
            labels_temp = np.array([0]*len(grid_back_points))
            pcd_output = np2pcd(grid_back_points, labels_temp)
            pcd_output.save_pcd(output_path, compression='binary')  # 保存为 ASCII 格式
            continue
        # 2）计算车辆grid点（同背景）
        grid_id_np_car = np.floor((car_i_coord[:, :2] - min_coord) / pix_size).astype(int)  # 背景点在grid中2d坐标 [[2 13], [4 5]]
        grid_set_exist_car = set(tuple(row) for row in grid_id_np_car)  # 所有背景点grid 去重复
        grid_car_points = []
        car_dict = dict()
        for pair in grid_set_exist_car:
            a = np.where(np.all(grid_id_np_car == np.array(pair), axis=1))[0]  # 所有在pair的grid中的点序号
            b = car_i_coord[a]  # 格子中点坐标
            h_min = min(b[:, 2])  # grid中最小点高程
            car_dict[pair] = h_min
            grid_3d = np.array([(pair[0]+0.5) * pix_size[0] + min_coord[0], (pair[1]+0.5) * pix_size[1] + min_coord[1], h_min])  # 构造grid点坐标
            grid_car_points.append(grid_3d)
        if False:  # 写出车辆栅格点
            output_path = r'/home/xuek/桌面/TestData/input/车辆补点20250529/test/3-栅格-car/栅格—car' + str(i) + '.pcd'
            labels_temp = np.array([0] * len(grid_car_points))
            pcd_output = np2pcd(grid_car_points, labels_temp)
            pcd_output.save_pcd(output_path, compression='binary')  # 保存为 ASCII 格式
            continue
        # 3）写出需要补充点的栅格
        grid_need = []  # 需要写出的grid
        for pair in grid_set_exist_car:
            if pair in grid_set_exist_back and back_dict[pair]<car_dict[pair]:
                continue
            grid_need.append(np.array([(pair[0]+0.5) * pix_size[0] + min_coord[0], (pair[1]+0.5) * pix_size[1] + min_coord[1], car_dict[pair]]) )
        # 优化1：过少点直接不计算
        if len(grid_need)<30: # 面积0.2*0.2*num
            continue
        if False:  # 写出需要补全点
            output_path = r'/home/xuek/桌面/TestData/input/车辆补点20250529/test/4-栅格-补充/栅格—need' + str(i) + '.pcd'
            labels_temp = np.array([0] * len(grid_need))
            pcd_output = np2pcd(grid_need, labels_temp)
            pcd_output.save_pcd(output_path, compression='binary')  # 保存为 ASCII 格式
            continue
        # 3 参考点优化
        # 参考点：补充点参考的点，来自于周边环境点
        # 3.1 更新背景点范围（xoy）
        idx_find = points_search.refresh_neighbor(grid_back_points, grid_need, search_radius=1)
        back_new = [grid_back_points[i] for i in idx_find]
        if False:  # 写出最近背景点
            output_path = r'/home/xuek/桌面/TestData/input/车辆补点20250529/test/5-1背景点筛选/51背景xy' + str(i) + '.pcd'
            labels_temp = np.array([0]*len(back_new))
            pcd_output = np2pcd(back_new, labels_temp)
            pcd_output.save_pcd(output_path, compression='binary')  # 保存为 ASCII 格式
            continue
        # 3.2 更新背景点高程（高度z）
        min_neighbor_height(back_new, search_radius=2, thresh=0.2)
        if False:  # 写出最近背景点
            output_path = r'/home/xuek/桌面/TestData/input/车辆补点20250529/test/5-2背景点高程_2/52背景z' + str(i) + '.pcd'
            labels_temp = np.array([0]*len(back_new))
            pcd_output = np2pcd(back_new, labels_temp)
            pcd_output.save_pcd(output_path, compression='binary')  # 保存为 ASCII 格式
            continue
        # 4 更新补充点
        # 4.1 计算补充点高程（来自最近邻居点）
        grid_need_np = np.array(grid_need)
        grid_back_points_np = np.array(grid_back_points)
        if grid_need_np.size < 5:
            continue
        idxss = points_search.find_nearest_ref_points(grid_back_points_np[:,:2], grid_need_np[:,:2])
        height = grid_back_points_np[idxss][:,2]  # 取出高程值
        add_cloud = np.hstack((grid_need_np[:,:2],height[:,np.newaxis]))
        if False:  # 写出需要补全点
            output_path = r'/home/xuek/桌面/TestData/input/车辆补点20250529/test/6-1补充点/61补充点' + str(i) + '.pcd'
            labels_temp = np.array([0] * len(add_cloud))
            pcd_output = np2pcd(add_cloud, labels_temp)
            pcd_output.save_pcd(output_path, compression='binary')  # 保存为 ASCII 格式
            continue
        # 4.2 优化：平滑处理
        # 开启平滑优化
        if True:
            radius = 1  # 平滑范围
            add_cloud = points_search.moving_average_smoothing(add_cloud, radius)
        if False:  # 写出需要补全点
            output_path = r'/home/xuek/桌面/TestData/input/车辆补点20250529/test/6-2补充点-平滑/62补充点-平滑' + str(i) + '.pcd'
            labels_temp = np.array([0] * len(add_cloud))
            pcd_output = np2pcd(add_cloud, labels_temp)
            pcd_output.save_pcd(output_path, compression='binary')  # 保存为 ASCII 格式
            continue
        add_clouds = np.vstack((add_clouds, add_cloud))
    return add_clouds #, add_labels


def add_terrain_func_test(input_file, out_file):
    '''
    用于测试 函数add_terrain
    Args:
        input_cloud: 输入点云地址（已经包含标签）
        out_folder: 写出地址（目前没用）
    Returns:
    '''
    # 计算补充点
    pcd_input = pypcd.PointCloud.from_path(input_file)
    points = np.vstack([pcd_input.pc_data['x'], pcd_input.pc_data['y'], pcd_input.pc_data['z']]).T  # nd(N,3)
    labels = pcd_input.pc_data['label'].astype(np.float32)
    car_bool = labels == 1
    pix_size = (0.2, 0.2)  # 栅格大小
    add_cloud = add_terrain(points, car_bool, pix_size)
    add_cloud = add_cloud.astype(np.float32)
    add_label = np.full(add_cloud.shape[0], 3)

    # 删除车点
    if True:
        points = points[~car_bool]
        labels = labels[~car_bool]

    # 合并点
    labels = np.concatenate([labels, add_label])  # 一维
    points = np.vstack([points, add_cloud])

    # 写出
    pcd_output = points_common.ndarray_2_pypcd_points(points, label=labels)
    pcd_output.save_pcd(out_file, compression='binary')  # 保存为 ASCII 格式
    return

if __name__ == "__main__":
    # test测试车辆剔除后补地面
    if True:
        input_cloud = r'/home/xuek/桌面/TestData/input/车辆补点20250529/zhongtian/seged.pcd'
        out_folder = r'/home/xuek/桌面/TestData/input/车辆补点20250529/zhongtian/seged_0604.pcd'
        add_terrain_func_test(input_cloud, out_folder)

