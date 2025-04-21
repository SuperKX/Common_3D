import os
import open3d as o3d
import numpy as np


# 合并点云,并写出
def merge_trans_files(file, output_file, transMtx):
    '''
    读入文件,变换矩阵处理后写出.
    注意:
    1) 验证会丢失intensity信息.
    '''
    # 读入
    pcd = o3d.io.read_point_cloud(file)
    # 转
    pcd.transform(transMtx)
    # 写出
    o3d.io.write_point_cloud(output_file, pcd)

    return


if __name__ == '__main__':
    # merge_trans_files 测试
    file = r'/home/xuek/桌面/TestData/input/staticmap_test/staticmap_test_0415/yangjian_tower1_tree_2025-03-21-15-17-04.pcd'
    output_file = r'/home/xuek/桌面/TestData/input/staticmap_test/staticmap_test_0415/yangjian_tower1_tree_2025-03-21-15-17-04-rotation.pcd'
    transMatrix = np.array([[0.953689813614,    0.300771355629,    0.003511849325,    4.327094078064],
    [-0.026559632272,    0.095833897591, - 0.995042920113, - 14.433804512024],
    [-0.299616962671,    0.948869049549,    0.099384188652,    12.330602645874],
    [0,0,0,1]])  # 点云进行变换的矩阵, None 表示不变换

    merge_trans_files(file, output_file, transMatrix)
