import warnings
import torch
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import argparse
import glob
import json
import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import open3d as o3d
import pickle
import re
import random
from pypcd import pypcd


def delete_label_cloud(labels, deleClass, *cloud_infor):
    '''
        输入标签 根据deleclass删除不需要的标签(点云顺序不变)
        调用:
            deleClass = [1]  # 要删除的标签
            cloud_infor = (points, intensity)
            labels, points, intensity = delete_label_cloud(labels, deleClass, cloud_infor)
        注意:
        1) 需要处理的信息,以元组的方式传给cloud_infor(coords,color,intensity), 如果数量跟label不一致,则不处理.
    '''
    result = []
    del_idx = np.empty((0, 1))
    classAll = np.unique(labels.astype(int))  # 所有标签
    print(f"写出所有标签{classAll}")
    for class_id in classAll:
        if class_id in deleClass:
            continue
        listi = np.where(labels == class_id)[0]
        if listi.size <= 0:
            continue
        listi = listi[:, np.newaxis]
        del_idx = np.concatenate((del_idx, listi), axis=0)
    # 获取所有保留索引
    del_idx = np.sort(del_idx.reshape(-1).astype(np.int32))
    # 删除cloud_infor中的点
    pts_num = labels.shape[0]
    for array in cloud_infor:
        if array.shape[0] == pts_num:  # 不处理数量不一致的数据
            array = array[del_idx]
        result.append(array)
    return labels, result


# 计算类别比例
def classRates(inputfolder):
    '''
    当前文件夹中,ply数据的各类别点云数量及占比
    '''
    files = os.listdir(inputfolder)  # 获取带后缀文件列表
    for filei in files:
        if filei.split(".")[1] != 'ply':  # and filei.split(".")[1] != 'pcd':
            continue
        inputply = os.path.join(inputfolder, filei)
        _, _, labels = ply_parsing(inputply)
        class_num = 4
        cloudnum = labels.size
        # print(f'{filei.split(".")[0]}, ',end ='')  # 文件名
        print(f'{filei.split(".")[0]} 点总数: {cloudnum};', end='')
        for i in range(class_num):
            num = np.sum(labels == i)
            print(f'类别 {i} 数量: {num}; 占比 {num / cloudnum:.3%};', end='')
        print('')
    return


# 合并点云,并写出
def merge_trans_files(inputfolder, group, output_file, transMtx):
    """
    合并inputfolder 中,group中的文件,并写出到output_file中
    transMtx 为点云变换矩阵,非none时进行变换
    """
    # 1 文件列表
    pcd_files = [os.path.join(inputfolder, groupi) for groupi in group]
    # 获取目录下所有 PCD 文件
    # pcd_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pcd')]
    # pcd_files.sort()  # 可选：按文件名排序

    # 2 初始化一个空的点云对象
    merged_pcd = o3d.geometry.PointCloud()

    # 3 逐个读取并合并点云
    for file in pcd_files:
        pcd = o3d.io.read_point_cloud(file)
        merged_pcd += pcd

    # 4 矩阵变换
    if transMtx != None:
        merged_pcd.transform(transMtx)

    # 5 保存合并后的点云
    o3d.io.write_point_cloud(output_file, merged_pcd)
    return


# pcd文件合并并变换
def pcd_merge_trans(inputfolder,outputfolder,merge_list,ignore_files,transMatrix):
    '''
    brief: 根据merge_list中数量,合并相应数量的相邻点云
    merge_list  根据该列表知识
    ignore_files    文件列表中该文件不处理
    '''
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
        # 1 读取文件列表p
        files = os.listdir(inputfolder)
        # 1.0 排除不处理数据
        for filei in files:
            if filei in ignore_files:  # 删除
                files.remove(filei)
            if filei.split(".")[1] != 'pcd':
                files.remove(filei)
        # 1.1 排序
        def sort_key(file_name):  # 按照数字排序
            numbers = re.findall(r'\d', file_name)
            numbers = "".join(numbers)
            return int(numbers)
        files = sorted(files, key=sort_key)

        for merge_num in merge_list:
            # 2 根据合并数量分组
            list_group = [files[i:i + merge_num] for i in range(0, len(files), merge_num) if
                          i + merge_num <= len(files)]
            # 3 筛选部分数据,其余抛弃
            if selectNum != None and len(list_group) > selectNum:
                list_group = random.sample(list_group, selectNum)
            # 4 读入并合并
            for j, groupi in enumerate(list_group):
                out_filename = str(merge_num) + "_" + str(j) + ".pcd"
                output_file = os.path.join(outputfolder, out_filename)
                merge_trans_files(inputfolder, groupi, output_file, transMatrix)








if __name__ == '__main__':

    # 1\ ply文件解析
    # try:
    #     coords, colors, labels_org = ply_parsing(file_cloud_ply)
    # except ValueError as e:
    #     print(f'发生错误{e}')

    # 2\ 统计数据各类别标签比例
    # inputfolder= r'/home/xuek/桌面/TestData/input/raser_car/train'
    # classRates(inputfolder)

    # # 3\ 点云按照顺序合并(及旋转)
    # inputfolder = r'/home/xuek/桌面/TestData/input/DB_DATA'
    # outputfolder = r'/home/xuek/桌面/TestData/input/DB_DATA_merges'
    # merge_list = [1, 2, 5, 10, 20, 50, 100]  # 合并数据的数量梯度.
    # ignore_files = ['db_0.pcd', 'map_ror.pcd', 'map_down.pcd']  # 不处理文件
    # transMatrix = np.array([[0.107301, 0.926163, -0.361535, 0.0150326],
    #                         [0.0195186, -0.365527, -0.930596, 4.24446],
    #                         [-0.994035, 0.0927971, -0.0572988, 61.772],
    #                         [0, 0, 0, 1]])  # 点云进行变换的矩阵, None 表示不变换
    # selectNum = 20  # 各组合并数据,最多取这么多,其余的不统计  3 None 表示不筛选
    # pcd_merge_trans(inputfolder, outputfolder, merge_list, ignore_files, transMatrix)

    # 4
    # folder = r'/home/xuek/桌面/TestData/output/DB_DATA_merges/dataPreperation'
    # filename = r'DB_DATA_merges_20_4'
    # file_proj_pkl = os.path.join(folder, filename+'.pkl')
    # with open(file_proj_pkl, 'rb') as f:
    #     proj_idx, labels = pickle.load(f)
    # newlabel = labelList[proj_idx]

    # 1 采样点云
    subsamp_pth = r'/home/xuek/桌面/TestData/output/DB_DATA_merges/dataPreperation/DB_DATA_merges_20_4.pth'
    dic_sample = torch.load(subsamp_pth)
    # 2 推理标签
    file_label_npy = r'/home/xuek/桌面/TestData/output/DB_DATA_merges/dataPredict/result/DB_DATA_merges_20_4_pred.npy'
    labelList = np.load(file_label_npy)  # 采样点标签
    # 写出
    file_save_file = r'/home/xuek/桌面/TestData/output/DB_DATA_merges/merges_20_4_sam.pcd'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dic_sample['coord'])

    label_colors = np.array([[0.5, 0.5, 0.5], [0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype='float')
    colors = label_colors[labelList]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file_save_file, pcd)










