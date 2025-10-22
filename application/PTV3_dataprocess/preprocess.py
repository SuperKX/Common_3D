import os
import argparse
import glob
import json
import time

import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import open3d as o3d
from scipy.spatial import KDTree
import pickle
import warnings
import torch
import points.points_io as points_io
import path.path_process as path_process
from points.points_transform import grid_sample_idx

def val_pth(folder_path):
    '''
    验证生成的pth文件是否合格
    注意：
        1）兼容输入的文件名或文件夹名
    para:
        folder_path     遍历地址，写出：

    '''
    # folder_path = r'/home/xuek/桌面/Pointcept/data/bimtwins6Class20250730'
    ext = os.path.splitext(folder_path)[1]
    if ext == '.pth':
        filelist = [folder_path]
    else:
        filelist = path_process.get_files_by_format(folder_path, formats=['.pth'])
    for file in filelist:
        pth_file = file
        print(f"处理文件: {os.path.split(file)[1]}")
        # pth_file = r'/home/xuek/桌面/Pointcept/data/bimtwins6Class20250730/val/40GMGS.pth'
        pth_info = torch.load(pth_file)
        label = pth_info['semantic_gt20']
        unique_labels, counts = np.unique(label, return_counts=True)
        num = label.shape[0]
        print(f"点云数量: {num}")
        for labeli, counti in zip(unique_labels, counts):
            print(f"    标签: {labeli} ,数量: {counti} ,占比: {counti / num:.2%}")

    return

def cloudinfo_to_ptv3dict(cloud_info,generate_normals=True, sample_grid=0.4):
    '''
    输入一个点云信息字典，返回PTV3处理的数据字典
    （并进行将采样、计算点法向）
    注意：
        * 参考：/home/xuek/桌面/PTV3_Versions/Common_3D/application/PTV3_dataprocess/preprocess.py
        * 当前不支持标签读入！！！！，因为o3d
    return
        save_dict, idx_proj
    '''
    # 0 构造返回字典
    save_dict = dict()
    # 1 点云grid sample
    if not 'coords' in cloud_info:
        raise ValueError(f'【错误】输入点云没有坐标信息！')
    idx_sample = grid_sample_idx(cloud_info['coords'], grid_size=0.4)
    # 2 采样后点云构造
    # 2.1 坐标
    coords_sub = cloud_info['coords'][idx_sample]
    save_dict['coord'] = coords_sub
    # 2.2 颜色
    if 'colors' in cloud_info:
        colors_sub = cloud_info['colors'][idx_sample]
        save_dict['color']=colors_sub
    # 2.3 法向
    if generate_normals == True:
        if 'normals' in cloud_info and cloud_info['normals'] != None:
            normals_sub = cloud_info['normals'][idx_sample]
        else:
            radius = 0.5  # 搜索半径
            max_nn = 30  # 邻域内用于估算法线的最大点数
            cloud_o3d_sample = o3d.geometry.PointCloud()
            cloud_o3d_sample.points = o3d.utility.Vector3dVector(colors_sub)
            cloud_o3d_sample.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
            normals_sub = np.asarray(cloud_o3d_sample.normals)
        save_dict['normal'] = normals_sub

    # 2.4 原始标签
    if 'labels' in cloud_info:
        labels_sub = cloud_info['labels'][idx_sample]
    else:  # 构造假标签
        print("没有找到标签！")
        labels_sub = np.zeros((len(coords_sub),), dtype=np.int8)  # 创建一个假的标签值
    save_dict["semantic_gt20"] = labels_sub
    # 写出
    if False:
        cloud_ndarray=np.hstack([coords_sub,colors_sub,labels_sub.reshape(-1, 1)])
        file_out =r'/home/xuek/桌面/TestData/临时测试区/input_34PTY1_out.ply'
        write_ply_file(file_out, cloud_ndarray)
    return save_dict

def labels_change(labels, label_dict):
    '''
    根据字典指引，修改点云标签
    params:
        labels      原始标签 ndarray(n，)
        label_dict  记录新旧点云映射，{新标签1：[旧标签1，就标签2, ...], },ex: label_dict={0:[0,1],1:[2,3]}；未修改标签则置为0
    return：
        labels_new  返回新标签
    '''

    # 1 原始标签信息统计
    unique_orig, counts_orig = np.unique(labels, return_counts=True)
    print(f'原始标签统计：')
    print(f"点云数量: {labels.size}")
    for u, c in zip(unique_orig, counts_orig):
        print(f'    标签：{u} ，数量：{c} ， 比例：{c / labels.size:.2%}')

    # 2 修改标签
    # 2.1 处理已知标签
    labels_new = np.zeros((len(labels)), dtype=np.int8)  # 构造新的标签ndarray（N，）
    changed_label_num = 0
    for new_label,old_label in label_dict.items():  # 逐个新标签
        mask = np.isin(labels, old_label)
        changed_label_num += mask.sum()
        labels_new[mask] = new_label
    if changed_label_num < labels.size:
        print(f'【注意】有 {labels.size - changed_label_num} 点 未修改标签！')

    # 3 新标签统计
    unique_new, counts_new = np.unique(labels_new, return_counts=True)
    print(f'新标签统计：')
    for u, c in zip(unique_new, counts_new):
        print(f'    标签：{u} ，数量：{c} ， 比例：{c / labels.size:.2%}')
    return labels_new

def preprocess_BIMTwins(
        input_file, output_file,
        generate_normals=True, sample_grid=0.4,
        label_dict={},
        val_result_pth=True,
        create_fake_label=False,
):
    '''
    brief:
        预处理：读入原始1个点云数据，生成ptv3训练输入的降采样点云.pth
        注意：
            1）输入是点云地址，写出pth文件地址
            2）label_dict要求如下描述。
    paras:
        input_file          输入原始点云
        output_file         写出pth文件
        generate_normals    是否写出法向
        sample_grid         采样密度
        label_dict          新旧字典映射：{新标签1：[旧标签1，就标签2, ...], },ex: label_dict={0:[0,1],1:[2,3]}；
                                1）未修改标签则置为0，会提示。
                                2）空，表示不修改标签，会提示。
        val_result_pth      是否验证生成的pth文件
        create_fake_label   如果没有标签，则创造假标签
    '''
    # 1 获取点云信息：坐标、颜色、标签
    try:
        coords, colors, labels = points_io.parse_ply_file(input_file)
    except ValueError as e:
        raise ValueError(f'发生错误{e}')
    if not isinstance(coords, np.ndarray):
        raise ValueError(f'没有点')
    elif not isinstance(labels, np.ndarray):
        if not create_fake_label:
            raise ValueError(f'输入数据没有标签！（如需构造虚假标签，修改变量“create_fake_label”）')
        else:
            print(f"【警告】未找到标签信息，此处将构造虚假标签！")
            labels = np.zeros(len(coords), int)

    # 3 标签替换
    if len(label_dict) == 0:
        print("【注意】：没有发生标签修改！")
        new_labels = labels
    else:
        new_labels = labels_change(labels, label_dict)

    # 4 构造cloud_info点云信息
    cloud_info = dict(coords=coords, colors=colors, labels=new_labels)

    # 5 将采样、点云化，处理为ptv3输入格式
    # （注意：当前不再构造kdtree上采样树）
    save_dict = cloudinfo_to_ptv3dict(cloud_info, generate_normals=generate_normals, sample_grid=sample_grid)
    if True:  # train模式需要
        scene_id = os.path.splitext(os.path.split(input_file)[1])[0]
        save_dict['scene_id'] = scene_id
    # 写出采样后的点云的 pth文件
    torch.save(save_dict, output_file)

    if val_result_pth:
        print(f"验证pth数据标签（降采样结果）：",end="")
        val_pth(output_file)
    return

'''
参数说明：
    dataset_root    输入数据（输入时已经按照train\ val\ test\ 分配好）
    output_root     输出数据地址
    parse_normals   是否使用点云法向
'''
if __name__ == "__main__":

    # 预处理数据
    if True:
        # 0 参数修改
        dataset_root = r'/media/xuek/Data210/测试数据/11分类-临时测试：农田草地突出'
        output_root = r'/home/xuek/桌面/TestData/PTV3_data/11分类-临时测试：农田草地突出'
        sample_grid = 0.4  # 采样大小：注意此处写死

        # classlist = [0"background", 1"building", 2"wigwam", 3"car", "vegetation", "farmland", "shed", "stockpiles", "bridge", "pole", "others", "grass"]
        # 1）6分类数据处理 ["background", "building", "car", "vegetation", "farmland", "grass"]
        label_dict_6class = {0: [0, 7, 9, 10],
                             1: [1, 2, 6, 8],
                             2: [3, ],
                             3: [4, ],
                             4: [5, ],
                             5: [11, ]}
        # 2）4分类["background", "building", "vegetation", "farmland"]
        label_dict_4class = {0: [0, 7, 9, 10, 3, 11],
                             1: [1, 2, 6, 8],
                             2: [4, ],
                             3: [5, ]}


        # 1 数据地址
        # 查找所有ply文件
        sceneList = sorted(glob.glob(dataset_root + "/train/*.ply"))
        sceneList += sorted(glob.glob(dataset_root + "/val/*.ply"))
        sceneList += sorted(glob.glob(dataset_root + "/test/*.ply"))

        # 2 创建输出文件地址
        train_output_dir = os.path.join(output_root, "train")
        os.makedirs(train_output_dir, exist_ok=True)
        val_output_dir = os.path.join(output_root, "val")
        os.makedirs(val_output_dir, exist_ok=True)
        test_output_dir = os.path.join(output_root, "test")
        os.makedirs(test_output_dir, exist_ok=True)

        # 3 批量处理
        time_start0 = time.time()
        for input_file in sceneList:
            time_starti = time.time()
            subFoler = os.path.split(os.path.split(input_file)[0])[1]  # 'train'
            scene_id = os.path.splitext(os.path.split(input_file)[1])[0]  # 'JXY'
            output_file = os.path.join(output_root, subFoler, f"{scene_id}.pth")
            print(f"Processing: {scene_id} in {subFoler}")
            preprocess_BIMTwins(
                input_file, output_file,
                generate_normals=True, sample_grid=sample_grid,
                label_dict=label_dict_4class,
                val_result_pth=True,
                create_fake_label=False,
            )
            print(f"场景 {scene_id} 耗时：{time.time()-time_starti}")
            print(" ")
        print(f"预处理 总耗时：{time.time() - time_start0}")